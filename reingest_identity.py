r"""
reingest_identity.py — Supplementary extraction pass for identity & health facts.

Re-ingests ALL sessions for each persona with a focused extraction prompt that
targets personal identity, health/medical, and biographical facts that Graphiti's
primary extraction misses (single-mention conditions, family details, etc.).

This addresses AV3 (stable identity) failures where EXTRACTION_MISSING is the
root cause — facts mentioned casually once that never became edges.

Same playbook as reingest_retractions.py: additive-only second pass, checkpoint
support, no modification of existing edges.

Usage:
    python reingest_identity.py --canary elena       # Test on one persona
    python reingest_identity.py --canary tom          # Test on another
    python reingest_identity.py --all                 # Run all 40 personas
    python reingest_identity.py --range 1 10          # Personas 1-10
    python reingest_identity.py --persona 3_elena     # Single persona by dir name
    python reingest_identity.py --status              # Show checkpoint progress
"""

import argparse
import asyncio
import json
import os
import sys
import time as time_module
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Environment setup (MUST be before graphiti_core import)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
os.environ.setdefault('SEMAPHORE_LIMIT', '8')


def load_env():
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), val)


load_env()

for var in ['OPENAI_API_KEY', 'XAI_API_KEY']:
    if not os.environ.get(var):
        print(f"ERROR: {var} not set. Check your .env file.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Imports (after env is loaded)
# ---------------------------------------------------------------------------

from openai import AsyncOpenAI
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from neo4j import AsyncGraphDatabase

from ingest_lifemembench import (
    get_graphiti_client,
    parse_session_date,
    group_id_for,
    enrich_persona_edges,
    POC_PERSONAS,
    LIFEMEMEVAL_DIR,
    ARTIFACTS_DIR,
    format_time,
)

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = ARTIFACTS_DIR / "reingest_identity_checkpoint.json"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.load(open(CHECKPOINT_FILE, encoding='utf-8'))
    return {'started_at': None, 'personas': {}}


def save_checkpoint(ckpt: dict):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ckpt, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Identity/health extraction instructions
# ---------------------------------------------------------------------------

IDENTITY_HEALTH_EXTRACTION_INSTRUCTIONS = """
Focus EXCLUSIVELY on extracting personal identity and health facts about the user.
These are facts that would still be true years from now — permanent or semi-permanent attributes.

Extract ALL of the following, even if mentioned casually, briefly, or only once:

HEALTH & MEDICAL:
- Diagnoses (physical and mental): ADHD, anxiety, PTSD, diabetes, kidney disease, heart conditions
- Medications: name, dosage if mentioned, what it's for
- Chronic conditions: hearing loss, tinnitus, chronic pain, allergies
- Disabilities or impairments
- Past surgeries or major medical events
- Mental health conditions and treatment

BIOGRAPHICAL & IDENTITY:
- Number of siblings, birth order
- Ethnicity, heritage, cultural background
- Military service, veteran status
- Educational background (degrees, institutions)
- Transgender/gender identity details
- Religious or spiritual identity
- Nationality, immigration status
- Languages spoken
- Age, birth year if mentioned
- Marital/relationship history (married, divorced, widowed)

FAMILY & RELATIONSHIPS:
- Names and relationships of family members (siblings, parents, children, spouse)
- Losses — deaths, estrangements, separations
- Family medical history if mentioned

EXTRACTION RULES:
- Extract as simple edges: (user)-[has/takes/experienced/is]->(fact)
- Include ALL specific details: names, numbers, dosages, dates, durations
- Even if the user mentions it as an aside or in passing — extract it
- "Yeah my doctor put me on levothyroxine for the thyroid thing" → extract: user takes levothyroxine for hypothyroidism
- "I'm the oldest of six" → extract: user has 5 siblings and is the oldest of 6
- "Lost both my brothers in the war" → extract: user lost both brothers in war
- "Been on testosterone for 5 years now" → extract: user is transgender, has been on testosterone for 5 years
- "The tinnitus from my service days" → extract: user has tinnitus from military service
- Do NOT extract preferences, hobbies, work details, or logistics — ONLY identity, health, and biography
- Preserve exact numbers, medication names, condition names, and durations
"""

# Reverse lookup: short_name -> persona_dir
_SHORT_TO_DIR = {v: k for k, v in POC_PERSONAS.items()}

# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------


def load_all_sessions(persona_dir: str) -> list[dict]:
    """Load all session JSON files for a persona, sorted by session_id."""
    sessions_dir = LIFEMEMEVAL_DIR / persona_dir / "sessions"
    if not sessions_dir.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")

    sessions = []
    for session_file in sorted(sessions_dir.glob("session_*.json")):
        with open(session_file, encoding="utf-8") as f:
            sessions.append(json.load(f))

    sessions.sort(key=lambda s: s["session_id"])
    return sessions


def build_episodes(sessions: list[dict], short_name: str) -> list[RawEpisode]:
    """Convert session dicts to RawEpisode objects for add_episode_bulk."""
    episodes = []
    for session in sessions:
        sid = session["session_id"]
        session_type = session.get("type", "unknown")

        content = "\n".join(
            f"{t['role']}: {t['content']}" for t in session["turns"]
        )

        episodes.append(RawEpisode(
            name=f"lmb_{short_name}_s{sid:02d}",
            content=content,
            source_description=(
                f"LifeMemBench {short_name} session {sid} ({session_type})"
            ),
            source=EpisodeType.message,
            reference_time=parse_session_date(session["date"]),
        ))

    return episodes


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------


async def count_edges(driver, gid: str) -> int:
    result = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid RETURN count(e) AS total",
        gid=gid,
    )
    return result.records[0]["total"]


async def find_new_edges(driver, gid: str, since_ts: float) -> list[dict]:
    result = await driver.execute_query(
        """
        MATCH (src)-[e:RELATES_TO]->(tgt)
        WHERE e.group_id = $gid AND e.created_at >= datetime({epochMillis: toInteger($ts_ms)})
        RETURN e.uuid AS uuid, e.fact AS fact, e.created_at AS created_at,
               src.name AS src_name, tgt.name AS tgt_name,
               e.fr_primary_category AS category, e.fr_enriched AS enriched
        ORDER BY e.created_at DESC
        """,
        gid=gid,
        ts_ms=since_ts * 1000,
    )
    return [dict(r) for r in result.records]


async def find_identity_health_edges(driver, gid: str) -> list[dict]:
    """Search for edges in identity/health categories."""
    result = await driver.execute_query(
        """
        MATCH (src)-[e:RELATES_TO]->(tgt)
        WHERE e.group_id = $gid
          AND e.fr_primary_category IN [
            'IDENTITY_SELF_CONCEPT', 'HEALTH_WELLBEING'
          ]
        RETURN e.uuid AS uuid, e.fact AS fact,
               src.name AS src_name, tgt.name AS tgt_name,
               e.fr_primary_category AS category
        ORDER BY e.fr_primary_category, e.fact
        """,
        gid=gid,
    )
    return [dict(r) for r in result.records]


# ---------------------------------------------------------------------------
# Post-ingestion cleanup: category filter + embedding dedup
# ---------------------------------------------------------------------------

TARGET_CATEGORIES = {'IDENTITY_SELF_CONCEPT', 'HEALTH_WELLBEING'}
DEDUP_SIMILARITY_THRESHOLD = 0.85


async def cleanup_new_edges(driver, gid: str, cutoff_ts: float) -> dict:
    """Delete off-category and duplicate edges created after cutoff_ts.

    Returns dict with stats:
        {extracted, kept_on_category, pruned_off_category, pruned_duplicate, final}
    """
    cutoff_ms = cutoff_ts * 1000

    # --- Phase 1: Category filter ---
    # Count new edges by category
    cat_result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
          AND e.created_at >= datetime({epochMillis: toInteger($ts_ms)})
        RETURN e.fr_primary_category AS cat, count(e) AS cnt
        """,
        gid=gid, ts_ms=cutoff_ms,
    )
    cat_counts = {r["cat"]: r["cnt"] for r in cat_result.records}
    total_extracted = sum(cat_counts.values())

    off_category = sum(cnt for cat, cnt in cat_counts.items() if cat not in TARGET_CATEGORIES)

    # Delete off-category edges
    if off_category > 0:
        await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.group_id = $gid
              AND e.created_at >= datetime({epochMillis: toInteger($ts_ms)})
              AND NOT e.fr_primary_category IN $cats
            DELETE e
            """,
            gid=gid, ts_ms=cutoff_ms, cats=list(TARGET_CATEGORIES),
        )

    on_category = total_extracted - off_category

    # --- Phase 2: Embedding dedup ---
    # Fetch remaining new edges with embeddings
    new_result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
          AND e.created_at >= datetime({epochMillis: toInteger($ts_ms)})
          AND e.fact_embedding IS NOT NULL
        RETURN e.uuid AS uuid, e.fact AS fact, e.fact_embedding AS emb
        """,
        gid=gid, ts_ms=cutoff_ms,
    )
    new_edges = [(r["uuid"], r["fact"], r["emb"]) for r in new_result.records]

    # Fetch pre-existing edges with embeddings
    old_result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
          AND e.created_at < datetime({epochMillis: toInteger($ts_ms)})
          AND e.fact_embedding IS NOT NULL
        RETURN e.fact_embedding AS emb
        """,
        gid=gid, ts_ms=cutoff_ms,
    )

    pruned_duplicate = 0
    if new_edges and old_result.records:
        # Build old embeddings matrix
        old_embs = np.array([r["emb"] for r in old_result.records], dtype=np.float32)
        old_norms = np.linalg.norm(old_embs, axis=1, keepdims=True)
        old_embs_normed = old_embs / np.maximum(old_norms, 1e-10)

        uuids_to_delete = []
        for uuid, fact, emb in new_edges:
            new_vec = np.array(emb, dtype=np.float32).reshape(1, -1)
            new_norm = np.linalg.norm(new_vec)
            if new_norm < 1e-10:
                continue
            new_vec_normed = new_vec / new_norm
            similarities = (new_vec_normed @ old_embs_normed.T).flatten()
            max_sim = float(similarities.max())
            if max_sim > DEDUP_SIMILARITY_THRESHOLD:
                uuids_to_delete.append(uuid)

        if uuids_to_delete:
            pruned_duplicate = len(uuids_to_delete)
            await driver.execute_query(
                """
                MATCH ()-[e:RELATES_TO]->()
                WHERE e.uuid IN $uuids
                DELETE e
                """,
                uuids=uuids_to_delete,
            )

    final = on_category - pruned_duplicate
    stats = {
        'extracted': total_extracted,
        'kept_on_category': on_category,
        'pruned_off_category': off_category,
        'pruned_duplicate': pruned_duplicate,
        'final': final,
    }

    print(f"    Cleanup: Extracted: {total_extracted}, "
          f"Kept (on-category): {on_category}, "
          f"Pruned (duplicate): {pruned_duplicate}, "
          f"Final new: {final}")

    return stats


# ---------------------------------------------------------------------------
# Canary mode
# ---------------------------------------------------------------------------


async def run_canary(short_name: str):
    """Re-ingest all sessions for one persona with identity/health focus."""

    if short_name not in _SHORT_TO_DIR:
        print(f"ERROR: Unknown persona '{short_name}'")
        print(f"Valid names: {', '.join(sorted(_SHORT_TO_DIR.keys()))}")
        sys.exit(1)

    persona_dir = _SHORT_TO_DIR[short_name]
    gid = group_id_for(short_name)

    print(f"{'=' * 70}")
    print(f"  IDENTITY/HEALTH CANARY: {short_name}")
    print(f"  Group: {gid}")
    print(f"  Persona dir: {persona_dir}")
    print(f"{'=' * 70}")

    # Load all sessions
    sessions = load_all_sessions(persona_dir)
    episodes = build_episodes(sessions, short_name)
    print(f"  Sessions loaded: {len(sessions)}")

    graphiti = get_graphiti_client()
    print("  Building indices...")
    await graphiti.build_indices_and_constraints()

    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'testpassword123')
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    # Snapshot before
    edges_before = await count_edges(driver, gid)
    identity_before = await find_identity_health_edges(driver, gid)
    print(f"  Edges before: {edges_before}")
    print(f"  Identity/health edges before: {len(identity_before)}")

    # Record timestamp
    ingest_start_ts = time_module.time()

    # Re-ingest with identity/health extraction prompt
    print(f"\n  Ingesting {len(episodes)} sessions with identity/health extraction prompt...")
    t0 = time_module.time()
    try:
        await graphiti.add_episode_bulk(
            bulk_episodes=episodes,
            group_id=gid,
            custom_extraction_instructions=IDENTITY_HEALTH_EXTRACTION_INSTRUCTIONS,
        )
        elapsed = time_module.time() - t0
        print(f"  Ingestion completed in {format_time(elapsed)}")
    except Exception as e:
        print(f"  ERROR during ingestion: {e}")
        await graphiti.close()
        await driver.close()
        return

    # Count after
    edges_after = await count_edges(driver, gid)
    new_edge_count = edges_after - edges_before
    print(f"  Edges after: {edges_after} (+{new_edge_count} new)")

    # Show new edges
    if new_edge_count > 0:
        new_edges = await find_new_edges(driver, gid, ingest_start_ts)
        print(f"\n  New edges created ({len(new_edges)}):")
        for e in new_edges:
            cat = e.get('category') or '?'
            print(f"    [{cat:>25s}] {e['src_name']} -> {e['tgt_name']}")
            print(f"      fact: {e['fact']}")
    else:
        print("\n  WARNING: No new edges created!")

    # Enrich new edges
    print(f"\n  Enriching edges for {short_name}...")
    enrich_result = await enrich_persona_edges(persona_dir, driver, xai_client)
    print(f"  Enrichment: {enrich_result['personal']} personal, "
          f"{enrich_result['world_knowledge']} WK, "
          f"{enrich_result['errors']} errors")

    # Cleanup: category filter + dedup
    print(f"\n  Running post-enrichment cleanup...")
    cleanup_stats = await cleanup_new_edges(driver, gid, ingest_start_ts)

    # Post-cleanup: show surviving identity/health edges
    identity_after = await find_identity_health_edges(driver, gid)
    new_identity = len(identity_after) - len(identity_before)
    edges_final = await count_edges(driver, gid)

    if new_identity > 0:
        before_uuids = {e['uuid'] for e in identity_before}
        print(f"\n  SURVIVING identity/health edges ({new_identity}):")
        for e in identity_after:
            if e['uuid'] not in before_uuids:
                print(f"    [{e['category']:>25s}] {e['fact']}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  CANARY SUMMARY: {short_name}")
    print(f"    Edges before:         {edges_before}")
    print(f"    Extracted:            +{new_edge_count}")
    print(f"    Pruned (off-category):{cleanup_stats['pruned_off_category']}")
    print(f"    Pruned (duplicate):   {cleanup_stats['pruned_duplicate']}")
    print(f"    Final new edges:      {cleanup_stats['final']}")
    print(f"    Edges after cleanup:  {edges_final}")
    print(f"    Identity/health:      {len(identity_before)} -> {len(identity_after)} (+{new_identity})")
    print(f"{'=' * 70}")

    await graphiti.close()
    await driver.close()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def reingest_persona(persona_dir: str, graphiti, driver) -> dict:
    """Re-ingest all sessions for one persona with identity/health focus.

    Returns dict with stats: {sessions, new_edges, time_s, error, ingest_start_ts}
    """
    short_name = POC_PERSONAS[persona_dir]
    gid = group_id_for(short_name)

    sessions = load_all_sessions(persona_dir)
    episodes = build_episodes(sessions, short_name)

    edges_before = await count_edges(driver, gid)
    ingest_start_ts = time_module.time()

    t0 = time_module.time()
    try:
        await graphiti.add_episode_bulk(
            bulk_episodes=episodes,
            group_id=gid,
            custom_extraction_instructions=IDENTITY_HEALTH_EXTRACTION_INSTRUCTIONS,
        )
    except Exception as e:
        return {
            'sessions': len(sessions),
            'new_edges': 0,
            'time_s': time_module.time() - t0,
            'error': str(e),
            'ingest_start_ts': ingest_start_ts,
        }

    edges_after = await count_edges(driver, gid)

    return {
        'sessions': len(sessions),
        'new_edges': edges_after - edges_before,
        'time_s': time_module.time() - t0,
        'error': None,
        'ingest_start_ts': ingest_start_ts,
    }


async def run_pipeline(personas: list[str]):
    """Main pipeline: re-ingest then enrich for each persona."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    if not checkpoint['started_at']:
        checkpoint['started_at'] = datetime.now(timezone.utc).isoformat()

    graphiti = get_graphiti_client()
    print("Building graph indices and constraints...")
    await graphiti.build_indices_and_constraints()

    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'testpassword123')
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    total_start = time_module.time()
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    for persona_dir in personas:
        short_name = POC_PERSONAS[persona_dir]
        gid = group_id_for(short_name)
        persona_ckpt = checkpoint["personas"].setdefault(short_name, {})

        # Phase 1: Re-ingest with identity/health focus
        if persona_ckpt.get("reingest_status") != "completed":
            print(f"\n{'=' * 60}")
            print(f"  RE-INGESTING (identity/health): {persona_dir}")
            print(f"{'=' * 60}")

            result = await reingest_persona(persona_dir, graphiti, driver)

            persona_ckpt["reingest_status"] = "completed" if result["error"] is None else "error"
            persona_ckpt["reingest_sessions"] = result["sessions"]
            persona_ckpt["reingest_new_edges"] = result["new_edges"]
            persona_ckpt["reingest_time_s"] = result["time_s"]
            persona_ckpt["reingest_error"] = result["error"]
            persona_ckpt["ingest_start_ts"] = result["ingest_start_ts"]
            save_checkpoint(checkpoint)

            if result["error"]:
                print(f"  ERROR: {result['error']}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\n  {MAX_CONSECUTIVE_FAILURES} consecutive failures — halting.")
                    break
                continue
            else:
                consecutive_failures = 0
                print(f"  Ingested {result['sessions']} sessions, "
                      f"+{result['new_edges']} new edges in {format_time(result['time_s'])}")
        else:
            print(f"\n  {short_name}: identity re-ingestion already completed, skipping.")

        # Phase 2: Enrich new edges
        if persona_ckpt.get("enrich_status") != "completed":
            print(f"  Enriching edges for {short_name}...")
            enrich_result = await enrich_persona_edges(persona_dir, driver, xai_client)

            persona_ckpt["enrich_status"] = "completed" if enrich_result["errors"] == 0 else "partial"
            persona_ckpt["enrich_personal"] = enrich_result["personal"]
            persona_ckpt["enrich_world_knowledge"] = enrich_result["world_knowledge"]
            persona_ckpt["enrich_errors"] = enrich_result["errors"]
            save_checkpoint(checkpoint)

            print(f"  Enriched: {enrich_result['personal']} personal, "
                  f"{enrich_result['world_knowledge']} WK")
        else:
            print(f"  {short_name}: enrichment already completed, skipping.")

        # Phase 3: Cleanup — category filter + dedup
        if persona_ckpt.get("cleanup_status") != "completed":
            ingest_ts = persona_ckpt.get("ingest_start_ts") or result.get("ingest_start_ts", 0)
            if ingest_ts:
                print(f"  Cleaning up off-category and duplicate edges...")
                cleanup = await cleanup_new_edges(driver, gid, ingest_ts)
                persona_ckpt["cleanup_status"] = "completed"
                persona_ckpt["cleanup_extracted"] = cleanup["extracted"]
                persona_ckpt["cleanup_pruned_off_category"] = cleanup["pruned_off_category"]
                persona_ckpt["cleanup_pruned_duplicate"] = cleanup["pruned_duplicate"]
                persona_ckpt["cleanup_final"] = cleanup["final"]
                persona_ckpt["completed_at"] = datetime.now(timezone.utc).isoformat()
                save_checkpoint(checkpoint)
            else:
                print(f"  WARNING: No ingest timestamp for cleanup, skipping.")
        else:
            print(f"  {short_name}: cleanup already completed, skipping.")

    # Summary
    total_elapsed = time_module.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"IDENTITY/HEALTH RE-INGESTION COMPLETE — {format_time(total_elapsed)}")
    print(f"{'=' * 70}")
    print_status(checkpoint)

    await graphiti.close()
    await driver.close()


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------


def print_status(checkpoint: dict = None):
    if checkpoint is None:
        checkpoint = load_checkpoint()

    if not checkpoint['personas']:
        print("  No personas processed yet.")
        return

    print(f"\n  {'Persona':<12} {'Extracted':<10} {'Off-cat':<10} {'Dedup':<8} {'Final':<8} {'Time':<10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    total_extracted = 0
    total_final = 0
    for short_name in sorted(checkpoint['personas'].keys()):
        p = checkpoint['personas'][short_name]
        extracted = p.get('cleanup_extracted', p.get('reingest_new_edges', '—'))
        off_cat = p.get('cleanup_pruned_off_category', '—')
        dedup = p.get('cleanup_pruned_duplicate', '—')
        final = p.get('cleanup_final', '—')
        time_s = p.get('reingest_time_s')
        time_str = format_time(time_s) if time_s else '—'

        if isinstance(extracted, int):
            total_extracted += extracted
        if isinstance(final, int):
            total_final += final

        print(f"  {short_name:<12} {str(extracted):<10} {str(off_cat):<10} {str(dedup):<8} {str(final):<8} {time_str:<10}")

    print(f"\n  Total extracted: {total_extracted}, Total final: {total_final}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Identity/health supplementary extraction pass for LifeMemBench"
    )
    parser.add_argument("--canary", type=str, metavar="NAME",
                        help="Run on one persona by short name (e.g., --canary elena)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 40 personas")
    parser.add_argument("--persona", type=str, metavar="DIR_NAME",
                        help="Single persona by directory name (e.g., 3_elena)")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                        help="Personas in range (e.g., --range 1 10)")
    parser.add_argument("--status", action="store_true",
                        help="Show checkpoint progress")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.canary:
        await run_canary(args.canary)
        return

    if args.persona:
        if args.persona not in POC_PERSONAS:
            print(f"ERROR: Unknown persona '{args.persona}'")
            print(f"Valid: {', '.join(POC_PERSONAS.keys())}")
            sys.exit(1)
        personas = [args.persona]
    elif args.range:
        start, end = args.range
        personas = [k for k in POC_PERSONAS if start <= int(k.split('_')[0]) <= end]
        if not personas:
            print(f"ERROR: No personas in range {start}-{end}")
            sys.exit(1)
    elif args.all:
        personas = list(POC_PERSONAS.keys())
    else:
        parser.print_help()
        return

    await run_pipeline(personas)


if __name__ == "__main__":
    asyncio.run(main())
