r"""
detect_supersession.py - Detect superseded facts in LifeMemBench persona graphs.

Reads all enriched edges per persona from Neo4j, groups them by behavioral
category, sends each group to Grok to detect supersession relationships,
resolves chains (A→B→C), and writes results back to Neo4j.

Supersession means: a newer fact REPLACES an older fact about the same topic.
Example: "User is vegetarian" → "User started eating fish" (diet changed).
NOT supersession: "User likes jazz" + "User plays piano" (both true).

The detected supersessions are used by evaluate_lifemembench.py to filter
outdated facts from the candidate pool before top-5 selection.

Usage:
    python detect_supersession.py --dry-run --persona 1_priya  # detect, print, don't write
    python detect_supersession.py --persona 1_priya            # single persona
    python detect_supersession.py --all                        # all 8 personas
    python detect_supersession.py --all --force                # re-check already-checked edges
    python detect_supersession.py --status                     # show checkpoint progress
"""

import argparse
import asyncio
import json
import os
import sys
import time as time_module
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows console encoding for Unicode characters in LLM output
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent


def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env()

from openai import AsyncOpenAI
from neo4j import AsyncGraphDatabase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POC_PERSONAS = {
    "1_priya":    "priya",
    "2_marcus":   "marcus",
    "3_elena":    "elena",
    "4_david":    "david",
    "5_amara":    "amara",
    "6_jake":     "jake",
    "7_fatima":   "fatima",
    "8_tom":      "tom",
    "9_kenji":    "kenji",
    "10_rosa":    "rosa",
    "11_callum":  "callum",
    "12_diane":   "diane",
    "13_raj":     "raj",
    "14_nadia":   "nadia",
    "15_samuel":  "samuel",
    "16_lily":    "lily",
    "17_omar":    "omar",
    "18_bruna":   "bruna",
    "19_patrick": "patrick",
    "20_aisha":   "aisha",
}

ARTIFACTS_DIR = PROJECT_ROOT / "LifeMemEval" / "artifacts"
CHECKPOINT_FILE = ARTIFACTS_DIR / "supersession_checkpoint.json"

SEMAPHORE_LIMIT = 20
WINDOW_SIZE = 30
WINDOW_OVERLAP = 5
MAX_GROUP_SIZE_FOR_SINGLE_CALL = 50


def group_id_for(short_name: str) -> str:
    return f"lifemembench_{short_name}"


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.load(open(CHECKPOINT_FILE, encoding="utf-8"))
    return {"started_at": None, "personas": {}}


def save_checkpoint(checkpoint: dict):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    tmp.replace(CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# Supersession detection prompt
# ---------------------------------------------------------------------------

SUPERSESSION_SYSTEM_PROMPT = """You are a memory system analyst. You detect when newer personal facts SUPERSEDE (replace/invalidate) older facts about the same topic.

SUPERSESSION means: a newer fact makes an older fact OUTDATED, INCORRECT, or NO LONGER THE CURRENT STATE because the user's situation has changed.

Examples of SUPERSESSION (should be detected):
- "User is vegetarian" → "User started eating fish" (diet changed)
- "User drives for Uber" → "User switched to Lyft" (job/gig changed)
- "User planning to get a dog" → "Landlord said no pets" (plan retracted)
- "User does hot yoga 4x/week" → "User got into rock climbing instead" (hobby replaced)
- "User considering moving to Denver" → "User decided to stay in Austin" (tentative plan resolved)
- "User's rent is $800/month" → "User's rent increased to $950/month" (numeric update)
- "User and wife agreed work stays at the office" → "User told wife about the surgery case at kitchen table last night" (specific event shows the general rule no longer holds)

Examples of NOT supersession (should NOT be detected):
- "User likes jazz" + "User plays piano" (different facts, both valid)
- "User has a sister named Maya" + "User has a brother named Raj" (additive, not replacement)
- "User went to Tokyo in March" + "User went to Paris in June" (different events, both valid)
- "User drinks coffee" + "User also drinks tea" (additive preference)
- "User has ADHD" + "User started taking Adderall" (related but ADHD is NOT superseded by medication)
- "User reads sci-fi novels" + "User recently read Dune" (specific instance doesn't supersede general habit)

RULES:
1. Only the NEWER fact can supersede an OLDER fact (check timestamps).
2. Both facts must be about the SAME specific attribute/property of the same subject.
3. Additive facts (both can be simultaneously true) are NOT supersessions.
4. Temporal events at different times are NOT supersessions unless one explicitly replaces the other.
5. A fact can only be superseded by ONE other fact (the most directly replacing one).
6. For chains (A → B → C): A is superseded by B, B is superseded by C. Only C is current.

Respond with strict JSON only:
{
  "supersessions": [
    {
      "superseded_uuid": "uuid of the OLD fact that is now outdated",
      "superseding_uuid": "uuid of the NEWER fact that replaces it",
      "confidence": 0.0-1.0,
      "reason": "brief explanation, max 50 words"
    }
  ]
}

Confidence guide:
- 0.9-1.0: Direct contradiction or explicit update ("was X, now Y")
- 0.7-0.9: Strong implication of replacement (same attribute, different value)
- 0.5-0.7: Likely supersession but ambiguous (tentative plans, soft preferences)
- Below 0.5: Don't include — too uncertain

If no supersessions exist, return: {"supersessions": []}"""


# ---------------------------------------------------------------------------
# Neo4j queries
# ---------------------------------------------------------------------------

async def load_enriched_edges(driver, group_id: str, force: bool = False) -> list[dict]:
    """Load all enriched, non-world-knowledge edges for a persona, chronologically."""
    result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
          AND e.fr_enriched = true
          AND (e.fr_is_world_knowledge IS NULL OR e.fr_is_world_knowledge = false)
        OPTIONAL MATCH (ep:Episodic)
        WHERE ep.uuid IN e.episodes
        WITH e, max(ep.valid_at) AS episode_time
        RETURN e.uuid AS uuid,
               e.fact AS fact,
               e.fr_primary_category AS primary_category,
               COALESCE(episode_time, e.created_at) AS created_at,
               e.fr_supersession_checked AS already_checked
        ORDER BY COALESCE(episode_time, e.created_at) ASC
        """,
        gid=group_id,
    )
    records = result.records if hasattr(result, "records") else result
    edges = []
    for rec in records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        # Convert created_at to unix timestamp
        ca = d.get("created_at")
        if hasattr(ca, "timestamp"):
            d["created_at_ts"] = ca.timestamp()
        elif isinstance(ca, (int, float)):
            d["created_at_ts"] = float(ca)
        else:
            d["created_at_ts"] = 0.0
        edges.append(d)

    if not force:
        edges = [e for e in edges if not e.get("already_checked")]

    return edges


def group_edges_by_category(edges: list[dict]) -> dict[str, list[dict]]:
    """Group edges by fr_primary_category, dropping singletons."""
    groups = defaultdict(list)
    for e in edges:
        cat = e.get("primary_category")
        if cat:
            groups[cat].append(e)
    return {cat: cat_edges for cat, cat_edges in groups.items() if len(cat_edges) >= 2}


# ---------------------------------------------------------------------------
# Windowing for large groups
# ---------------------------------------------------------------------------

def split_into_windows(edges: list[dict]) -> list[list[dict]]:
    """Split a chronologically sorted edge list into overlapping windows.

    Each window is WINDOW_SIZE edges. Adjacent windows overlap by WINDOW_OVERLAP
    so that supersession relationships spanning boundaries are detected.
    """
    if len(edges) <= MAX_GROUP_SIZE_FOR_SINGLE_CALL:
        return [edges]

    windows = []
    step = WINDOW_SIZE - WINDOW_OVERLAP  # 25
    for start in range(0, len(edges), step):
        window = edges[start:start + WINDOW_SIZE]
        if len(window) < 2:
            break
        windows.append(window)
    return windows


# ---------------------------------------------------------------------------
# Grok API interaction
# ---------------------------------------------------------------------------

def build_user_prompt(edges: list[dict], category: str) -> str:
    """Build the user message listing all facts in a category group."""
    lines = [
        f"Category: {category}",
        f"Facts ({len(edges)} total, chronological order — oldest first):",
        "",
    ]
    for i, e in enumerate(edges, 1):
        ts = e.get("created_at_ts", 0)
        if ts > 0:
            ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            ts_str = "unknown"
        lines.append(f"{i}. [{ts_str}] UUID={e['uuid']}")
        lines.append(f"   Fact: {e['fact']}")
        lines.append("")
    lines.append("Identify all supersession pairs. If none exist, return empty list.")
    return "\n".join(lines)


def validate_response(raw: dict, valid_uuids: set[str]) -> list[dict]:
    """Validate and normalize the LLM supersession response."""
    supersessions = raw.get("supersessions", [])
    validated = []
    for s in supersessions:
        if not isinstance(s, dict):
            continue
        superseded = s.get("superseded_uuid", "")
        superseding = s.get("superseding_uuid", "")
        confidence = s.get("confidence", 0.0)
        reason = str(s.get("reason", ""))[:200]

        # Basic validation
        if not superseded or not superseding or superseded == superseding:
            continue
        # Verify UUIDs exist in the input set
        if superseded not in valid_uuids or superseding not in valid_uuids:
            continue
        # Clamp confidence
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        # Skip low confidence
        if confidence < 0.5:
            continue

        validated.append({
            "superseded_uuid": superseded,
            "superseding_uuid": superseding,
            "confidence": round(confidence, 3),
            "reason": reason,
        })
    return validated


async def call_grok(prompt: str, xai_client, sem, valid_uuids: set[str]) -> list[dict]:
    """Call Grok to detect supersessions, with retry logic."""
    async with sem:
        for attempt in range(3):
            try:
                resp = await xai_client.chat.completions.create(
                    model=os.environ.get("GROK_MODEL", "grok-4-1-fast-reasoning"),
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SUPERSESSION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                )
                raw = json.loads(resp.choices[0].message.content)
                return validate_response(raw, valid_uuids)
            except Exception as e:
                if attempt < 2:
                    wait = 2 * (2 ** attempt)
                    print(f"      Retry {attempt + 1}/3 after {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    print(f"      ERROR after 3 retries: {e}")
                    return []


async def detect_in_group(
    edges: list[dict], category: str, xai_client, sem
) -> list[dict]:
    """Detect supersessions within a category group, handling windowing."""
    windows = split_into_windows(edges)
    all_supersessions = []
    seen_pairs = set()

    valid_uuids = {e["uuid"] for e in edges}

    for window in windows:
        prompt = build_user_prompt(window, category)
        window_uuids = {e["uuid"] for e in window}
        result = await call_grok(prompt, xai_client, sem, window_uuids)

        for s in result:
            pair_key = (s["superseded_uuid"], s["superseding_uuid"])
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_supersessions.append(s)

    return all_supersessions


# ---------------------------------------------------------------------------
# Chain resolution
# ---------------------------------------------------------------------------

def resolve_chains(supersessions: list[dict]) -> list[dict]:
    """Resolve supersession chains: if A→B and B→C, make A→C and B→C.

    This ensures filtering only needs to check fr_superseded_by IS NOT NULL.
    The terminal (current) fact is never marked as superseded.
    """
    # Build forward map: superseded -> superseding
    forward = {}
    for s in supersessions:
        forward[s["superseded_uuid"]] = s["superseding_uuid"]

    def find_terminal(uuid: str, visited: set) -> str:
        if uuid in visited:
            return uuid  # Cycle detected, break
        visited.add(uuid)
        if uuid in forward:
            return find_terminal(forward[uuid], visited)
        return uuid

    resolved = []
    for s in supersessions:
        terminal = find_terminal(s["superseding_uuid"], set())
        chain_note = " [chain-resolved]" if terminal != s["superseding_uuid"] else ""
        resolved.append({
            "superseded_uuid": s["superseded_uuid"],
            "superseding_uuid": terminal,
            "confidence": s["confidence"],
            "reason": s["reason"] + chain_note,
        })

    return resolved


# ---------------------------------------------------------------------------
# Neo4j writes
# ---------------------------------------------------------------------------

async def write_supersessions(driver, supersessions: list[dict]):
    """Write supersession properties to Neo4j edges."""
    for s in supersessions:
        await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.uuid = $uuid
            SET e.fr_superseded_by = $superseding_uuid,
                e.fr_supersession_confidence = $confidence,
                e.fr_supersession_reason = $reason,
                e.fr_supersession_checked = true
            """,
            uuid=s["superseded_uuid"],
            superseding_uuid=s["superseding_uuid"],
            confidence=s["confidence"],
            reason=s["reason"],
        )


async def mark_all_checked(driver, edge_uuids: list[str]):
    """Mark all processed edges as supersession-checked (even non-superseded ones)."""
    # Batch in chunks of 500 to avoid huge parameter lists
    for i in range(0, len(edge_uuids), 500):
        chunk = edge_uuids[i:i + 500]
        await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.uuid IN $uuids AND e.fr_supersession_checked IS NULL
            SET e.fr_supersession_checked = true
            """,
            uuids=chunk,
        )


async def clear_supersession_data(driver, group_id: str):
    """Clear all supersession properties for a persona (--force)."""
    await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
        SET e.fr_superseded_by = null,
            e.fr_supersession_confidence = null,
            e.fr_supersession_reason = null,
            e.fr_supersession_checked = null
        """,
        gid=group_id,
    )


# ---------------------------------------------------------------------------
# Per-persona orchestration
# ---------------------------------------------------------------------------

async def process_persona(
    persona_dir: str, driver, xai_client, sem,
    dry_run: bool = False, force: bool = False,
) -> dict:
    """Run supersession detection for a single persona."""
    short_name = POC_PERSONAS[persona_dir]
    gid = group_id_for(short_name)
    t0 = time_module.time()

    # Clear old data if --force
    if force and not dry_run:
        print(f"    Clearing previous supersession data (--force)...")
        await clear_supersession_data(driver, gid)

    # Load edges
    edges = await load_enriched_edges(driver, gid, force=force)
    if not edges:
        print(f"    No edges to process for {short_name} (all checked or none found)")
        return {
            "status": "completed",
            "edges_checked": 0,
            "groups_checked": 0,
            "supersessions_found": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "errors": 0,
            "time_s": 0,
        }

    # Group by category
    groups = group_edges_by_category(edges)

    # Cross-category supersession pass: add a group containing ALL edges so that
    # supersession pairs spanning different categories are evaluated.
    if len(edges) >= 2:
        groups["__ALL__"] = edges

    singleton_edges = len(edges) - sum(
        len(g) for cat, g in groups.items() if cat != "__ALL__"
    )
    cat_group_count = sum(1 for cat in groups if cat != "__ALL__")
    print(f"    {len(edges)} edges -> {cat_group_count} category groups + cross-category pass, "
          f"{singleton_edges} singletons skipped")

    # Detect supersessions per category group
    all_supersessions = []
    seen_pairs = set()  # deduplicate across category groups and __ALL__
    groups_checked = 0
    errors = 0

    tasks = []
    for category, cat_edges in sorted(groups.items()):
        tasks.append((category, cat_edges))

    # Process concurrently via asyncio.gather
    async def process_group(category, cat_edges):
        try:
            result = await detect_in_group(cat_edges, category, xai_client, sem)
            return category, cat_edges, result, None
        except Exception as e:
            return category, cat_edges, [], e

    coros = [process_group(cat, cat_edges) for cat, cat_edges in tasks]
    results = await asyncio.gather(*coros, return_exceptions=True)

    for item in results:
        if isinstance(item, Exception):
            errors += 1
            print(f"      UNEXPECTED ERROR: {item}")
            continue
        category, cat_edges, supersessions, error = item
        groups_checked += 1
        if error:
            errors += 1
            print(f"      {category}: {len(cat_edges)} edges -> ERROR: {error}")
        elif supersessions:
            new_count = 0
            for s in supersessions:
                pair_key = (s["superseded_uuid"], s["superseding_uuid"])
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    all_supersessions.append(s)
                    new_count += 1
            print(f"      {category}: {len(cat_edges)} edges -> "
                  f"{len(supersessions)} supersession(s)"
                  f"{f' ({len(supersessions) - new_count} duplicates skipped)' if new_count < len(supersessions) else ''}")
            for s in supersessions:
                # Find the facts for display
                old_fact = next((e["fact"] for e in cat_edges if e["uuid"] == s["superseded_uuid"]), "?")
                new_fact = next((e["fact"] for e in cat_edges if e["uuid"] == s["superseding_uuid"]), "?")
                print(f"        [{s['confidence']:.2f}] \"{old_fact[:60]}\" -> \"{new_fact[:60]}\"")
                print(f"               Reason: {s['reason']}")
        else:
            print(f"      {category}: {len(cat_edges)} edges -> no supersessions")

    # Resolve chains
    if all_supersessions:
        resolved = resolve_chains(all_supersessions)
        chain_resolved = sum(1 for r in resolved if "[chain-resolved]" in r.get("reason", ""))
        if chain_resolved:
            print(f"\n    Chain resolution: {chain_resolved} edge(s) updated to point to terminal")
    else:
        resolved = []

    # Confidence breakdown
    high_conf = sum(1 for s in resolved if s["confidence"] >= 0.8)
    med_conf = sum(1 for s in resolved if 0.5 <= s["confidence"] < 0.8)

    # Write to Neo4j
    if resolved and not dry_run:
        print(f"\n    Writing {len(resolved)} supersession(s) to Neo4j...")
        await write_supersessions(driver, resolved)
        all_uuids = [e["uuid"] for e in edges]
        await mark_all_checked(driver, all_uuids)
        print(f"    Marked {len(all_uuids)} edges as supersession-checked")
    elif dry_run:
        print(f"\n    DRY RUN — would write {len(resolved)} supersession(s)")

    elapsed = time_module.time() - t0

    return {
        "status": "completed" if errors == 0 else "partial",
        "edges_checked": len(edges),
        "groups_checked": groups_checked,
        "supersessions_found": len(resolved),
        "high_confidence": high_conf,
        "medium_confidence": med_conf,
        "errors": errors,
        "time_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Summary / status
# ---------------------------------------------------------------------------

def print_summary_table(checkpoint: dict):
    """Print a summary table of supersession detection results."""
    print("\n" + "=" * 72)
    print("  SUPERSESSION DETECTION SUMMARY")
    print("=" * 72)
    print(f"  {'Persona':<10} {'Status':<12} {'Edges':>7} {'Groups':>7} "
          f"{'Found':>7} {'High':>6} {'Med':>6} {'Err':>5} {'Time':>7}")
    print("  " + "-" * 68)

    total_edges = 0
    total_found = 0
    total_errors = 0

    for persona_dir, short_name in POC_PERSONAS.items():
        p = checkpoint.get("personas", {}).get(short_name, {})
        if not p:
            print(f"  {short_name:<10} {'pending':<12}")
            continue

        edges = p.get("edges_checked", 0)
        groups = p.get("groups_checked", 0)
        found = p.get("supersessions_found", 0)
        high = p.get("high_confidence", 0)
        med = p.get("medium_confidence", 0)
        errs = p.get("errors", 0)
        t = p.get("time_s", 0)

        total_edges += edges
        total_found += found
        total_errors += errs

        print(f"  {short_name:<10} {p.get('status', '?'):<12} {edges:>7} {groups:>7} "
              f"{found:>7} {high:>6} {med:>6} {errs:>5} {t:>6.0f}s")

    print("  " + "-" * 68)
    print(f"  {'TOTAL':<10} {'':12} {total_edges:>7} {'':>7} "
          f"{total_found:>7} {'':>6} {'':>6} {total_errors:>5}")
    print()


def show_status():
    """Show checkpoint status."""
    checkpoint = load_checkpoint()
    if not checkpoint.get("personas"):
        print("No supersession detection has been run yet.")
        return
    print_summary_table(checkpoint)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Detect superseded facts in LifeMemBench persona graphs"
    )
    parser.add_argument("--all", action="store_true",
                        help="Process all 8 personas")
    parser.add_argument("--persona", type=str, metavar="DIR_NAME",
                        help="Single persona directory name (e.g., 1_priya)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect supersessions but don't write to Neo4j")
    parser.add_argument("--force", action="store_true",
                        help="Clear and re-check already-checked edges")
    parser.add_argument("--status", action="store_true",
                        help="Show progress for all personas")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    # Determine personas to process
    if args.persona:
        if args.persona not in POC_PERSONAS:
            print(f"ERROR: Unknown persona '{args.persona}'")
            print(f"Valid personas: {', '.join(POC_PERSONAS.keys())}")
            sys.exit(1)
        personas = [args.persona]
    elif args.all:
        personas = list(POC_PERSONAS.keys())
    else:
        parser.print_help()
        return

    # Connect
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

    # Load checkpoint
    checkpoint = load_checkpoint()
    if not checkpoint.get("started_at"):
        checkpoint["started_at"] = datetime.now(timezone.utc).isoformat()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — detecting supersessions, printing results, no writes")
        print("=" * 60)

    total_start = time_module.time()

    for persona_dir in personas:
        short_name = POC_PERSONAS[persona_dir]

        # Skip if already completed and not --force
        persona_ckpt = checkpoint.get("personas", {}).get(short_name, {})
        if (not args.force and not args.dry_run
                and persona_ckpt.get("status") == "completed"):
            print(f"\n  {short_name}: already completed, skipping. Use --force to redo.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  SUPERSESSION DETECTION: {persona_dir} ({short_name})")
        print(f"{'=' * 60}")

        result = await process_persona(
            persona_dir, driver, xai_client, sem,
            dry_run=args.dry_run, force=args.force,
        )

        # Update checkpoint (skip for dry run)
        if not args.dry_run:
            checkpoint.setdefault("personas", {})[short_name] = result
            save_checkpoint(checkpoint)

        # Per-persona summary
        print(f"\n  {short_name} summary:")
        print(f"    Edges checked:       {result['edges_checked']}")
        print(f"    Category groups:     {result['groups_checked']}")
        print(f"    Supersessions found: {result['supersessions_found']} "
              f"(high={result['high_confidence']}, medium={result['medium_confidence']})")
        print(f"    Errors:              {result['errors']}")
        print(f"    Time:                {result['time_s']:.0f}s")

    total_elapsed = time_module.time() - total_start
    print(f"\n  Total time: {total_elapsed:.0f}s")

    # Print overall summary
    print_summary_table(checkpoint)

    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
