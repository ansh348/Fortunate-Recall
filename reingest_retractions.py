r"""
reingest_retractions.py — Targeted re-ingestion of retraction sessions.

Re-ingests ONLY the specific sessions containing retraction events:
  - david session 18: book project abandoned
  - amara session 17: UCL LLM dropped

Uses an enhanced extraction prompt that explicitly instructs the LLM to
extract retraction events as distinct edges.

Usage: python reingest_retractions.py
"""

import asyncio
import json
import os
import sys
import time as time_module
from pathlib import Path

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
    CUSTOM_EXTRACTION_INSTRUCTIONS,
    LIFEMEMEVAL_DIR,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRACTION_SESSIONS = [
    {
        "persona_dir": "4_david",
        "short_name": "david",
        "session_file": LIFEMEMEVAL_DIR / "4_david" / "sessions" / "session_18.json",
        "description": "Book project abandoned — retraction event",
    },
    {
        "persona_dir": "5_amara",
        "short_name": "amara",
        "session_file": LIFEMEMEVAL_DIR / "5_amara" / "sessions" / "session_17.json",
        "description": "UCL LLM dropped — retraction event",
    },
]

# Enhanced extraction instructions that append retraction-specific guidance
RETRACTION_EXTRACTION_INSTRUCTIONS = CUSTOM_EXTRACTION_INSTRUCTIONS + """

RETRACTION DETECTION (CRITICAL):
When a user cancels, abandons, drops, or retracts a previous plan, goal, or commitment,
extract the retraction as a distinct edge. Preserve the specific thing being retracted.

Examples of retraction language:
- "I'm done", "it's dead", "I'm letting it go", "the whole idea is dead"
- "I've abandoned", "I've dropped", "I've given up on", "I've walked away from"
- "not worth it", "waste of time", "no longer pursuing"

For each retraction, extract an edge that:
1. Names the specific thing being retracted (e.g., "the book project", "the UCL LLM")
2. Captures that it was ABANDONED/DROPPED/RETRACTED (not just paused)
3. Includes the reason if stated (e.g., "market doesn't exist", "practitioners said it was worthless")

These retraction edges are essential for the memory system to know what is NO LONGER true
about the user's plans and commitments.
"""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_session(session_path: Path) -> dict:
    """Load a single session JSON file."""
    with open(session_path, encoding="utf-8") as f:
        return json.load(f)


def build_single_episode(session: dict, short_name: str) -> RawEpisode:
    """Convert a single session dict to a RawEpisode."""
    sid = session["session_id"]
    session_type = session.get("type", "unknown")

    content = "\n".join(
        f"{t['role']}: {t['content']}" for t in session["turns"]
    )

    return RawEpisode(
        name=f"lmb_{short_name}_s{sid:02d}",
        content=content,
        source_description=(
            f"LifeMemBench {short_name} session {sid} ({session_type})"
        ),
        source=EpisodeType.message,
        reference_time=parse_session_date(session["date"]),
    )


async def count_edges_before(driver, gid: str) -> int:
    """Count current edges for a group_id."""
    result = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid RETURN count(e) AS total",
        gid=gid,
    )
    return result.records[0]["total"]


async def find_new_edges(driver, gid: str, since_ts: float) -> list[dict]:
    """Find edges created after a given timestamp."""
    result = await driver.execute_query(
        """
        MATCH (src)-[e:RELATES_TO]->(tgt)
        WHERE e.group_id = $gid AND e.created_at >= datetime({epochMillis: toInteger($ts_ms)})
        RETURN e.uuid AS uuid, e.fact AS fact, e.created_at AS created_at,
               src.name AS src_name, tgt.name AS tgt_name
        ORDER BY e.created_at DESC
        """,
        gid=gid,
        ts_ms=since_ts * 1000,
    )
    return [dict(r) for r in result.records]


async def find_retraction_edges(driver, gid: str) -> list[dict]:
    """Search for edges containing retraction-related keywords."""
    retraction_keywords = [
        "abandon", "dropped", "dead", "letting go", "retract",
        "gave up", "given up", "walked away", "no longer",
        "stopped pursuing", "cancelled", "done with",
    ]

    all_matches = []
    for keyword in retraction_keywords:
        result = await driver.execute_query(
            """
            MATCH (src)-[e:RELATES_TO]->(tgt)
            WHERE e.group_id = $gid AND toLower(e.fact) CONTAINS toLower($kw)
            RETURN e.uuid AS uuid, e.fact AS fact, e.created_at AS created_at,
                   src.name AS src_name, tgt.name AS tgt_name,
                   e.fr_primary_category AS category, e.fr_enriched AS enriched
            """,
            gid=gid,
            kw=keyword,
        )
        for r in result.records:
            d = dict(r)
            if d["uuid"] not in {m["uuid"] for m in all_matches}:
                all_matches.append(d)

    return all_matches


async def main():
    print("=" * 70)
    print("TARGETED RETRACTION RE-INGESTION")
    print("=" * 70)

    graphiti = get_graphiti_client()
    print("Building indices and constraints...")
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

    for cfg in RETRACTION_SESSIONS:
        short_name = cfg["short_name"]
        gid = group_id_for(short_name)
        session_path = cfg["session_file"]

        print(f"\n{'=' * 60}")
        print(f"  RE-INGESTING: {cfg['description']}")
        print(f"  Group: {gid}")
        print(f"  Session: {session_path.name}")
        print(f"{'=' * 60}")

        # Load session
        session = load_session(session_path)
        episode = build_single_episode(session, short_name)
        print(f"  Session date: {session['date']}")
        print(f"  Topic: {session.get('topic', 'N/A')}")
        print(f"  Turns: {len(session['turns'])}")

        # Count edges before
        edges_before = await count_edges_before(driver, gid)
        print(f"  Edges before: {edges_before}")

        # Record timestamp for finding new edges
        ingest_start_ts = time_module.time()

        # Re-ingest with enhanced extraction instructions
        print(f"  Ingesting with retraction-enhanced extraction prompt...")
        t0 = time_module.time()
        try:
            await graphiti.add_episode_bulk(
                bulk_episodes=[episode],
                group_id=gid,
                custom_extraction_instructions=RETRACTION_EXTRACTION_INSTRUCTIONS,
            )
            elapsed = time_module.time() - t0
            print(f"  Ingestion completed in {elapsed:.0f}s")
        except Exception as e:
            print(f"  ERROR during ingestion: {e}")
            continue

        # Count edges after
        edges_after = await count_edges_before(driver, gid)
        new_edge_count = edges_after - edges_before
        print(f"  Edges after: {edges_after} (+{new_edge_count} new)")

        # Show new edges
        if new_edge_count > 0:
            new_edges = await find_new_edges(driver, gid, ingest_start_ts)
            print(f"\n  New edges created:")
            for e in new_edges:
                print(f"    [{e['uuid'][:12]}] {e['src_name']} -> {e['tgt_name']}")
                print(f"      fact: {e['fact']}")

        # Enrich new edges
        print(f"\n  Enriching new edges for {short_name}...")
        enrich_result = await enrich_persona_edges(
            cfg["persona_dir"], driver, xai_client,
        )
        print(f"  Enrichment: {enrich_result['personal']} personal, "
              f"{enrich_result['world_knowledge']} WK, "
              f"{enrich_result['errors']} errors")

    # ---------------------------------------------------------------------------
    # Verification: search for retraction edges
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("VERIFICATION: Searching for retraction-related edges")
    print(f"{'=' * 70}")

    for cfg in RETRACTION_SESSIONS:
        short_name = cfg["short_name"]
        gid = group_id_for(short_name)

        print(f"\n  --- {short_name} ({gid}) ---")
        matches = await find_retraction_edges(driver, gid)

        if matches:
            print(f"  Found {len(matches)} retraction-related edges:")
            for m in matches:
                cat = m.get("category", "?")
                enriched = "enriched" if m.get("enriched") else "NOT enriched"
                print(f"    [{cat:>25s}] [{enriched}] {m['fact']}")
        else:
            print(f"  WARNING: No retraction-related edges found!")

    # ---------------------------------------------------------------------------
    # Also search for the specific facts we need
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("TARGETED SEARCH: Specific retraction facts needed")
    print(f"{'=' * 70}")

    # david_q04: book abandoned
    print("\n  david_q04 needs: edge about abandoned book project")
    david_result = await driver.execute_query(
        """
        MATCH (src)-[e:RELATES_TO]->(tgt)
        WHERE e.group_id = 'lifemembench_david'
          AND (toLower(e.fact) CONTAINS 'book' OR toLower(e.fact) CONTAINS 'manuscript')
          AND (toLower(e.fact) CONTAINS 'abandon' OR toLower(e.fact) CONTAINS 'dead'
               OR toLower(e.fact) CONTAINS 'letting go' OR toLower(e.fact) CONTAINS 'done'
               OR toLower(e.fact) CONTAINS 'gave up' OR toLower(e.fact) CONTAINS 'quit')
        RETURN e.uuid AS uuid, e.fact AS fact, e.fr_primary_category AS category,
               src.name AS src_name, tgt.name AS tgt_name
        """,
    )
    if david_result.records:
        for r in david_result.records:
            print(f"    FOUND: [{r['category']}] {r['fact']}")
    else:
        # Broader search
        print("    (no exact match — trying broader search)")
        david_broad = await driver.execute_query(
            """
            MATCH (src)-[e:RELATES_TO]->(tgt)
            WHERE e.group_id = 'lifemembench_david'
              AND (toLower(e.fact) CONTAINS 'book' AND
                   (toLower(e.fact) CONTAINS 'abandon' OR toLower(e.fact) CONTAINS 'dead'
                    OR toLower(e.fact) CONTAINS 'stop' OR toLower(e.fact) CONTAINS 'quit'
                    OR toLower(e.fact) CONTAINS 'let go' OR toLower(e.fact) CONTAINS 'gave up'
                    OR toLower(e.fact) CONTAINS 'finished'))
            RETURN e.uuid AS uuid, e.fact AS fact, e.fr_primary_category AS category
            """,
        )
        if david_broad.records:
            for r in david_broad.records:
                print(f"    FOUND (broad): [{r['category']}] {r['fact']}")
        else:
            print("    NOT FOUND — checking all book-related edges:")
            david_all_book = await driver.execute_query(
                """
                MATCH (src)-[e:RELATES_TO]->(tgt)
                WHERE e.group_id = 'lifemembench_david'
                  AND toLower(e.fact) CONTAINS 'book'
                RETURN e.uuid AS uuid, e.fact AS fact, e.fr_primary_category AS category
                """,
            )
            for r in david_all_book.records:
                print(f"    [book edge] [{r['category']}] {r['fact']}")

    # amara_q06: UCL LLM dropped
    print("\n  amara_q06 needs: edge about dropped UCL LLM / master's")
    amara_result = await driver.execute_query(
        """
        MATCH (src)-[e:RELATES_TO]->(tgt)
        WHERE e.group_id = 'lifemembench_amara'
          AND (toLower(e.fact) CONTAINS 'ucl' OR toLower(e.fact) CONTAINS 'llm'
               OR toLower(e.fact) CONTAINS 'master')
          AND (toLower(e.fact) CONTAINS 'drop' OR toLower(e.fact) CONTAINS 'dead'
               OR toLower(e.fact) CONTAINS 'abandon' OR toLower(e.fact) CONTAINS 'reject'
               OR toLower(e.fact) CONTAINS 'decided against' OR toLower(e.fact) CONTAINS 'not pursuing')
        RETURN e.uuid AS uuid, e.fact AS fact, e.fr_primary_category AS category,
               src.name AS src_name, tgt.name AS tgt_name
        """,
    )
    if amara_result.records:
        for r in amara_result.records:
            print(f"    FOUND: [{r['category']}] {r['fact']}")
    else:
        print("    (no exact match — trying broader search)")
        amara_broad = await driver.execute_query(
            """
            MATCH (src)-[e:RELATES_TO]->(tgt)
            WHERE e.group_id = 'lifemembench_amara'
              AND (toLower(e.fact) CONTAINS 'ucl' OR toLower(e.fact) CONTAINS 'llm'
                   OR toLower(e.fact) CONTAINS 'master')
            RETURN e.uuid AS uuid, e.fact AS fact, e.fr_primary_category AS category
            """,
        )
        if amara_broad.records:
            for r in amara_broad.records:
                print(f"    [related edge] [{r['category']}] {r['fact']}")
        else:
            print("    NOT FOUND")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total_elapsed = time_module.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Total time: {total_elapsed:.0f}s")
    print(f"{'=' * 70}")
    print("\nNext step: run full evaluation")
    print("  python evaluate_lifemembench.py --all --config full")

    await graphiti.close()
    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
