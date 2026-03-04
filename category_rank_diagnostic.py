"""
category_rank_diagnostic.py -- Compare cosine vs recency ranking within categories.

For each of 19 target questions (8 gains + 11 regressions):
1. Find the correct edge UUID via keyword search in Neo4j
2. Get its fr_primary_category
3. Fetch ALL edges in that category for that persona (with fact_embedding + created_at)
4. Sort by cosine similarity to the question
5. Sort by created_at DESC (recency)
6. Report the rank of the correct edge in each ordering
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
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

for var in ["OPENAI_API_KEY"]:
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from neo4j import AsyncGraphDatabase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUESTIONS_PATH = PROJECT_ROOT / "LifeMemEval" / "lifemembench_questions.json"

PERSONAS = {
    "priya":  "lifemembench_priya",
    "marcus": "lifemembench_marcus",
    "elena":  "lifemembench_elena",
    "david":  "lifemembench_david",
    "amara":  "lifemembench_amara",
    "jake":   "lifemembench_jake",
    "tom":    "lifemembench_tom",
    "omar":   "lifemembench_omar",
}

# Keywords to find the correct edge for each question.
# Gains from KEYWORD_REGISTRY, regressions defined from correct_answer.
ANSWER_KEYWORDS = {
    # === ROUTING GAINS (8) ===
    "amara_q02":  ["samsung", "galaxy s25", "galaxy"],
    "amara_q03":  ["boxing", "bethnal green"],
    "marcus_q02": ["ram 1500", "2024 ram"],
    "omar_q02":   ["gulfton"],
    "tom_q03":    ["walking group"],
    "david_q11":  ["parent-teacher conference", "parent teacher"],
    "jake_q09":   ["county cork", "brennan"],
    "omar_q07":   ["dallas"],
    # === ROUTING REGRESSIONS (11) ===
    "david_q09":  ["bend", "sarah wants to move"],
    "david_q12":  ["hiking", "watercolor", "debate team"],
    "elena_q07":  ["night shift", "destroying her health", "can't sleep"],
    "elena_q09":  ["cicero", "logan square"],
    "jake_q08":   ["southie", "south boston", "quincy"],
    "marcus_q01": ["tirehub", "tire hub"],
    "marcus_q04": ["diabetes", "metformin", "type 2 diabetes"],
    "priya_q08":  ["nyc", "new york city", "relocate"],
    "priya_q11":  ["rock climbing", "tamil recipe"],
    "priya_q13":  ["4 times a week", "hot yoga"],
    "priya_q14":  ["tokyo", "march 20"],
}

ROUTING_GAINS = {
    "amara_q02", "amara_q03", "marcus_q02", "omar_q02",
    "tom_q03", "david_q11", "jake_q09", "omar_q07",
}

ROUTING_REGRESSIONS = {
    "david_q09", "david_q12", "elena_q07", "elena_q09",
    "jake_q08", "marcus_q01", "marcus_q04", "priya_q08",
    "priya_q11", "priya_q13", "priya_q14",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _to_unix_ts(value) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "to_native"):
        return value.to_native().timestamp()
    if hasattr(value, "timestamp"):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


async def find_correct_edge(driver, group_id: str, keywords: list[str]) -> list[dict]:
    """Find edges matching any keyword for a persona. Returns all matches."""
    result = await driver.execute_query(
        """
        MATCH (s)-[e:RELATES_TO]->(t)
        WHERE e.group_id = $gid
          AND e.expired_at IS NULL
        RETURN e.uuid AS uuid,
               e.fact AS fact,
               e.fr_primary_category AS category,
               e.fr_enriched AS enriched,
               e.created_at AS created_at,
               e.fact_embedding AS fact_embedding
        """,
        gid=group_id,
    )
    records = result.records if hasattr(result, "records") else result
    matches = []
    for rec in records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        fact = (d.get("fact") or "").lower()
        matched = [kw for kw in keywords if kw.lower() in fact]
        if matched:
            d["matched_keywords"] = matched
            d["created_at_ts"] = _to_unix_ts(d.get("created_at"))
            matches.append(d)
    return matches


async def fetch_all_category_edges(driver, group_id: str, category: str) -> list[dict]:
    """Fetch ALL edges in a category for a persona."""
    result = await driver.execute_query(
        """
        MATCH (s)-[e:RELATES_TO]->(t)
        WHERE e.group_id = $gid
          AND e.fr_enriched = true
          AND e.fr_primary_category = $cat
          AND e.expired_at IS NULL
          AND (e.fr_is_world_knowledge IS NULL OR e.fr_is_world_knowledge = false)
        RETURN e.uuid AS uuid,
               e.fact AS fact,
               e.fr_primary_category AS category,
               e.created_at AS created_at,
               e.fact_embedding AS fact_embedding
        """,
        gid=group_id,
        cat=category,
    )
    records = result.records if hasattr(result, "records") else result
    edges = []
    for rec in records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        d["created_at_ts"] = _to_unix_ts(d.get("created_at"))
        raw = d.get("fact_embedding")
        d["fact_embedding"] = list(raw) if raw is not None else None
        edges.append(d)
    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    with open(QUESTIONS_PATH) as f:
        questions = {q["id"]: q for q in json.load(f)}

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig())

    results = []  # (qid, group, category, total_in_cat, cosine_rank, recency_rank, cosine_val, fact_preview)

    target_qids = sorted(ROUTING_GAINS | ROUTING_REGRESSIONS)

    for i, qid in enumerate(target_qids):
        q = questions[qid]
        persona = q["persona"]
        group_id = q["group_id"]
        question = q["question"]
        keywords = ANSWER_KEYWORDS[qid]
        group_label = "GAIN" if qid in ROUTING_GAINS else "REGR"

        print(f"[{i+1}/{len(target_qids)}] {qid} ({group_label}): searching for {keywords}")

        # Step 1: Find correct edge
        matches = await find_correct_edge(driver, group_id, keywords)
        if not matches:
            print(f"  ** NO MATCHING EDGE FOUND for {keywords}")
            results.append((qid, group_label, "?", 0, -1, -1, 0.0, "NO MATCH"))
            continue

        # Prefer enriched matches
        enriched = [m for m in matches if m.get("enriched")]
        best = enriched[0] if enriched else matches[0]
        correct_uuid = best["uuid"]
        correct_cat = best.get("category") or "UNKNOWN"
        correct_fact = best.get("fact", "")

        print(f"  Found: [{correct_uuid[:8]}] cat={correct_cat}")
        print(f"  Fact: {correct_fact[:80]}")

        # Step 2: Fetch ALL edges in that category
        cat_edges = await fetch_all_category_edges(driver, group_id, correct_cat)
        total_in_cat = len(cat_edges)
        print(f"  Category '{correct_cat}' has {total_in_cat} edges")

        # Check correct edge is in the category results
        correct_in_cat = any(e["uuid"] == correct_uuid for e in cat_edges)
        if not correct_in_cat:
            print(f"  ** Correct edge NOT in category results (maybe not enriched?)")
            results.append((qid, group_label, correct_cat, total_in_cat, -1, -1, 0.0, correct_fact[:60]))
            continue

        # Step 3: Embed the query
        query_vec = await embedder.create(input_data=question.replace('\n', ' '))

        # Step 4: Compute cosine similarity for all category edges
        for e in cat_edges:
            if e["fact_embedding"] is not None:
                e["cosine"] = cosine_sim(query_vec, e["fact_embedding"])
            else:
                e["cosine"] = -1.0

        # Step 5: Sort by cosine (descending) and by recency (descending)
        by_cosine = sorted(cat_edges, key=lambda e: e["cosine"], reverse=True)
        by_recency = sorted(cat_edges, key=lambda e: e["created_at_ts"], reverse=True)

        cosine_rank = next((i+1 for i, e in enumerate(by_cosine) if e["uuid"] == correct_uuid), -1)
        recency_rank = next((i+1 for i, e in enumerate(by_recency) if e["uuid"] == correct_uuid), -1)

        correct_cosine = next(e["cosine"] for e in cat_edges if e["uuid"] == correct_uuid)

        print(f"  Cosine rank: {cosine_rank}/{total_in_cat} (cos={correct_cosine:.4f})")
        print(f"  Recency rank: {recency_rank}/{total_in_cat}")
        print()

        results.append((qid, group_label, correct_cat, total_in_cat,
                        cosine_rank, recency_rank, correct_cosine, correct_fact[:60]))

    await driver.close()

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("CATEGORY RANK DIAGNOSTIC: Cosine vs Recency ranking of correct edge within its category")
    print("=" * 120)

    # Header
    print(f"\n{'QID':<16s} {'GRP':<5s} {'CATEGORY':<25s} {'#CAT':>5s} {'COS_RK':>7s} {'REC_RK':>7s} {'COS':>7s} {'DELTA':>7s} {'FACT'}")
    print("-" * 120)

    gains_rows = [r for r in results if r[1] == "GAIN"]
    regr_rows = [r for r in results if r[1] == "REGR"]

    def print_section(label, rows):
        print(f"\n  --- {label} ---")
        for qid, grp, cat, total, cos_rk, rec_rk, cos_val, fact in sorted(rows, key=lambda r: r[4]):
            delta = rec_rk - cos_rk if cos_rk > 0 and rec_rk > 0 else 0
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            cos_rk_str = f"{cos_rk}/{total}" if cos_rk > 0 else "N/A"
            rec_rk_str = f"{rec_rk}/{total}" if rec_rk > 0 else "N/A"
            print(f"  {qid:<16s} {grp:<5s} {cat:<25s} {total:>5d} {cos_rk_str:>7s} {rec_rk_str:>7s} {cos_val:>7.4f} {delta_str:>7s} {fact}")

    print_section("ROUTING GAINS (routing helped these)", gains_rows)
    print_section("ROUTING REGRESSIONS (routing hurt these)", regr_rows)

    # Summary stats
    def avg_ranks(rows):
        cos_ranks = [r[4] for r in rows if r[4] > 0]
        rec_ranks = [r[5] for r in rows if r[5] > 0]
        deltas = [r[5] - r[4] for r in rows if r[4] > 0 and r[5] > 0]
        avg_c = sum(cos_ranks) / len(cos_ranks) if cos_ranks else 0
        avg_r = sum(rec_ranks) / len(rec_ranks) if rec_ranks else 0
        avg_d = sum(deltas) / len(deltas) if deltas else 0
        return avg_c, avg_r, avg_d

    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")

    avg_c_g, avg_r_g, avg_d_g = avg_ranks(gains_rows)
    avg_c_r, avg_r_r, avg_d_r = avg_ranks(regr_rows)

    print(f"  GAINS:       avg cosine_rank={avg_c_g:.1f}  avg recency_rank={avg_r_g:.1f}  avg delta(rec-cos)={avg_d_g:+.1f}")
    print(f"  REGRESSIONS: avg cosine_rank={avg_c_r:.1f}  avg recency_rank={avg_r_r:.1f}  avg delta(rec-cos)={avg_d_r:+.1f}")

    print()
    if avg_d_g > 5 and avg_c_g < 10:
        print("  -> GAINS: cosine ranking STRONGLY helps (correct edge in top-10 by cosine but buried by recency)")
    elif avg_d_g > 0:
        print("  -> GAINS: cosine ranking helps somewhat (correct edge ranks better by cosine than recency)")
    else:
        print("  -> GAINS: cosine ranking does NOT help (correct edge ranks similarly or worse)")

    if avg_c_r < 10:
        print("  -> REGRESSIONS: correct edge already in top-10 by cosine -- routing finds it BUT displaces graphiti results")
    elif avg_c_r > 20:
        print("  -> REGRESSIONS: correct edge ranks poorly by cosine too -- routing won't help regardless")


if __name__ == "__main__":
    asyncio.run(main())
