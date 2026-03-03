"""
audit_recall_vs_ranking.py — Diagnose RECALL vs RANKING failures for Priya's MISS questions.

For each MISS question:
  1. Identifies "correct-answer edges" in Neo4j via keyword matching
  2. Runs build_candidate_pool (the real retrieval pipeline)
  3. Checks if correct edges made it into the pool (RECALL)
  4. Runs rerank_candidates and reports rank of correct edges (RANKING)
  5. Prints the top-5 facts that actually surfaced

Usage: python audit_recall_vs_ranking.py
"""

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

# --- Import real pipeline functions from evaluate_lifemembench ---
# (triggers load_env() and API key checks at import time)
from evaluate_lifemembench import (
    build_candidate_pool,
    rerank_candidates,
    load_edge_cache,
    get_persona_t_now,
    get_graphiti_client,
    Candidate,
    PERSONAS,
    QUESTIONS_PATH,
)
from decay_engine import DecayEngine, TemporalContext
from graphiti_bridge import build_temporal_context
from retrieval_router import RoutingConfig
from openai import AsyncOpenAI

# --- Correct-answer keywords per MISS question (from audit_extraction_ceiling.py) ---
MISS_KEYWORDS = {
    "priya_q01": ["pescatarian", "fish", "vegetarian"],
    "priya_q03": ["adhd", "attention deficit", "attention"],
    "priya_q04": ["tokyo", "march 20", "conference"],
    "priya_q05": ["dog", "landlord", "pet", "golden retriever"],
    "priya_q07": ["rock climbing", "climbing", "bouldering"],
    "priya_q09": ["migraine", "omega-3", "omega"],
    "priya_q10": ["neurips", "paper deadline", "june 2026"],
    "priya_q11": ["rock climbing", "climbing", "tamil", "cooking", "travel"],
    "priya_q12": ["tamil", "chennai", "arjun", "seattle", "brother"],
    "priya_q14": ["tokyo", "march 20", "march 2026", "conference"],
}


async def main():
    # Load questions
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    priya_qs = {q["id"]: q for q in questions if q["persona"] == "priya"}

    # Initialize Graphiti + Neo4j
    graphiti = get_graphiti_client()
    driver = graphiti.driver
    group_id = PERSONAS["priya"]["group_id"]

    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    # Load edge cache + t_now (same as evaluate_persona)
    edge_cache = await load_edge_cache(driver, group_id)
    t_now = await get_persona_t_now(driver, group_id)

    ctx = build_temporal_context(
        last_session_ts=t_now - 86400.0,
        session_message_count=0,
        now=t_now,
    )
    engine = DecayEngine.default()
    routing_config = RoutingConfig(enable_routing=True)
    alpha = 0.1

    # Fetch ALL edges to identify correct-answer UUIDs
    result = await driver.execute_query(
        """
        MATCH (s)-[e:RELATES_TO]->(t)
        WHERE e.group_id = $gid
        RETURN e.uuid AS uuid, e.fact AS fact
        """,
        gid=group_id,
    )
    records = result.records if hasattr(result, "records") else result
    all_edges = {r.data()["uuid"]: r.data()["fact"] for r in records}

    # Identify correct-answer UUIDs per MISS question
    correct_uuids = {}
    for qid, keywords in MISS_KEYWORDS.items():
        uuids = {}
        for uuid, fact in all_edges.items():
            fact_lower = fact.lower()
            matched = [kw for kw in keywords if kw.lower() in fact_lower]
            if matched:
                uuids[uuid] = (fact, matched)
        correct_uuids[qid] = uuids

    print(f"=== RECALL vs RANKING AUDIT: PRIYA (alpha={alpha}, behavioral) ===")
    print(f"Total edges in graph: {len(all_edges)}")
    print(f"Edge cache (enriched): {len(edge_cache)}\n")

    recall_failures = 0
    ranking_failures = 0
    total = 0

    for qid, keywords in MISS_KEYWORDS.items():
        q = priya_qs[qid]
        total += 1

        print(f"{qid} ({q['attack_vector'][:3]}): {q['question']}")
        print(f"  Correct: {q['correct_answer']}")
        print(f"  Correct-answer edges: {len(correct_uuids[qid])} UUIDs")

        # Build candidate pool (real pipeline)
        pool = await build_candidate_pool(
            q["question"], driver, graphiti, group_id,
            xai_client=xai_client, edge_cache=edge_cache,
            routing_config=routing_config,
        )
        pool_uuids = {c.uuid for c in pool}

        # RECALL check
        in_pool = set(correct_uuids[qid].keys()) & pool_uuids
        not_in_pool = set(correct_uuids[qid].keys()) - pool_uuids
        print(f"  Pool size: {len(pool)}")
        print(f"  RECALL: {len(in_pool)}/{len(correct_uuids[qid])} correct edges in pool")

        if not in_pool:
            recall_failures += 1
            print(f"  >>> RECALL FAILURE — no correct edge entered the pool")
            # Show which correct edges were missed
            for uuid in list(not_in_pool)[:3]:
                fact, kws = correct_uuids[qid][uuid]
                print(f"      MISSED: [{uuid[:8]}] ({', '.join(kws)}) \"{fact[:100]}\"")
            print()
            continue

        # Rerank
        reranked = rerank_candidates(pool, ctx, engine, edge_cache, alpha=alpha)

        # RANKING check — find ranks of correct edges
        correct_ranks = []
        for rank, (cand, act, sem, bl) in enumerate(reranked, 1):
            if cand.uuid in correct_uuids[qid]:
                correct_ranks.append((rank, cand, act, sem, bl))

        best_rank = correct_ranks[0][0] if correct_ranks else None
        in_top5 = best_rank is not None and best_rank <= 5

        print(f"  RANKING:")
        for rank, cand, act, sem, bl in correct_ranks[:5]:
            print(f"    #{rank:3d} [{cand.source:15s}] bl={bl:.4f} sem={sem:.4f} "
                  f"act={act:.4f} \"{cand.fact[:80]}\"")

        if in_top5:
            print(f"  >>> OK — correct edge at rank #{best_rank}")
        else:
            ranking_failures += 1
            print(f"  >>> RANKING FAILURE — best correct rank: #{best_rank} (out of top-5)")

        # Print top-5 that actually surfaced
        print(f"  TOP-5:")
        for rank, (cand, act, sem, bl) in enumerate(reranked[:5], 1):
            is_correct = "***" if cand.uuid in correct_uuids[qid] else "   "
            print(f"    #{rank} {is_correct} [{cand.source:15s}] bl={bl:.4f} sem={sem:.4f} "
                  f"act={act:.4f} \"{cand.fact[:90]}\"")

        # Show edges NOT in pool if any
        if not_in_pool:
            print(f"  NOT IN POOL ({len(not_in_pool)}):")
            for uuid in list(not_in_pool)[:3]:
                fact, kws = correct_uuids[qid][uuid]
                print(f"    [{uuid[:8]}] ({', '.join(kws)}) \"{fact[:100]}\"")

        print()

    # Summary
    ok = total - recall_failures - ranking_failures
    print(f"{'='*60}")
    print(f"SUMMARY ({total} MISS questions):")
    print(f"  RECALL failures:  {recall_failures} (correct edge never enters pool)")
    print(f"  RANKING failures: {ranking_failures} (correct edge in pool but not top-5)")
    print(f"  OK (top-5):       {ok}")

    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
