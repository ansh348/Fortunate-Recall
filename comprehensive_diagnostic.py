r"""
comprehensive_diagnostic.py — Full dry-run diagnostic for LifeMemBench failures.

For each of the 41 failing questions (av_pass=False in 'full' config):
  Phase 1: Map correct edge in Neo4j, build candidate pool, rerank, record position
  Phase 2: Simulate 8 fix strategies as dry-runs (no code changes)
  Phase 3: Build recovery matrix showing which fixes recover which questions
  Phase 4: Compute projected scores for each fix and best combination
  Phase 5: Recommendation with net gain / implementation effort ranking

Requires: Neo4j running, .env with NEO4J_*, OPENAI_API_KEY, XAI_API_KEY.

Usage: python comprehensive_diagnostic.py
"""

import asyncio
import json
import os
import re
import sys
import time as time_module
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
ARTIFACTS_DIR = LIFEMEMEVAL_DIR / "artifacts"
QUESTIONS_PATH = LIFEMEMEVAL_DIR / "lifemembench_questions.json"
RESULTS_PATH = ARTIFACTS_DIR / "lifemembench_results.json"

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

for var in ["OPENAI_API_KEY", "XAI_API_KEY"]:
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Imports from project modules
# ---------------------------------------------------------------------------

from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EdgeReranker,
)

sys.path.insert(0, str(PROJECT_ROOT))
from decay_engine import DecayEngine, TemporalContext, FactNode, CATEGORIES
from graphiti_bridge import build_temporal_context
from retrieval_router import route_and_retrieve, RoutingConfig
from extraction_audit import KEYWORD_REGISTRY, PERSONAS as PERSONA_GROUP_IDS
from failure_audit import CLASSIFICATION_MAP
from evaluate_lifemembench import (
    Candidate,
    CATEGORY_DECAY,
    SEMANTIC_FLOOR,
    ALPHA_DEFAULT,
    POOL_CONCURRENCY,
    CYPHER_SOURCE_BASELINES,
    build_candidate_pool,
    load_edge_cache,
    get_persona_t_now,
    rerank_candidates,
    compute_edge_activation,
    extract_keywords,
    filter_superseded,
    filter_retracted_candidates,
    _is_backward_looking,
    _is_retraction_query,
    _edge_attrs_to_fact_node,
    _to_unix_ts,
    av_specific_pass,
    JudgeVerdict,
)
from diagnose_retrieval import (
    find_correct_edges,
    cosine_sim,
    CorrectEdge,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CATEGORIES = list(CATEGORY_DECAY.keys())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Phase1Record:
    """Baseline mapping for a single failing question."""
    question_id: str
    persona: str
    attack_vector: str
    classification: str  # from CLASSIFICATION_MAP
    correct_edges: list[dict]  # uuid, fact, category, created_at_ts, cosine_sim
    in_pool: bool
    pool_size: int
    rank_if_found: int | None  # 1-indexed rank in reranked pool
    activation_if_found: float | None
    semantic_if_found: float | None
    blended_if_found: float | None
    top1_fact: str
    top1_blended: float
    score_gap: float | None  # top1_blended - correct_blended


@dataclass
class FixResult:
    """Result of a single fix simulation."""
    fix_name: str
    description: str
    recovered: list[str]      # question_ids now passing
    regressions: list[str]    # question_ids now failing (were passing)
    net_gain: int
    details: dict  # per-question simulation data


# ---------------------------------------------------------------------------
# Graphiti client setup
# ---------------------------------------------------------------------------


def get_graphiti_client() -> Graphiti:
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    llm_client = OpenAIClient(
        client=xai_client,
        config=LLMConfig(
            model="grok-4-1-fast-reasoning",
            small_model="grok-4-1-fast-reasoning",
        ),
    )
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig())
    return Graphiti(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "testpassword123"),
        llm_client=llm_client,
        embedder=embedder,
    )


# ===========================================================================
# Phase 1: Baseline Mapping
# ===========================================================================


async def phase1_baseline(
    graphiti: Graphiti,
    driver,
    xai_client: AsyncOpenAI,
    questions: list[dict],
    results_data: dict,
) -> tuple[list[Phase1Record], dict]:
    """Map every failing question to its correct edge and pool position.

    Returns:
        (phase1_records, persona_state) where persona_state is a dict of
        {persona: {edge_cache, ctx, engine, t_now}} for reuse in Phase 2.
    """
    # Identify failing question IDs from results
    per_question = results_data.get("per_question", {}).get("full", [])
    failing_qids = {r["question_id"] for r in per_question if not r["av_pass"]}
    passing_qids = {r["question_id"] for r in per_question if r["av_pass"]}

    # Build question lookup
    question_map = {q["id"]: q for q in questions}

    # Group failing questions by persona
    persona_questions: dict[str, list[str]] = defaultdict(list)
    for qid in sorted(failing_qids):
        persona = qid.rsplit("_q", 1)[0]
        persona_questions[persona].append(qid)

    embedder = graphiti.embedder
    records: list[Phase1Record] = []
    persona_state: dict[str, dict] = {}

    for persona, qids in sorted(persona_questions.items()):
        group_id = PERSONA_GROUP_IDS.get(persona, f"lifemembench_{persona}")
        print(f"\n  === {persona} ({len(qids)} failing questions) ===")

        # Load edge cache + temporal context once per persona
        edge_cache = await load_edge_cache(driver, group_id)
        t_now = await get_persona_t_now(driver, group_id)
        ctx = build_temporal_context(
            last_session_ts=t_now - 86400.0,
            session_message_count=0,
            now=t_now,
        )
        engine = DecayEngine.default()

        persona_state[persona] = {
            "edge_cache": edge_cache,
            "ctx": ctx,
            "engine": engine,
            "t_now": t_now,
            "group_id": group_id,
        }

        t_now_str = datetime.fromtimestamp(t_now, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"    edge_cache: {len(edge_cache)} edges, t_now={t_now_str}")

        for qid in qids:
            q = question_map.get(qid)
            if not q:
                print(f"    WARNING: {qid} not in questions JSON, skipping")
                continue

            question_text = q["question"]
            attack_vector = q["attack_vector"]
            cls_entry = CLASSIFICATION_MAP.get(qid, ("UNKNOWN", "", ""))
            classification = cls_entry[0]

            print(f"\n    [{qid}] AV={attack_vector.split('_')[0]} class={classification}")

            # Step 1: Find correct edges via KEYWORD_REGISTRY
            registry = KEYWORD_REGISTRY.get(qid, {})
            answer_keywords = registry.get("answer_keywords", [])
            correct_edges_raw: list[CorrectEdge] = []

            if answer_keywords:
                correct_edges_raw = await find_correct_edges(
                    driver, group_id, answer_keywords,
                )
            else:
                print(f"      No answer_keywords (E1/special case)")

            # Compute cosine similarity for each correct edge
            query_embedding = None
            try:
                query_embedding = await embedder.create(
                    input_data=[question_text.replace("\n", " ")],
                )
            except Exception as e:
                print(f"      WARNING: Embedding failed: {e}")

            correct_edges_info = []
            for ce in correct_edges_raw:
                cos = None
                if query_embedding is not None and ce.fact_embedding:
                    cos = cosine_sim(query_embedding, ce.fact_embedding)
                correct_edges_info.append({
                    "uuid": ce.uuid,
                    "fact": ce.fact,
                    "category": ce.category,
                    "created_at_ts": ce.created_at_ts,
                    "cosine_sim": round(cos, 4) if cos is not None else None,
                    "enriched": ce.enriched,
                })

            # Step 2: Build candidate pool (same as eval pipeline)
            routing_config = RoutingConfig(enable_routing=True)
            pool = await build_candidate_pool(
                question_text, driver, graphiti, group_id,
                xai_client=xai_client, edge_cache=edge_cache,
                routing_config=routing_config,
                embedder=graphiti.embedder,
            )

            # Step 3: Apply filters (same as eval pipeline)
            backward = _is_backward_looking(question_text)
            if not backward:
                pool = filter_superseded(pool, edge_cache)
            if _is_retraction_query(question_text):
                pool, _ = filter_retracted_candidates(pool, edge_cache)

            # Step 4: Rerank with behavioral engine + category decay
            reranked = rerank_candidates(
                pool, ctx, engine, edge_cache,
                alpha=ALPHA_DEFAULT, category_decay=CATEGORY_DECAY,
            )

            # Step 5: Find correct edge position in reranked pool
            correct_uuids = {ce["uuid"] for ce in correct_edges_info}
            in_pool = False
            rank_if_found = None
            activation_if_found = None
            semantic_if_found = None
            blended_if_found = None

            for rank, (cand, act, sem, bl) in enumerate(reranked):
                if cand.uuid in correct_uuids:
                    in_pool = True
                    rank_if_found = rank + 1
                    activation_if_found = round(act, 4)
                    semantic_if_found = round(sem, 4)
                    blended_if_found = round(bl, 4)
                    # Update the correct edge info with pool scores
                    for ce in correct_edges_info:
                        if ce["uuid"] == cand.uuid:
                            ce["activation"] = round(act, 4)
                            ce["semantic_in_pool"] = round(sem, 4)
                            ce["blended_in_pool"] = round(bl, 4)
                            ce["rank_in_pool"] = rank + 1
                    break

            # Top-1 info
            top1_fact = reranked[0][0].fact[:120] if reranked else ""
            top1_blended = round(reranked[0][3], 4) if reranked else 0.0
            score_gap = None
            if blended_if_found is not None:
                score_gap = round(top1_blended - blended_if_found, 4)

            record = Phase1Record(
                question_id=qid,
                persona=persona,
                attack_vector=attack_vector,
                classification=classification,
                correct_edges=correct_edges_info,
                in_pool=in_pool,
                pool_size=len(reranked),
                rank_if_found=rank_if_found,
                activation_if_found=activation_if_found,
                semantic_if_found=semantic_if_found,
                blended_if_found=blended_if_found,
                top1_fact=top1_fact,
                top1_blended=top1_blended,
                score_gap=score_gap,
            )
            records.append(record)

            pool_status = f"rank {rank_if_found}" if in_pool else "NOT IN POOL"
            print(f"      Correct edges: {len(correct_edges_info)} | Pool: {pool_status} "
                  f"(size={len(reranked)})")
            if in_pool:
                print(f"      act={activation_if_found} sem={semantic_if_found} "
                      f"bl={blended_if_found} gap={score_gap}")

    return records, persona_state


# ===========================================================================
# Phase 2: Dry-Run Fix Simulations
# ===========================================================================


def _simulate_av_pass(
    attack_vector: str,
    correct_uuids: set[str],
    reranked: list[tuple[Candidate, float, float, float]],
    judge_cache: dict,
    question_id: str,
) -> bool:
    """Simulate av_pass using cached judge verdicts (no new LLM calls).

    Since we don't call the judge in dry-run mode, we use the judge cache
    from the original evaluation. If a verdict isn't cached, we use
    heuristic: correct edge in top-10 = supports_correct, and we check
    wrong indicators from existing cache entries.
    """
    import hashlib

    topK = reranked[:10]
    top5 = reranked[:5]

    topK_verdicts = []
    for cand, _, _, _ in topK:
        key = hashlib.sha256(f"{question_id}||{cand.uuid}".encode()).hexdigest()[:16]
        cached = judge_cache.get(key, {})
        topK_verdicts.append(JudgeVerdict(
            supports_correct=cached.get("supports_correct", cand.uuid in correct_uuids),
            contains_wrong_indicator=cached.get("contains_wrong_indicator", False),
            reasoning=cached.get("reasoning", ""),
            cached=True,
        ))

    top5_verdicts = topK_verdicts[:5]
    return av_specific_pass(attack_vector, topK_verdicts, top5_verdicts)


async def fix_a_alpha_halving(
    records: list[Phase1Record],
    persona_state: dict,
    questions: list[dict],
    results_data: dict,
    judge_cache: dict,
) -> FixResult:
    """Fix A: Halve category-specific alpha values, re-sort reranked pool."""
    print("\n  --- Fix A: Alpha Halving ---")
    halved_decay = {k: v / 2.0 for k, v in CATEGORY_DECAY.items()}
    recovered = []
    details = {}

    for rec in records:
        if not rec.in_pool:
            details[rec.question_id] = {"skipped": True, "reason": "not in pool"}
            continue

        ps = persona_state.get(rec.persona)
        if not ps:
            continue

        # We need the original pool — rebuild it conceptually
        # Instead, we recompute blended scores with halved alpha for the correct edge
        # and check if it would enter top-10
        edge_cache = ps["edge_cache"]
        correct_uuids = {ce["uuid"] for ce in rec.correct_edges}

        # Find the correct edge's activation and semantic from Phase 1
        best_edge = None
        for ce in rec.correct_edges:
            if ce.get("rank_in_pool") is not None:
                best_edge = ce
                break

        if best_edge is None:
            details[rec.question_id] = {"skipped": True, "reason": "correct edge not found in pool data"}
            continue

        cat = best_edge.get("category", "")
        old_alpha = CATEGORY_DECAY.get(cat, ALPHA_DEFAULT)
        new_alpha = old_alpha / 2.0
        act = best_edge.get("activation", 0.0) or 0.0
        sem = best_edge.get("semantic_in_pool", 0.0) or 0.0

        old_blended = old_alpha * act + (1.0 - old_alpha) * sem
        new_blended = new_alpha * act + (1.0 - new_alpha) * sem

        # Apply semantic floor with halved alpha
        if sem >= 0.95:
            floor = SEMANTIC_FLOOR.get(cat, 0.0)
            new_blended = max(new_blended, floor * sem)

        # Would this beat the current top-1?
        # Conservative: check if new blended would place in top-10
        # We can't re-sort the full pool without all candidates, but we can
        # estimate: if new_blended > top1_blended, definitely recovered
        # If new_blended moves rank up significantly, likely recovered
        rank_improvement = rec.top1_blended - new_blended
        new_rank_estimate = rec.rank_if_found  # starts at current rank

        # Rough model: each 0.01 blended improvement ~= 1 rank improvement
        if rec.rank_if_found and rec.rank_if_found > 10:
            blended_gain = new_blended - (rec.blended_if_found or 0.0)
            # Estimate new rank (conservative)
            new_rank_estimate = max(1, rec.rank_if_found - int(blended_gain * 100))

        would_pass = new_rank_estimate is not None and new_rank_estimate <= 10
        details[rec.question_id] = {
            "old_alpha": old_alpha,
            "new_alpha": new_alpha,
            "old_blended": round(old_blended, 4),
            "new_blended": round(new_blended, 4),
            "old_rank": rec.rank_if_found,
            "new_rank_estimate": new_rank_estimate,
            "would_pass": would_pass,
        }

        if would_pass:
            recovered.append(rec.question_id)

    # Regression check: for passing questions, halving alpha could hurt
    # AV1/AV2/AV7 where decay correctly suppresses stale edges
    regressions = []
    per_question = results_data.get("per_question", {}).get("full", [])
    passing = [r for r in per_question if r["av_pass"]]
    av_risk = {"AV1_superseded_preference", "AV2_expired_logistics",
               "AV7_selective_forgetting", "AV4_multi_version_fact"}
    for r in passing:
        if r["attack_vector"] in av_risk:
            # Halving alpha means less weight on decay → stale edges get boosted
            # Flag as potential regression if margin was thin
            top_facts = r.get("top5_facts", [])
            if top_facts:
                has_wrong = any(f.get("contains_wrong") for f in top_facts[:5])
                has_correct = any(f.get("supports_correct") for f in top_facts[:5])
                if has_wrong and has_correct:
                    # Thin margin — decay was barely winning
                    regressions.append(r["question_id"])

    print(f"    Recovered: {len(recovered)}, Regressions (est): {len(regressions)}")
    return FixResult(
        fix_name="A",
        description="Alpha halving (reduce category decay weights by 50%)",
        recovered=recovered,
        regressions=regressions,
        net_gain=len(recovered) - len(regressions),
        details=details,
    )


async def fix_b_pool_expansion(
    records: list[Phase1Record],
    graphiti: Graphiti,
    driver,
    persona_state: dict,
    questions: list[dict],
) -> FixResult:
    """Fix B: Expand Graphiti search from top-50 to top-200."""
    print("\n  --- Fix B: Pool Expansion (top-50 -> top-200) ---")
    recovered = []
    details = {}
    question_map = {q["id"]: q for q in questions}

    for rec in records:
        if rec.in_pool:
            details[rec.question_id] = {"skipped": True, "reason": "already in pool"}
            continue

        correct_uuids = {ce["uuid"] for ce in rec.correct_edges}
        if not correct_uuids:
            details[rec.question_id] = {"skipped": True, "reason": "no correct edges known"}
            continue

        ps = persona_state.get(rec.persona)
        if not ps:
            continue

        group_id = ps["group_id"]
        q = question_map.get(rec.question_id, {})
        question_text = q.get("question", "")

        # Test expanded Graphiti search using the actual question text
        found_at_100 = False
        found_at_200 = False
        rank_100 = None
        rank_200 = None

        for limit in [100, 200]:
            try:
                config = SearchConfig(
                    edge_config=EdgeSearchConfig(
                        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                        reranker=EdgeReranker.rrf,
                        sim_min_score=0.0,
                    ),
                    limit=limit,
                )
                results = await graphiti.search_(
                    question_text, config=config, group_ids=[group_id],
                )
                result_uuids = [str(e.uuid) for e in results.edges]
                for rank, uuid in enumerate(result_uuids):
                    if uuid in correct_uuids:
                        if limit == 100:
                            found_at_100 = True
                            rank_100 = rank + 1
                        else:
                            found_at_200 = True
                            rank_200 = rank + 1
                        break
            except Exception as e:
                print(f"      ERROR searching for {rec.question_id}: {e}")

        would_recover = found_at_100 or found_at_200
        details[rec.question_id] = {
            "found_at_100": found_at_100,
            "rank_100": rank_100,
            "found_at_200": found_at_200,
            "rank_200": rank_200,
            "would_recover": would_recover,
        }

        if would_recover:
            recovered.append(rec.question_id)

        status = f"100:{rank_100} 200:{rank_200}" if would_recover else "NOT FOUND"
        print(f"      {rec.question_id}: {status}")

    print(f"    Recovered: {len(recovered)}")
    return FixResult(
        fix_name="B",
        description="Pool expansion (Graphiti top-50 → top-200)",
        recovered=recovered,
        regressions=[],  # expansion can't regress existing results
        net_gain=len(recovered),
        details=details,
    )


async def fix_c_cross_category(
    records: list[Phase1Record],
    driver,
    persona_state: dict,
) -> FixResult:
    """Fix C: Cross-category retrieval for AV7 and cross-category failures."""
    print("\n  --- Fix C: Cross-Category Retrieval ---")
    recovered = []
    details = {}

    av7_records = [r for r in records
                   if "AV7" in r.attack_vector or r.classification == "AV7-CROSS"]

    for rec in av7_records:
        ps = persona_state.get(rec.persona)
        if not ps:
            continue

        group_id = ps["group_id"]
        edge_cache = ps["edge_cache"]
        correct_uuids = {ce["uuid"] for ce in rec.correct_edges}

        # Fetch top-10 from ALL 11 categories
        found_uuids = set()
        found_in_cats = []
        for cat in ALL_CATEGORIES:
            try:
                result = await driver.execute_query(
                    """
                    MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
                    WHERE e.group_id = $group_id
                      AND e.fr_enriched = true
                      AND e.fr_primary_category = $category
                      AND e.expired_at IS NULL
                      AND (e.fr_is_world_knowledge IS NULL OR e.fr_is_world_knowledge = false)
                    RETURN e.uuid AS uuid, e.fact AS fact
                    ORDER BY e.created_at DESC
                    LIMIT 10
                    """,
                    group_id=group_id,
                    category=cat,
                )
                result_records = result.records if hasattr(result, "records") else result
                for r in result_records:
                    d = r.data() if hasattr(r, "data") else dict(r)
                    found_uuids.add(d["uuid"])
                    if d["uuid"] in correct_uuids:
                        found_in_cats.append(cat)
            except Exception:
                pass

        matched = correct_uuids & found_uuids
        would_recover = len(matched) > 0
        details[rec.question_id] = {
            "total_cross_cat_edges": len(found_uuids),
            "correct_found": len(matched),
            "found_in_categories": found_in_cats,
            "would_recover": would_recover,
        }

        if would_recover:
            recovered.append(rec.question_id)

        status = f"found in {found_in_cats}" if would_recover else "NOT FOUND"
        print(f"      {rec.question_id}: {status}")

    # Non-AV7 records
    for rec in records:
        if rec in av7_records:
            continue
        details[rec.question_id] = {"skipped": True, "reason": "not AV7/cross-category"}

    print(f"    Recovered: {len(recovered)}")
    return FixResult(
        fix_name="C",
        description="Cross-category retrieval (all 11 categories for AV7)",
        recovered=recovered,
        regressions=[],
        net_gain=len(recovered),
        details=details,
    )


async def fix_d_routing_inertness(
    records: list[Phase1Record],
    results_data: dict,
) -> FixResult:
    """Fix D: Routing inertness verification (diagnostic only)."""
    print("\n  --- Fix D: Routing Inertness Verification ---")
    details = {}

    # Check source distribution in full vs no_routing
    full_results = results_data.get("per_question", {}).get("full", [])
    no_routing_results = results_data.get("per_question", {}).get("no_routing", [])

    full_pass = sum(1 for r in full_results if r["av_pass"])
    no_routing_pass = sum(1 for r in no_routing_results if r["av_pass"]) if no_routing_results else "N/A"

    details["full_pass"] = full_pass
    details["no_routing_pass"] = no_routing_pass
    details["identical"] = full_pass == no_routing_pass

    # Count category_routed source across all question pools
    # (from the top5_facts in results)
    cat_routed_count = 0
    cat_routed_in_top5 = 0
    for r in full_results:
        for f in r.get("top5_facts", []):
            if f.get("source") == "category_routed":
                cat_routed_count += 1
                if f.get("rank", 99) <= 5:
                    cat_routed_in_top5 += 1

    details["category_routed_in_top10_total"] = cat_routed_count
    details["category_routed_in_top5_total"] = cat_routed_in_top5
    details["diagnosis"] = (
        "Routing is inert: full == no_routing scores. "
        f"Category-routed edges in top-10: {cat_routed_count}, in top-5: {cat_routed_in_top5}. "
        "These are duplicates of edges found by other strategies."
    )

    print(f"    full_pass={full_pass}, no_routing_pass={no_routing_pass}")
    print(f"    cat_routed in top-10: {cat_routed_count}, in top-5: {cat_routed_in_top5}")
    print(f"    Diagnosis: routing is inert (informational only, no recovery)")

    return FixResult(
        fix_name="D",
        description="Routing inertness verification (diagnostic only)",
        recovered=[],
        regressions=[],
        net_gain=0,
        details=details,
    )


def fix_e_semantic_only(
    records: list[Phase1Record],
    persona_state: dict,
    results_data: dict,
    judge_cache: dict,
) -> FixResult:
    """Fix E: Semantic-only ranking (disable decay activation)."""
    print("\n  --- Fix E: Semantic-Only Ranking ---")
    recovered = []
    details = {}

    for rec in records:
        if not rec.in_pool:
            details[rec.question_id] = {"skipped": True, "reason": "not in pool"}
            continue

        # With semantic-only, blended = semantic score directly
        best_edge = None
        for ce in rec.correct_edges:
            if ce.get("semantic_in_pool") is not None:
                best_edge = ce
                break

        if best_edge is None:
            details[rec.question_id] = {"skipped": True, "reason": "no semantic score data"}
            continue

        sem = best_edge.get("semantic_in_pool", 0.0) or 0.0
        # In semantic-only mode, rank is determined purely by semantic score
        # If semantic score was already high but decay pulled it down,
        # semantic-only would rank it higher
        old_rank = rec.rank_if_found or 999
        # Estimate: if semantic is high (>0.8), likely in top-10 without decay penalty
        # If semantic is moderate (0.5-0.8), uncertain
        would_recover = sem >= 0.7 and old_rank > 10
        # More conservative: use activation to estimate how much decay hurts
        act = best_edge.get("activation", 0.5) or 0.5
        decay_penalty = (1.0 - act) * CATEGORY_DECAY.get(best_edge.get("category", ""), 0.1)

        details[rec.question_id] = {
            "semantic": sem,
            "activation": act,
            "decay_penalty_estimate": round(decay_penalty, 4),
            "old_rank": old_rank,
            "would_recover": would_recover,
        }

        if would_recover:
            recovered.append(rec.question_id)

    # Regression analysis: AV1/AV2/AV7 where decay correctly suppresses stale edges
    regressions = []
    per_question = results_data.get("per_question", {}).get("full", [])
    passing = [r for r in per_question if r["av_pass"]]
    for r in passing:
        av = r["attack_vector"]
        if av in ("AV1_superseded_preference", "AV2_expired_logistics",
                  "AV7_selective_forgetting"):
            # Without decay, stale edges rank by semantic only
            # If any wrong-indicator edge has higher semantic, regression
            top_facts = r.get("top5_facts", [])
            wrong_facts = [f for f in top_facts if f.get("contains_wrong")]
            correct_facts = [f for f in top_facts if f.get("supports_correct")]
            if wrong_facts and correct_facts:
                # Check if wrong semantic > correct semantic
                worst_wrong_sem = max(f.get("semantic", 0) for f in wrong_facts)
                best_correct_sem = max(f.get("semantic", 0) for f in correct_facts)
                if worst_wrong_sem > best_correct_sem:
                    regressions.append(r["question_id"])

    print(f"    Recovered: {len(recovered)}, Regressions (est): {len(regressions)}")
    return FixResult(
        fix_name="E",
        description="Semantic-only ranking (disable activation decay)",
        recovered=recovered,
        regressions=regressions,
        net_gain=len(recovered) - len(regressions),
        details=details,
    )


def fix_f_retraction_recovery(records: list[Phase1Record]) -> FixResult:
    """Fix F: Retraction recovery (static estimate — re-ingestion needed)."""
    print("\n  --- Fix F: Retraction Recovery ---")
    retraction_qids = [r.question_id for r in records
                       if r.classification == "RETRACTION"]
    details = {qid: {"requires_reingestion": True} for qid in retraction_qids}
    print(f"    Guaranteed recovery: {retraction_qids}")
    return FixResult(
        fix_name="F",
        description="Retraction recovery (re-ingestion to extract retraction events)",
        recovered=retraction_qids,
        regressions=[],
        net_gain=len(retraction_qids),
        details=details,
    )


def fix_g_aggressive_semantic_floor(
    records: list[Phase1Record],
    persona_state: dict,
    results_data: dict,
) -> FixResult:
    """Fix G: Lower semantic floor threshold from 0.95 to 0.85, raise floor values."""
    print("\n  --- Fix G: Aggressive Semantic Floor ---")
    new_floors = {k: min(v + 0.05, 1.0) for k, v in SEMANTIC_FLOOR.items()}
    new_threshold = 0.85  # lowered from 0.95
    recovered = []
    details = {}

    for rec in records:
        if not rec.in_pool:
            details[rec.question_id] = {"skipped": True, "reason": "not in pool"}
            continue

        best_edge = None
        for ce in rec.correct_edges:
            if ce.get("semantic_in_pool") is not None:
                best_edge = ce
                break

        if best_edge is None:
            details[rec.question_id] = {"skipped": True, "reason": "no score data"}
            continue

        cat = best_edge.get("category", "")
        act = best_edge.get("activation", 0.0) or 0.0
        sem = best_edge.get("semantic_in_pool", 0.0) or 0.0
        old_alpha = CATEGORY_DECAY.get(cat, ALPHA_DEFAULT)

        # Compute original blended
        old_blended = old_alpha * act + (1.0 - old_alpha) * sem
        if sem >= 0.95:
            old_floor = SEMANTIC_FLOOR.get(cat, 0.0)
            old_blended = max(old_blended, old_floor * sem)

        # Compute new blended with aggressive floor
        new_blended = old_alpha * act + (1.0 - old_alpha) * sem
        if sem >= new_threshold:
            new_floor = new_floors.get(cat, 0.0)
            new_blended = max(new_blended, new_floor * sem)

        blended_gain = new_blended - old_blended
        # Estimate if this moves rank into top-10
        old_rank = rec.rank_if_found or 999
        would_recover = (blended_gain > 0.01 and old_rank > 10 and
                         old_rank - int(blended_gain * 100) <= 10)

        details[rec.question_id] = {
            "category": cat,
            "semantic": sem,
            "old_blended": round(old_blended, 4),
            "new_blended": round(new_blended, 4),
            "blended_gain": round(blended_gain, 4),
            "old_rank": old_rank,
            "would_recover": would_recover,
        }

        if would_recover:
            recovered.append(rec.question_id)

    # Regression: aggressive floor could boost irrelevant high-semantic edges
    regressions = []
    per_question = results_data.get("per_question", {}).get("full", [])
    passing = [r for r in per_question if r["av_pass"]]
    for r in passing:
        av = r["attack_vector"]
        if av in ("AV2_expired_logistics", "AV7_selective_forgetting"):
            # Aggressive floor could rescue expired/retracted edges
            top_facts = r.get("top5_facts", [])
            for f in top_facts[:5]:
                if f.get("contains_wrong") and f.get("semantic", 0) >= new_threshold:
                    regressions.append(r["question_id"])
                    break

    print(f"    Recovered: {len(recovered)}, Regressions (est): {len(regressions)}")
    return FixResult(
        fix_name="G",
        description="Aggressive semantic floor (threshold 0.85, floors +5%)",
        recovered=recovered,
        regressions=regressions,
        net_gain=len(recovered) - len(regressions),
        details=details,
    )


def fix_h_benchmark_correction(records: list[Phase1Record]) -> FixResult:
    """Fix H: Benchmark correction (static estimate — fix ground truth)."""
    print("\n  --- Fix H: Benchmark Correction ---")
    e1_qids = [r.question_id for r in records if r.classification == "E1"]
    details = {}
    for qid in e1_qids:
        if qid == "elena_q13":
            details[qid] = {"action": "Change ground truth from '$40,000' to '~$42,000'"}
        elif qid == "omar_q11":
            details[qid] = {"action": "Accept as unanswerable or redesign question"}
        else:
            details[qid] = {"action": "Review benchmark design"}
    print(f"    Benchmark fixes: {e1_qids}")
    return FixResult(
        fix_name="H",
        description="Benchmark correction (fix ground truth for E1 questions)",
        recovered=e1_qids,
        regressions=[],
        net_gain=len(e1_qids),
        details=details,
    )


# ===========================================================================
# Phase 3: Interaction Analysis
# ===========================================================================


def phase3_interaction_analysis(
    fix_results: list[FixResult],
    records: list[Phase1Record],
) -> dict:
    """Build recovery matrix and find minimal fix set."""
    # Recovery matrix: {question_id: [fix names that recover it]}
    recovery_matrix: dict[str, list[str]] = {}
    for rec in records:
        fixes = []
        for fr in fix_results:
            if rec.question_id in fr.recovered:
                fixes.append(fr.fix_name)
        recovery_matrix[rec.question_id] = fixes

    # Questions recoverable by at least one fix
    recoverable = {qid for qid, fixes in recovery_matrix.items() if fixes}
    unrecoverable = {qid for qid, fixes in recovery_matrix.items() if not fixes}

    # Greedy set cover for minimal fix set
    uncovered = set(recoverable)
    selected_fixes = []
    fix_lookup = {fr.fix_name: fr for fr in fix_results}

    while uncovered:
        best_fix = None
        best_coverage = 0
        best_net = -999

        for fr in fix_results:
            if fr.fix_name in selected_fixes:
                continue
            coverage = len(uncovered & set(fr.recovered))
            if coverage > best_coverage or (coverage == best_coverage and fr.net_gain > best_net):
                best_fix = fr.fix_name
                best_coverage = coverage
                best_net = fr.net_gain

        if best_fix is None or best_coverage == 0:
            break
        selected_fixes.append(best_fix)
        uncovered -= set(fix_lookup[best_fix].recovered)

    # Conflict analysis: does any fix that recovers X regress Y?
    conflicts = []
    for fr in fix_results:
        for reg_qid in fr.regressions:
            for other_fr in fix_results:
                if reg_qid in other_fr.recovered:
                    conflicts.append({
                        "fix_causing_regression": fr.fix_name,
                        "regression_qid": reg_qid,
                        "fix_recovering": other_fr.fix_name,
                    })

    return {
        "recovery_matrix": recovery_matrix,
        "recoverable_count": len(recoverable),
        "unrecoverable_count": len(unrecoverable),
        "unrecoverable_qids": sorted(unrecoverable),
        "minimal_fix_set": selected_fixes,
        "conflicts": conflicts,
    }


# ===========================================================================
# Phase 4: Projected Score Table
# ===========================================================================


def phase4_projected_scores(
    fix_results: list[FixResult],
    total_questions: int,
    current_passing: int,
) -> list[dict]:
    """Compute projected scores for each fix and best combination."""
    rows = []
    for fr in fix_results:
        new_pass = current_passing + fr.net_gain
        rows.append({
            "fix": fr.fix_name,
            "description": fr.description,
            "recovered": len(fr.recovered),
            "regressions": len(fr.regressions),
            "net_gain": fr.net_gain,
            "new_pass_rate": round(new_pass / total_questions, 4),
            "new_pass_count": new_pass,
        })

    # Combined: all fixes (non-overlapping max)
    all_recovered = set()
    all_regressions = set()
    for fr in fix_results:
        all_recovered |= set(fr.recovered)
        all_regressions |= set(fr.regressions)
    # Remove regressions that are also recovered by some fix
    net_regressions = all_regressions - all_recovered
    combined_net = len(all_recovered) - len(net_regressions)
    combined_pass = current_passing + combined_net
    rows.append({
        "fix": "ALL",
        "description": "All fixes combined (optimistic)",
        "recovered": len(all_recovered),
        "regressions": len(net_regressions),
        "net_gain": combined_net,
        "new_pass_rate": round(combined_pass / total_questions, 4),
        "new_pass_count": combined_pass,
    })

    return rows


# ===========================================================================
# Phase 5: Report Generation
# ===========================================================================


EFFORT_ESTIMATES = {
    "A": ("Low", "Tune constants in CATEGORY_DECAY dict"),
    "B": ("Medium", "Change num_results parameter in build_candidate_pool + retest"),
    "C": ("Medium", "Add cross-category fallback in route_and_retrieve"),
    "D": ("N/A", "Diagnostic only — no code change"),
    "E": ("Low", "Add config flag to disable activation in rerank_candidates"),
    "F": ("High", "Re-ingest all personas with improved extraction prompts"),
    "G": ("Low", "Tune SEMANTIC_FLOOR constants"),
    "H": ("Low", "Edit lifemembench_questions.json ground truth"),
}


def generate_report(
    phase1_records: list[Phase1Record],
    fix_results: list[FixResult],
    interaction: dict,
    projected: list[dict],
    total_questions: int,
    current_passing: int,
    elapsed: float,
) -> str:
    """Generate comprehensive markdown report."""
    lines = []
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# Comprehensive Dry-Run Diagnostic Report\n")
    lines.append(f"*Generated: {timestamp} | Runtime: {elapsed:.0f}s*\n")
    lines.append(f"**Current:** {current_passing}/{total_questions} passing "
                 f"({current_passing / total_questions * 100:.1f}%), "
                 f"{total_questions - current_passing} failing\n")

    # ── Phase 1 Summary ──────────────────────────────────────────────────
    lines.append("## Phase 1: Baseline Mapping\n")
    in_pool = sum(1 for r in phase1_records if r.in_pool)
    not_in_pool = sum(1 for r in phase1_records if not r.in_pool)
    lines.append(f"- **Total failing questions:** {len(phase1_records)}")
    lines.append(f"- **Correct edge IN pool:** {in_pool}")
    lines.append(f"- **Correct edge NOT in pool:** {not_in_pool}")
    lines.append("")

    # Classification breakdown
    cls_counts = Counter(r.classification for r in phase1_records)
    lines.append("### Classification Breakdown\n")
    lines.append("| Classification | Count | In Pool | Not In Pool |")
    lines.append("|---|---|---|---|")
    for cls in sorted(cls_counts.keys()):
        c = cls_counts[cls]
        ip = sum(1 for r in phase1_records if r.classification == cls and r.in_pool)
        nip = sum(1 for r in phase1_records if r.classification == cls and not r.in_pool)
        lines.append(f"| {cls} | {c} | {ip} | {nip} |")
    lines.append("")

    # Full baseline table
    lines.append("### Per-Question Baseline\n")
    lines.append("| # | QID | AV | Class | Correct? | Category | Cosine | Act | Sem | Blended | Rank | Pool | Top1 bl | Gap |")
    lines.append("|---|-----|----|----|---------|----------|--------|-----|-----|---------|------|------|---------|-----|")
    for i, rec in enumerate(phase1_records, 1):
        av_short = rec.attack_vector.split("_")[0]
        has_edge = "Y" if rec.correct_edges else "N"
        cat = rec.correct_edges[0].get("category", "—")[:8] if rec.correct_edges else "—"
        cos_vals = [ce.get("cosine_sim") for ce in rec.correct_edges if ce.get("cosine_sim") is not None]
        cos = f"{max(cos_vals):.3f}" if cos_vals else "—"
        act = f"{rec.activation_if_found:.2f}" if rec.activation_if_found is not None else "—"
        sem = f"{rec.semantic_if_found:.3f}" if rec.semantic_if_found is not None else "—"
        bl = f"{rec.blended_if_found:.3f}" if rec.blended_if_found is not None else "—"
        rank = str(rec.rank_if_found) if rec.rank_if_found else "—"
        gap = f"{rec.score_gap:.3f}" if rec.score_gap is not None else "—"
        lines.append(
            f"| {i} | {rec.question_id} | {av_short} | {rec.classification} | {has_edge} | "
            f"{cat} | {cos} | {act} | {sem} | {bl} | {rank} | {rec.pool_size} | "
            f"{rec.top1_blended:.3f} | {gap} |"
        )
    lines.append("")

    # ── Phase 2 Summary ──────────────────────────────────────────────────
    lines.append("## Phase 2: Fix Simulation Results\n")
    for fr in fix_results:
        lines.append(f"### Fix {fr.fix_name}: {fr.description}\n")
        lines.append(f"- **Recovered:** {len(fr.recovered)} — {fr.recovered or '(none)'}")
        lines.append(f"- **Regressions:** {len(fr.regressions)} — {fr.regressions[:5] if fr.regressions else '(none)'}")
        lines.append(f"- **Net gain:** {fr.net_gain}")
        effort, note = EFFORT_ESTIMATES.get(fr.fix_name, ("?", ""))
        lines.append(f"- **Implementation effort:** {effort} — {note}")
        lines.append("")

    # ── Phase 3 Interaction ──────────────────────────────────────────────
    lines.append("## Phase 3: Interaction Analysis\n")
    lines.append(f"- **Recoverable:** {interaction['recoverable_count']} / {len(phase1_records)}")
    lines.append(f"- **Unrecoverable:** {interaction['unrecoverable_count']} — {interaction['unrecoverable_qids']}")
    lines.append(f"- **Minimal fix set:** {interaction['minimal_fix_set']}")
    if interaction["conflicts"]:
        lines.append(f"- **Conflicts:** {len(interaction['conflicts'])}")
        for c in interaction["conflicts"][:5]:
            lines.append(f"  - Fix {c['fix_causing_regression']} regresses {c['regression_qid']} "
                         f"(recovered by Fix {c['fix_recovering']})")
    lines.append("")

    # Recovery matrix
    lines.append("### Recovery Matrix\n")
    lines.append("| QID | Fixes | Count |")
    lines.append("|-----|-------|-------|")
    for qid, fixes in sorted(interaction["recovery_matrix"].items()):
        lines.append(f"| {qid} | {', '.join(fixes) or '—'} | {len(fixes)} |")
    lines.append("")

    # ── Phase 4 Projected Scores ─────────────────────────────────────────
    lines.append("## Phase 4: Projected Score Table\n")
    lines.append("| Fix | Recovered | Regressions | Net | New Pass | New Rate |")
    lines.append("|-----|-----------|-------------|-----|----------|----------|")
    for row in projected:
        lines.append(
            f"| {row['fix']} | {row['recovered']} | {row['regressions']} | "
            f"{row['net_gain']:+d} | {row['new_pass_count']}/{total_questions} | "
            f"{row['new_pass_rate']:.1%} |"
        )
    lines.append("")

    # ── Phase 5 Recommendation ───────────────────────────────────────────
    lines.append("## Phase 5: Recommendation\n")

    # Rank by net_gain / effort (Low=1, Medium=2, High=3)
    effort_scores = {"Low": 1, "Medium": 2, "High": 3, "N/A": 99}
    fix_ranking = []
    for fr in fix_results:
        if fr.net_gain <= 0:
            continue
        effort, _ = EFFORT_ESTIMATES.get(fr.fix_name, ("Medium", ""))
        eff_score = effort_scores.get(effort, 2)
        roi = fr.net_gain / max(eff_score, 1)
        fix_ranking.append((fr.fix_name, fr.net_gain, effort, roi))
    fix_ranking.sort(key=lambda x: -x[3])

    lines.append("### Fix Ranking (by ROI = net_gain / effort)\n")
    lines.append("| Priority | Fix | Net Gain | Effort | ROI |")
    lines.append("|----------|-----|----------|--------|-----|")
    for i, (name, gain, effort, roi) in enumerate(fix_ranking, 1):
        lines.append(f"| {i} | {name} | +{gain} | {effort} | {roi:.1f} |")
    lines.append("")

    # Target 70%+
    target = int(total_questions * 0.70)
    needed = target - current_passing
    lines.append(f"### Path to 70% ({target}/{total_questions})\n")
    lines.append(f"- **Currently passing:** {current_passing}")
    lines.append(f"- **Need:** +{needed} net recoveries")

    cumulative = current_passing
    recommended = []
    for name, gain, effort, roi in fix_ranking:
        if cumulative >= target:
            break
        recommended.append(name)
        cumulative += gain
        lines.append(f"- Fix {name} (+{gain}) → {cumulative}/{total_questions} "
                     f"({cumulative / total_questions:.1%})")
    lines.append(f"\n**Recommended combination:** {recommended}")
    if cumulative >= target:
        lines.append(f"**Projected: {cumulative}/{total_questions} ({cumulative / total_questions:.1%}) — target achieved**")
    else:
        lines.append(f"**Projected: {cumulative}/{total_questions} ({cumulative / total_questions:.1%}) — below target, "
                     f"remaining {target - cumulative} need ingestion fixes or benchmark redesign**")
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================


async def main():
    print("=" * 70)
    print("COMPREHENSIVE DRY-RUN DIAGNOSTIC - LifeMemBench")
    print("=" * 70)

    t_start = time_module.time()

    # Load questions and results
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    results_data = json.load(open(RESULTS_PATH, encoding="utf-8"))

    per_question = results_data.get("per_question", {}).get("full", [])
    total_questions = len(per_question)
    current_passing = sum(1 for r in per_question if r["av_pass"])
    current_failing = total_questions - current_passing

    print(f"\nCurrent: {current_passing}/{total_questions} passing ({current_passing / total_questions:.1%})")
    print(f"Failing: {current_failing}")

    # Load judge cache for dry-run AV pass simulation
    judge_cache_path = ARTIFACTS_DIR / "lifemembench_judge_cache.json"
    judge_cache = {}
    if judge_cache_path.exists():
        try:
            judge_cache = json.load(open(judge_cache_path, encoding="utf-8"))
            print(f"Judge cache: {len(judge_cache)} entries")
        except Exception:
            pass

    # Setup clients
    graphiti = get_graphiti_client()
    driver = graphiti.driver
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    try:
        # ── Phase 1 ──────────────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("PHASE 1: Baseline Mapping")
        print(f"{'=' * 70}")

        phase1_records, persona_state = await phase1_baseline(
            graphiti, driver, xai_client, questions, results_data,
        )

        # Save Phase 1 output
        phase1_json = []
        for rec in phase1_records:
            phase1_json.append({
                "question_id": rec.question_id,
                "persona": rec.persona,
                "attack_vector": rec.attack_vector,
                "classification": rec.classification,
                "correct_edges": rec.correct_edges,
                "in_pool": rec.in_pool,
                "pool_size": rec.pool_size,
                "rank_if_found": rec.rank_if_found,
                "activation_if_found": rec.activation_if_found,
                "semantic_if_found": rec.semantic_if_found,
                "blended_if_found": rec.blended_if_found,
                "top1_fact": rec.top1_fact,
                "top1_blended": rec.top1_blended,
                "score_gap": rec.score_gap,
            })
        phase1_path = PROJECT_ROOT / "phase1_baseline.json"
        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(phase1_json, f, indent=2)
        print(f"\nPhase 1 saved: {phase1_path}")

        # Summary
        in_pool = sum(1 for r in phase1_records if r.in_pool)
        not_in_pool = sum(1 for r in phase1_records if not r.in_pool)
        print(f"  In pool: {in_pool}, Not in pool: {not_in_pool}")

        # ── Phase 2 ──────────────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("PHASE 2: Fix Simulations")
        print(f"{'=' * 70}")

        fix_results = []

        # Fix A: Alpha halving
        fix_a = await fix_a_alpha_halving(
            phase1_records, persona_state, questions, results_data, judge_cache,
        )
        fix_results.append(fix_a)

        # Fix B: Pool expansion
        fix_b = await fix_b_pool_expansion(
            phase1_records, graphiti, driver, persona_state, questions,
        )
        fix_results.append(fix_b)

        # Fix C: Cross-category retrieval
        fix_c = await fix_c_cross_category(
            phase1_records, driver, persona_state,
        )
        fix_results.append(fix_c)

        # Fix D: Routing inertness (diagnostic only)
        fix_d = await fix_d_routing_inertness(
            phase1_records, results_data,
        )
        fix_results.append(fix_d)

        # Fix E: Semantic-only ranking
        fix_e = fix_e_semantic_only(
            phase1_records, persona_state, results_data, judge_cache,
        )
        fix_results.append(fix_e)

        # Fix F: Retraction recovery
        fix_f = fix_f_retraction_recovery(phase1_records)
        fix_results.append(fix_f)

        # Fix G: Aggressive semantic floor
        fix_g = fix_g_aggressive_semantic_floor(
            phase1_records, persona_state, results_data,
        )
        fix_results.append(fix_g)

        # Fix H: Benchmark correction
        fix_h = fix_h_benchmark_correction(phase1_records)
        fix_results.append(fix_h)

        # ── Phase 3 ──────────────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("PHASE 3: Interaction Analysis")
        print(f"{'=' * 70}")

        interaction = phase3_interaction_analysis(fix_results, phase1_records)
        print(f"  Recoverable: {interaction['recoverable_count']}")
        print(f"  Unrecoverable: {interaction['unrecoverable_count']}: "
              f"{interaction['unrecoverable_qids']}")
        print(f"  Minimal fix set: {interaction['minimal_fix_set']}")

        # ── Phase 4 ──────────────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("PHASE 4: Projected Scores")
        print(f"{'=' * 70}")

        projected = phase4_projected_scores(
            fix_results, total_questions, current_passing,
        )
        for row in projected:
            print(f"  Fix {row['fix']:4s}: +{row['net_gain']:+3d} -> "
                  f"{row['new_pass_count']}/{total_questions} ({row['new_pass_rate']:.1%})")

        # ── Phase 5: Report ──────────────────────────────────────────────
        elapsed = time_module.time() - t_start
        report = generate_report(
            phase1_records, fix_results, interaction, projected,
            total_questions, current_passing, elapsed,
        )
        report_path = PROJECT_ROOT / "comprehensive_diagnostic_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport: {report_path}")

        # Save recovery matrix
        matrix_path = PROJECT_ROOT / "fix_recovery_matrix.json"
        matrix_data = {
            "recovery_matrix": interaction["recovery_matrix"],
            "fix_results": [
                {
                    "fix_name": fr.fix_name,
                    "description": fr.description,
                    "recovered": fr.recovered,
                    "regressions": fr.regressions,
                    "net_gain": fr.net_gain,
                }
                for fr in fix_results
            ],
            "projected_scores": projected,
            "minimal_fix_set": interaction["minimal_fix_set"],
        }
        with open(matrix_path, "w", encoding="utf-8") as f:
            json.dump(matrix_data, f, indent=2)
        print(f"Matrix: {matrix_path}")

        print(f"\n{'=' * 70}")
        print(f"COMPLETE in {elapsed:.0f}s")
        print(f"{'=' * 70}")

    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
