r"""
evaluate_v4.py - Kill gate that actually tests the thesis.

Root-cause fixes from v2/v3:
----------------------------------------------------------------------
v2 problem: Only 10 Graphiti results per question. Pool too small.
v3 fix:     Fat candidate pool (Graphiti + Cypher keyword + neighborhood).
v3 problem: Activation computed from ENTITY nodes, not EDGES.
            "User caught 7 bass" got IDENTITY activation (from entity "User")
            instead of HOBBIES_INTERESTS activation (from the edge fact itself).
            Result: behavioral WORSE than uniform (MRR 0.030 vs 0.052).

v4 fixes:
    1. EDGE-LEVEL CLASSIFICATION: Classify each RELATES_TO edge's fact text
       directly into the behavioral ontology. "User caught 7 bass" -> HOBBIES.
       One Grok call per edge. This is the architectural fix.

    2. BLENDED RERANKING: final_score = alpha * semantic + (1-alpha) * activation.
       v3 sorted purely by activation, which overruled semantic relevance.
       v2 had blending but on entity-level activations.

    3. FOCUSED EVALUATION: Only score questions where the answer exists in the
       candidate pool ("answerable questions"). Report pool ceiling separately.
       MRR + per-question rank comparison across engines on answerable subset.

Usage:
    Step 1 (once): python evaluate_v4.py --enrich-edges
        Classifies all RELATES_TO edges in full_234 group. ~30min.
    Step 2:        python evaluate_v4.py --evaluate
        Runs the kill gate on enriched edges.
    Combined:      python evaluate_v4.py --enrich-edges --evaluate
----------------------------------------------------------------------
"""

import asyncio
import json
import os
import re
import sys
import time as time_module
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "LongMemEval" / "data"
ARTIFACTS_DIR = DATA_DIR / "full_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

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

for var in ["OPENAI_API_KEY", "XAI_API_KEY"]:
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

sys.path.insert(0, str(PROJECT_ROOT))
from decay_engine import DecayEngine, TemporalContext, FactNode, CATEGORIES
from graphiti_bridge import build_temporal_context
from retrieval_router import route_and_retrieve, RoutingConfig, reset_routing_state


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
# Phase A: Edge-level classification
# ===========================================================================

CLASSIFY_EDGE_PROMPT = """You are a memory classification system. Classify this conversational fact into exactly ONE primary category.

FACT: "{fact}"

CATEGORIES:
- OBLIGATIONS: Promises, commitments, deadlines, duties, recurring responsibilities
- RELATIONAL_BONDS: Family members, friends, colleagues, relationship dynamics, social connections
- HEALTH_WELLBEING: Physical/mental health conditions, medications, allergies, diagnoses
- IDENTITY_SELF_CONCEPT: Name, age, ethnicity, nationality, deeply held beliefs, personality traits
- HOBBIES_RECREATION: Recreational activities, sports, games, collections, leisure pursuits
- PREFERENCES_HABITS: Likes, dislikes, tastes, moral values, aesthetic preferences, opinions
- INTELLECTUAL_INTERESTS: Job, career, education, skills, workplace, academic achievements
- LOGISTICAL_CONTEXT: Schedules, appointments, locations, addresses, routines, travel plans
- PROJECTS_ENDEAVORS: Ongoing projects, goals, deadlines, milestones, creative works
- FINANCIAL_MATERIAL: Income, expenses, budgets, investments, purchases, financial goals
- OTHER: Anything that genuinely doesn't fit above

Also assign membership weights (0.0-1.0) showing how much this fact belongs to each relevant category. Primary category gets highest weight. Most facts belong to 1-2 categories.

Respond ONLY with valid JSON, no markdown:
{{"primary_category": "CATEGORY_NAME", "membership_weights": {{"CAT1": 0.8, "CAT2": 0.2}}, "confidence": 0.9}}"""


async def classify_edge_fact(fact_text: str, xai_client: AsyncOpenAI) -> dict:
    """Classify a single edge fact into the behavioral ontology."""
    try:
        resp = await xai_client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "system", "content": "You classify facts into memory categories. Respond only with valid JSON."},
                {"role": "user", "content": CLASSIFY_EDGE_PROMPT.format(fact=fact_text[:500])},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        return {
            "primary_category": result.get("primary_category", "OTHER"),
            "membership_weights": result.get("membership_weights", {}),
            "confidence": result.get("confidence", 0.5),
        }
    except Exception as e:
        return {"primary_category": "OTHER", "membership_weights": {"OTHER": 1.0}, "confidence": 0.0, "error": str(e)}


async def enrich_edges(driver, group_id: str):
    """Classify all RELATES_TO edges and write fr_ properties directly on edges."""

    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    # Count total and already-enriched edges
    count_result = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid RETURN count(e) AS total",
        gid=group_id,
    )
    total = count_result.records[0]["total"]

    enriched_result = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid AND e.fr_enriched = true RETURN count(e) AS done",
        gid=group_id,
    )
    already_done = enriched_result.records[0]["done"]

    print(f"Total edges: {total}")
    print(f"Already enriched: {already_done}")
    print(f"Need classification: {total - already_done}")

    if already_done >= total:
        print("All edges already enriched. Skipping.")
        return

    # Fetch unenriched edges in batches
    batch_size = 50
    skip = 0
    classified = already_done
    t0 = time_module.time()
    errors = 0

    while True:
        batch_result = await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.group_id = $gid AND (e.fr_enriched IS NULL OR e.fr_enriched = false)
            RETURN e.uuid AS uuid, e.fact AS fact
            LIMIT $limit
            """,
            gid=group_id,
            limit=batch_size,
        )
        records = batch_result.records if hasattr(batch_result, "records") else batch_result
        if not records:
            break

        # Classify in parallel (10 at a time within the batch)
        sem = asyncio.Semaphore(10)

        async def classify_one(rec):
            nonlocal errors
            d = rec.data() if hasattr(rec, "data") else dict(rec)
            uuid = d["uuid"]
            fact = d.get("fact", "") or ""
            async with sem:
                result = await classify_edge_fact(fact, xai_client)

            if "error" in result:
                errors += 1

            # Write to Neo4j
            weights_json = json.dumps(result["membership_weights"])
            try:
                await driver.execute_query(
                    """
                    MATCH ()-[e:RELATES_TO]->()
                    WHERE e.uuid = $uuid
                    SET e.fr_enriched = true,
                        e.fr_primary_category = $cat,
                        e.fr_membership_weights = $weights,
                        e.fr_confidence = $conf,
                        e.fr_classified_ts = $ts
                    """,
                    uuid=uuid,
                    cat=result["primary_category"],
                    weights=weights_json,
                    conf=result["confidence"],
                    ts=time_module.time(),
                )
            except Exception as write_err:
                errors += 1
                print(f"    Write error for {uuid}: {write_err}")

        tasks = [classify_one(rec) for rec in records]
        await asyncio.gather(*tasks, return_exceptions=True)

        classified += len(records)
        elapsed = time_module.time() - t0
        remaining = (elapsed / max(1, classified - already_done)) * (total - classified)
        print(f"  Classified {classified}/{total} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, {errors} errors)")

    elapsed = time_module.time() - t0
    print(f"\nEdge enrichment complete: {classified}/{total} in {elapsed:.0f}s ({errors} errors)")

    # Distribution report
    dist_result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid AND e.fr_enriched = true
        RETURN e.fr_primary_category AS cat, count(*) AS cnt
        ORDER BY cnt DESC
        """,
        gid=group_id,
    )
    print("\nEdge classification distribution:")
    for rec in dist_result.records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        print(f"  {d['cat']:30s}: {d['cnt']}")


# ===========================================================================
# Edge activation cache + computation
# ===========================================================================

# Cache: edge_uuid -> {primary_category, membership_weights, ...}
_edge_cache: dict[str, dict] = {}
_unenriched_fallback_count: int = 0
_answer_oracle: dict[str, set[str]] = {}
ALPHA_SWEEP_VALUES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
POOL_CONCURRENCY = 15


def _edge_attrs_to_fact_node(attrs: dict, edge_uuid: str) -> FactNode:
    """Convert edge fr_ attributes to a FactNode for the decay engine."""
    raw_weights = attrs.get("membership_weights", "{}")
    if isinstance(raw_weights, str):
        weights = json.loads(raw_weights)
    else:
        weights = raw_weights or {}
    for cat in CATEGORIES:
        weights.setdefault(cat, 0.0)

    return FactNode(
        fact_id=edge_uuid,
        membership_weights=weights,
        primary_category=attrs.get("primary_category", "OTHER"),
        last_updated_ts=attrs.get("created_at_ts", 0.0) or 0.0,
        base_activation=1.0,
        future_anchor_ts=None,
        emotional_loading=False,
        emotional_loading_ts=None,
        last_reactivation_ts=None,
        access_count=0,
    )


def _to_unix_ts(value) -> float:
    """Convert Neo4j/Python datetime-ish values to unix seconds."""
    if value is None:
        return 0.0
    if hasattr(value, "to_native"):
        return value.to_native().timestamp()
    if hasattr(value, "timestamp"):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _load_haystack_session_counts(question_ids: set[str]) -> dict[str, int]:
    """Load question_id -> haystack session count from LongMemEval S split."""
    s_path = DATA_DIR / "longmemeval_s_cleaned.json"
    if not s_path.exists():
        return {}

    try:
        s_data = json.load(open(s_path, encoding="utf-8"))
    except Exception:
        return {}

    counts = {}
    for row in s_data:
        qid = row.get("question_id")
        if qid in question_ids:
            counts[qid] = len(row.get("haystack_sessions", []) or [])
    return counts


def _parse_episodic_name(name: str) -> tuple[str, int] | None:
    """
    Parse episodic names produced by ingest_poc.py: q{question_id}_s{session_idx}
    """
    m = re.match(r"^q(.+)_s(\d+)$", name or "")
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _parse_full_episodic_name(name: str) -> int | None:
    """Parse 'full_s{index}' → global session index."""
    m = re.match(r"^full_s(\d+)$", name or "")
    return int(m.group(1)) if m else None


def _load_session_map() -> dict[str, dict] | None:
    """Load session_map.json → {hash: {index, question_ids, num_turns}}."""
    sm_path = ARTIFACTS_DIR / "session_map.json"
    if not sm_path.exists():
        return None
    return json.load(open(sm_path, encoding="utf-8"))


async def build_question_temporal_anchors(
    driver,
    group_id: str,
    questions: list[dict],
    default_now_ts: float,
) -> dict[str, dict]:
    """
    Build per-question temporal anchors using ingested session episodes.

    For each question:
      1) identify the last haystack session index
      2) find the episodic node q{question_id}_s{idx}
      3) set t_now to max(created_at) of edges linked to that episode

    If the last session has no edges, back off to earlier sessions for the same
    question. If no session has edges, fall back to global max edge timestamp.
    """
    question_ids = [q["question_id"] for q in questions]
    qid_set = set(question_ids)
    session_counts = _load_haystack_session_counts(qid_set)
    for q in questions:
        qid = q["question_id"]
        if qid not in session_counts:
            session_counts[qid] = int(q.get("num_sessions", 0) or 0)
        session_counts[qid] = max(1, session_counts[qid])

    episodic_result = await driver.execute_query(
        """
        MATCH (ep:Episodic)
        WHERE ep.group_id = $gid
        RETURN ep.uuid AS uuid, ep.name AS name
        """,
        gid=group_id,
    )
    episodic_records = episodic_result.records if hasattr(episodic_result, "records") else episodic_result

    question_sessions: dict[str, dict[int, str]] = {qid: {} for qid in qid_set}
    full_index_to_uuids: dict[int, list[str]] = {}
    poc_found = False

    for rec in episodic_records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        name = str(d.get("name", ""))
        uuid = d["uuid"]

        # Try PoC format: q{qid}_s{idx}
        parsed = _parse_episodic_name(name)
        if parsed:
            poc_found = True
            qid, idx = parsed
            if qid in qid_set:
                question_sessions.setdefault(qid, {})[idx] = uuid
            continue

        # Try full format: full_s{index}
        full_idx = _parse_full_episodic_name(name)
        if full_idx is not None:
            full_index_to_uuids.setdefault(full_idx, []).append(uuid)

    # Full ingestion: use session_map.json to map global indices → question_ids
    if not poc_found and full_index_to_uuids:
        session_map = _load_session_map()
        if session_map:
            index_to_qids: dict[int, list[str]] = {}
            for entry in session_map.values():
                idx = entry["index"]
                for qid in entry["question_ids"]:
                    if qid in qid_set:
                        index_to_qids.setdefault(idx, []).append(qid)

            for idx, ep_uuids in full_index_to_uuids.items():
                for qid in index_to_qids.get(idx, []):
                    question_sessions.setdefault(qid, {})[idx] = ep_uuids[0]

            mapped = sum(1 for v in question_sessions.values() if v)
            print(f"  Temporal anchoring: mapped {mapped}/{len(qid_set)} questions via session_map.json")

    edges_result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
        UNWIND e.episodes AS ep_uuid
        RETURN ep_uuid AS ep_uuid, count(e) AS edge_count, max(e.created_at) AS max_created_at
        """,
        gid=group_id,
    )
    edge_records = edges_result.records if hasattr(edges_result, "records") else edges_result

    episode_stats: dict[str, tuple[int, float]] = {}
    global_max_ts = default_now_ts
    for rec in edge_records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        ep_uuid = d["ep_uuid"]
        edge_count = int(d.get("edge_count") or 0)
        max_ts = _to_unix_ts(d.get("max_created_at"))
        episode_stats[ep_uuid] = (edge_count, max_ts)
        if max_ts > global_max_ts:
            global_max_ts = max_ts

    anchors = {}
    for q in questions:
        qid = q["question_id"]
        ep_map = question_sessions.get(qid, {})

        selected = None
        if ep_map:
            sorted_indices = sorted(ep_map.keys(), reverse=True)
            target_last_idx = sorted_indices[0]
            for idx in sorted_indices:
                ep_uuid = ep_map[idx]
                edge_count, ts = episode_stats.get(ep_uuid, (0, 0.0))
                if edge_count > 0 and ts > 0:
                    selected = {
                        "current_timestamp": ts,
                        "source": "last_session_edge" if idx == target_last_idx else "fallback_earlier_session_edge",
                        "session_index": idx,
                        "edge_count": edge_count,
                    }
                    break

        if selected is None:
            selected = {
                "current_timestamp": global_max_ts,
                "source": "global_edge_max_fallback",
                "session_index": None,
                "edge_count": 0,
            }

        anchors[qid] = selected

    return anchors


def compute_edge_activation_v4(
    edge_uuid: str,
    ctx: TemporalContext,
    engine: DecayEngine,
) -> float:
    """Compute activation from edge-level classification. The correct way."""
    global _unenriched_fallback_count
    attrs = _edge_cache.get(edge_uuid)
    if attrs is None:
        _unenriched_fallback_count += 1
        return 0.5  # unenriched fallback

    # World-knowledge edges are not personal memories — sink them.
    if attrs.get("is_world_knowledge"):
        return 0.0

    fact_node = _edge_attrs_to_fact_node(attrs, edge_uuid)

    # Compute absolute hours from edge creation time
    if ctx.current_timestamp and fact_node.last_updated_ts > 0:
        abs_hours = (ctx.current_timestamp - fact_node.last_updated_ts) / 3600.0
    else:
        abs_hours = 0.0

    fact_ctx = TemporalContext(
        absolute_hours=max(0.0, abs_hours),
        relative_hours=ctx.relative_hours,
        conversational_messages=ctx.conversational_messages,
        current_timestamp=ctx.current_timestamp,
    )
    return engine.compute_activation(fact_node, fact_ctx)


# ===========================================================================
# Candidate pool (same hybrid approach as v3, but with edge UUIDs tracked)
# ===========================================================================

STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "the", "a", "an", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "about", "that",
    "this", "it", "its", "and", "but", "or", "if", "not", "no", "so", "up", "out",
    "what", "which", "who", "when", "where", "how", "many", "much", "did", "any",
    "some", "all", "most", "more", "also", "very", "just", "than", "then", "now",
    "back", "going", "went", "get", "got", "take", "took", "make", "made",
    "currently", "recently", "previous", "last", "first",
    "suggest", "suggestions", "recommend", "wondering", "planning", "thinking",
    "looking", "trying", "decide", "whether", "know", "tell", "remember",
    "conversation", "chat", "mentioned", "talked", "discussed",
}


def extract_keywords(text: str, min_len: int = 3) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOPWORDS]


@dataclass
class Candidate:
    uuid: str           # edge UUID
    fact: str           # edge fact text
    source: str         # 'graphiti' or 'cypher_kw' etc
    graphiti_score: float  # semantic score from retrieval source


CYPHER_SOURCE_BASELINES = {
    "cypher_kw": 0.4,
    "cypher_intersect": 0.5,
    "cypher_neighbor": 0.3,
    "category_routed": 0.35,
}


async def build_candidate_pool(
    question: str, driver, graphiti, group_id: str,
    xai_client=None, edge_cache: dict = None, routing_config=None,
) -> list[Candidate]:
    """Build fat candidate pool. Same as v3 but tracks graphiti_score."""
    seen_uuids = set()
    candidates = []

    def add(c: Candidate):
        if c.uuid not in seen_uuids:
            seen_uuids.add(c.uuid)
            candidates.append(c)

    keywords = extract_keywords(question)

    # --- Strategy 1: Graphiti semantic search (top 50) ---
    try:
        graphiti_results = await graphiti.search(
            question, group_ids=[group_id], num_results=50,
        )
        for i, r in enumerate(graphiti_results):
            fact_text = ""
            if hasattr(r, "fact"):
                fact_text = str(r.fact)
            elif hasattr(r, "name"):
                fact_text = str(r.name)

            uuid = str(getattr(r, "uuid", None) or id(r))

            # Graphiti returns results in ranked order; approximate score as 1 - rank/total
            # This preserves relative ordering from Graphiti's hybrid retrieval
            graphiti_score = 1.0 - (i / max(len(graphiti_results), 1))

            add(Candidate(uuid, fact_text, "graphiti", graphiti_score))
    except Exception as e:
        print(f"    Graphiti search error: {e}")

    # --- Strategy 2: Cypher keyword search on edge facts ---
    for kw in keywords[:8]:
        try:
            result = await driver.execute_query(
                """
                MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
                WHERE e.group_id = $group_id
                  AND toLower(e.fact) CONTAINS $keyword
                RETURN e.uuid AS uuid, e.fact AS fact
                LIMIT 20
                """,
                group_id=group_id,
                keyword=kw,
            )
            records = result.records if hasattr(result, "records") else result
            for rec in records:
                d = rec.data() if hasattr(rec, "data") else dict(rec)
                add(Candidate(d["uuid"], d["fact"], "cypher_kw", CYPHER_SOURCE_BASELINES["cypher_kw"]))
        except Exception:
            pass

    # --- Strategy 3: Multi-keyword intersection ---
    if len(keywords) >= 2:
        for i in range(min(3, len(keywords))):
            for j in range(i + 1, min(5, len(keywords))):
                kw1, kw2 = keywords[i], keywords[j]
                try:
                    result = await driver.execute_query(
                        """
                        MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
                        WHERE e.group_id = $group_id
                          AND toLower(e.fact) CONTAINS $kw1
                          AND toLower(e.fact) CONTAINS $kw2
                        RETURN e.uuid AS uuid, e.fact AS fact
                        LIMIT 10
                        """,
                        group_id=group_id,
                        kw1=kw1,
                        kw2=kw2,
                    )
                    records = result.records if hasattr(result, "records") else result
                    for rec in records:
                        d = rec.data() if hasattr(rec, "data") else dict(rec)
                        add(Candidate(d["uuid"], d["fact"], "cypher_intersect", CYPHER_SOURCE_BASELINES["cypher_intersect"]))
                except Exception:
                    pass

    # --- Strategy 4: Entity name match -> neighborhood edges ---
    for kw in keywords[:5]:
        try:
            result = await driver.execute_query(
                """
                MATCH (n:Entity)
                WHERE n.group_id = $group_id
                  AND toLower(n.name) CONTAINS $keyword
                WITH n LIMIT 5
                MATCH (n)-[e:RELATES_TO]-(other:Entity)
                WHERE e.group_id = $group_id
                RETURN e.uuid AS uuid, e.fact AS fact
                LIMIT 30
                """,
                group_id=group_id,
                keyword=kw,
            )
            records = result.records if hasattr(result, "records") else result
            for rec in records:
                d = rec.data() if hasattr(rec, "data") else dict(rec)
                add(Candidate(d["uuid"], d["fact"], "cypher_neighbor", CYPHER_SOURCE_BASELINES["cypher_neighbor"]))
        except Exception:
            pass

    # --- Strategy 5: Category-aware routing ---
    if xai_client is not None and edge_cache is not None:
        effective_config = routing_config or RoutingConfig()
        try:
            routed = await route_and_retrieve(
                query=question,
                driver=driver,
                group_id=group_id,
                xai_client=xai_client,
                edge_cache=edge_cache,
                config=effective_config,
            )
            for uuid, fact, source, score in routed:
                add(Candidate(uuid, fact, source, score))
        except Exception:
            pass

    return candidates


# ===========================================================================
# Reranking: BLENDED (the v2 idea + v3 pool + v4 edge activation)
# ===========================================================================

def rerank_candidates(
    candidates: list[Candidate],
    ctx: TemporalContext,
    engine: DecayEngine,
    alpha: float = 0.5,
) -> list[tuple[Candidate, float, float, float]]:
    """Rerank by blended score = alpha * activation + (1-alpha) * semantic.

    Returns: list of (candidate, activation, semantic_score, blended_score)
    sorted by blended_score descending.
    """
    scored = []
    for c in candidates:
        activation = compute_edge_activation_v4(c.uuid, ctx, engine)
        blended = alpha * activation + (1.0 - alpha) * c.graphiti_score
        scored.append((c, activation, c.graphiti_score, blended))

    scored.sort(key=lambda x: (-x[3], -x[2]))
    return scored


# ===========================================================================
# Answer matching (improved from v3)
# ===========================================================================

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[,$%]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _answer_match(expected: str, fact_text: str) -> bool:
    expected_norm = _normalize(expected)
    fact_norm = _normalize(fact_text)

    # Direct substring
    if expected_norm in fact_norm:
        return True

    expected_words = expected_norm.split()

    # Single-word answers: exact number or whole-word match
    if len(expected_words) == 1:
        word = expected_words[0]
        if word.replace(".", "").isdigit():
            fact_numbers = re.findall(r"\d+(?:\.\d+)?", fact_norm)
            if word in fact_numbers:
                return True
        if re.search(r"\b" + re.escape(word) + r"\b", fact_norm):
            return True

    # Multi-word: token overlap >= 50%
    tokens = [w for w in expected_words if len(w) > 2]
    if tokens:
        matches = sum(1 for w in tokens if w in fact_norm)
        if matches / len(tokens) >= 0.5:
            return True

    return False


def _is_answer(question_id: str, edge_uuid: str, fact_text: str, expected_answer: str) -> bool:
    """Check if edge is an answer edge. Uses LLM oracle if available, else substring."""
    if _answer_oracle:
        return edge_uuid in _answer_oracle.get(question_id, set())
    return _answer_match(expected_answer, fact_text)


async def _build_one_pool(
    qi: int,
    q: dict,
    graphiti,
    driver,
    group_id: str,
    question_anchors: dict,
    default_now_ts: float,
    sem: asyncio.Semaphore,
    total: int,
    xai_client=None,
    edge_cache: dict = None,
    routing_config=None,
) -> dict:
    """Build candidate pool for one question. Safe for concurrent execution."""
    async with sem:
        question_id = q["question_id"]
        question_text = q["question"]
        expected_answer = str(q["answer"])

        anchor = question_anchors.get(question_id, {})
        q_now_ts = anchor.get("current_timestamp", default_now_ts)
        ctx = build_temporal_context(
            last_session_ts=q_now_ts - 86400.0,
            session_message_count=0,
            now=q_now_ts,
        )

        pool = await build_candidate_pool(
            question_text, driver, graphiti, group_id,
            xai_client=xai_client, edge_cache=edge_cache,
            routing_config=routing_config,
        )
        answer_in_pool = sum(1 for c in pool if _is_answer(question_id, c.uuid, c.fact, expected_answer))
        sources = Counter(c.source for c in pool)

        print(f"  [{qi+1}/{total}] pool={len(pool)} ans={answer_in_pool} | {question_text[:55]}")

        return {
            "qi": qi,
            "q": q,
            "question_id": question_id,
            "question_text": question_text,
            "expected_answer": expected_answer,
            "pool": pool,
            "ctx": ctx,
            "anchor": anchor,
            "q_now_ts": q_now_ts,
            "answer_in_pool": answer_in_pool,
            "sources": sources,
            "is_answerable": answer_in_pool > 0,
        }


# ===========================================================================
# Main evaluation
# ===========================================================================

async def run_evaluate(
    graphiti,
    group_id: str,
    alpha: float = 0.5,
    results_suffix: str | None = None,
    prebuilt_pools: list | None = None,
) -> tuple[dict, list]:
    """Run the kill gate evaluation using edge-level activations.

    Returns (summary_dict, question_pools) — pools can be reused for sweep.
    """
    global _edge_cache, _unenriched_fallback_count, _answer_oracle

    _unenriched_fallback_count = 0

    engines = {
        "behavioral": DecayEngine.default(),
        "uniform": DecayEngine.uniform(),
        "cognitive": DecayEngine.cognitive(),
    }
    engine_names = list(engines.keys()) + ["no_rerank"]

    # ---- PHASE 1: Setup + Pool Building (skip if prebuilt) ----
    if prebuilt_pools is None:
        _edge_cache.clear()

        q_path = ARTIFACTS_DIR / "full_questions.json"
        if not q_path.exists():
            print("ERROR: full_questions.json not found.")
            sys.exit(1)

        questions = json.load(open(q_path, encoding="utf-8"))
        print(f"Evaluating {len(questions)} questions (alpha={alpha})")

        # --- Load LLM-verified answer oracle ---
        oracle_path = ARTIFACTS_DIR / "answer_oracle.json"
        _answer_oracle.clear()
        if oracle_path.exists():
            raw_oracle = json.load(open(oracle_path, encoding="utf-8"))
            _answer_oracle.update({qid: set(uuids) for qid, uuids in raw_oracle.items()})
            print(f"Loaded LLM answer oracle: {len(_answer_oracle)} questions with verified answer edges")
        else:
            print("WARNING: answer_oracle.json not found — falling back to substring matching")
            print("  Run llm_answer_judge.py first for trustworthy results!")

        # Pre-load EDGE cache
        print("Pre-loading edge classification cache...")
        cache_result = await graphiti.driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.group_id = $gid AND e.fr_enriched = true
            RETURN e.uuid AS uuid,
                   e.fr_primary_category AS primary_category,
                   e.fr_membership_weights AS membership_weights,
                   e.fr_confidence AS confidence,
                   e.created_at AS created_at,
                   e.fr_is_world_knowledge AS is_world_knowledge
            """,
            gid=group_id,
        )
        records = cache_result.records if hasattr(cache_result, "records") else cache_result
        for rec in records:
            d = rec.data() if hasattr(rec, "data") else dict(rec)
            uuid = d.pop("uuid")
            d["created_at_ts"] = _to_unix_ts(d.get("created_at"))
            _edge_cache[uuid] = d

        enriched_count = len(_edge_cache)
        wk_count = sum(1 for v in _edge_cache.values() if v.get("is_world_knowledge"))
        personal_count = enriched_count - wk_count
        print(f"Cached {enriched_count} enriched edges ({personal_count} personal, {wk_count} world-knowledge)")

        # Initialize category-aware routing
        reset_routing_state()
        enable_routing = "--no-routing" not in sys.argv
        routing_config = RoutingConfig(enable_routing=enable_routing)
        xai_client_routing = AsyncOpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
        )
        print(f"Category routing: {'ENABLED' if enable_routing else 'DISABLED (--no-routing)'}")

        ts_values = [v.get("created_at_ts", 0.0) for v in _edge_cache.values()]
        nonzero_ts = sum(1 for t in ts_values if t > 0)
        default_now_ts = max(ts_values) if ts_values else time_module.time()
        if ts_values:
            ts_min = min(t for t in ts_values if t > 0) if nonzero_ts else 0
            ts_max = max(ts_values)
            spread_h = (ts_max - ts_min) / 3600 if ts_min > 0 else 0
            print(f"Timestamps: {nonzero_ts}/{enriched_count} non-zero, spread={spread_h:.0f}h ({spread_h/24:.0f}d)")

        if enriched_count == 0:
            print("ERROR: No enriched edges found. Run --enrich-edges first.")
            sys.exit(1)

        # Build per-question temporal anchors
        question_anchors = await build_question_temporal_anchors(
            graphiti.driver, group_id, questions, default_now_ts,
        )
        anchor_modes = Counter(v.get("source", "?") for v in question_anchors.values())
        print(f"Question temporal anchors: {dict(anchor_modes)}")

        # Diagnostic: verify activations differ across engines
        sample_uuid = next(iter(_edge_cache))
        sample_attrs = _edge_cache[sample_uuid]
        print(f"\nSample edge: category={sample_attrs.get('primary_category')}")
        sample_fact = _edge_attrs_to_fact_node(sample_attrs, sample_uuid)
        sample_ctx = TemporalContext(
            absolute_hours=24.0, relative_hours=24.0,
            conversational_messages=0, current_timestamp=default_now_ts,
        )
        for ename, eng in engines.items():
            a = eng.compute_activation(sample_fact, sample_ctx)
            print(f"  {ename}: {a:.4f}")

        # ---- Build all pools in parallel ----
        print(f"\nBuilding candidate pools ({POOL_CONCURRENCY} concurrent)...")
        t0 = time_module.time()
        sem = asyncio.Semaphore(POOL_CONCURRENCY)
        pool_tasks = [
            _build_one_pool(qi, q, graphiti, graphiti.driver, group_id,
                            question_anchors, default_now_ts, sem, len(questions),
                            xai_client=xai_client_routing,
                            edge_cache=_edge_cache,
                            routing_config=routing_config)
            for qi, q in enumerate(questions)
        ]
        pool_results = await asyncio.gather(*pool_tasks, return_exceptions=True)

        # Collect results, handle errors
        question_pools: list[dict] = []
        errors = 0
        for i, r in enumerate(pool_results):
            if isinstance(r, Exception):
                print(f"  ERROR Q{i+1}: {r}")
                errors += 1
                question_pools.append({
                    "qi": i, "q": questions[i],
                    "question_id": questions[i]["question_id"],
                    "question_text": questions[i]["question"],
                    "expected_answer": str(questions[i]["answer"]),
                    "pool": [], "ctx": None, "anchor": {},
                    "q_now_ts": default_now_ts,
                    "answer_in_pool": 0, "sources": Counter(),
                    "is_answerable": False,
                })
            else:
                question_pools.append(r)

        elapsed = time_module.time() - t0
        avg_pool = sum(len(qp["pool"]) for qp in question_pools) / max(len(question_pools), 1)
        print(f"All pools built in {elapsed:.1f}s (avg {avg_pool:.0f} candidates, {errors} errors)")
    else:
        # ---- Reuse prebuilt pools (sweep mode) ----
        question_pools = prebuilt_pools
        questions = [qp["q"] for qp in question_pools]
        print(f"\nReusing prebuilt pools for alpha={alpha} ({len(questions)} questions)")

    # ---- PHASE 2: Rerank + Score (fast, sequential) ----
    results = {name: [] for name in engine_names}
    pool_sizes = []
    answerable_count = 0

    print(f"\n{'='*70}")
    print(f"KILL GATE v4: Edge-Level Decay + Blended Reranking (alpha={alpha})")
    print(f"{'='*70}\n")

    for qp in question_pools:
        qi = qp["qi"]
        question_id = qp["question_id"]
        question_text = qp["question_text"]
        expected_answer = qp["expected_answer"]
        pool = qp["pool"]
        ctx = qp["ctx"]
        anchor = qp["anchor"]
        q_now_ts = qp["q_now_ts"]
        answer_in_pool = qp["answer_in_pool"]
        sources = qp["sources"]
        is_answerable = qp["is_answerable"]

        pool_sizes.append(len(pool))
        if is_answerable:
            answerable_count += 1

        anchor_dt = datetime.fromtimestamp(q_now_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"[{qi+1}/{len(questions)}] {question_text[:65]}...")
        print(f"  Answer: {str(expected_answer)[:50]}")
        print(
            f"  t_now: {anchor_dt} UTC "
            f"(source={anchor.get('source', 'unknown')}, session={anchor.get('session_index')}, edges={anchor.get('edge_count', 0)})"
        )
        print(f"  Pool: {len(pool)} candidates ({dict(sources)})")
        print(f"  Answer in pool: {answer_in_pool} facts {'[ANSWERABLE]' if is_answerable else '[CEILING]'}")

        if not pool:
            for name in engine_names:
                results[name].append({
                    "question_id": question_id,
                    "question_type": qp["q"].get("question_type", "unknown"),
                    "hit": False,
                    "answerable": False,
                    "reason": "empty_pool",
                })
            print(f"  EMPTY POOL - skipping")
            continue

        # --- No-rerank baseline: Graphiti's original top 5 ---
        graphiti_top5 = [c for c in pool if c.source == "graphiti"][:5]
        nr_hit = any(_is_answer(question_id, c.uuid, c.fact, expected_answer) for c in graphiti_top5)
        results["no_rerank"].append({
            "question_id": question_id,
            "question_type": qp["q"].get("question_type", "unknown"),
            "hit": nr_hit,
            "answerable": is_answerable,
            "answer_in_pool": answer_in_pool,
            "pool_size": len(pool),
        })

        # --- Rerank with each engine ---
        for engine_name, engine in engines.items():
            reranked = rerank_candidates(pool, ctx, engine, alpha=alpha)

            hit = False
            top_facts = []
            for cand, activation, sem_score, blended in reranked[:5]:
                is_match = _is_answer(question_id, cand.uuid, cand.fact, expected_answer)
                if is_match:
                    hit = True
                top_facts.append({
                    "text": cand.fact[:120],
                    "activation": round(activation, 4),
                    "semantic": round(sem_score, 4),
                    "blended": round(blended, 4),
                    "source": cand.source,
                    "category": _edge_cache.get(cand.uuid, {}).get("primary_category", "?"),
                    "match": is_match,
                })

            # Find rank of first correct answer
            answer_rank = None
            answer_activation = None
            answer_category = None
            for rank, (cand, activation, sem, blended) in enumerate(reranked):
                if _is_answer(question_id, cand.uuid, cand.fact, expected_answer):
                    answer_rank = rank + 1
                    answer_activation = activation
                    answer_category = _edge_cache.get(cand.uuid, {}).get("primary_category", "?")
                    break

            results[engine_name].append({
                "question_id": question_id,
                "question_type": qp["q"].get("question_type", "unknown"),
                "hit": hit,
                "answerable": is_answerable,
                "answer_rank": answer_rank,
                "answer_activation": round(answer_activation, 4) if answer_activation is not None else None,
                "answer_category": answer_category,
                "answer_in_pool": answer_in_pool,
                "pool_size": len(pool),
                "top_facts": top_facts,
            })

        # Per-question summary line
        marks = {name: ("*" if results[name][-1]["hit"] else " ") for name in engine_names}
        ranks_str = []
        for name in list(engines.keys()):
            r = results[name][-1].get("answer_rank")
            if r:
                ranks_str.append(f"{name[0].upper()}=@{r}")
            else:
                ranks_str.append(f"{name[0].upper()}=miss")

        print(f"  Hits: B={marks['behavioral']} U={marks['uniform']} C={marks['cognitive']} NR={marks['no_rerank']}  |  {' '.join(ranks_str)}")

        # For answerable questions, show WHY the ranks differ
        if is_answerable:
            for ename in engines:
                r = results[ename][-1]
                act_str = f"act={r['answer_activation']}" if r["answer_activation"] is not None else ""
                cat_str = f"cat={r['answer_category']}" if r["answer_category"] else ""
                rank_str = f"rank={r['answer_rank']}" if r["answer_rank"] else "rank=miss"
                print(f"    {ename:12s}: {rank_str} {act_str} {cat_str}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"KILL GATE v4 RESULTS (alpha={alpha})")
    print(f"{'='*70}\n")

    print(f"Pool stats: avg={sum(pool_sizes)/len(pool_sizes):.0f}, "
          f"min={min(pool_sizes)}, max={max(pool_sizes)}")
    print(f"Answerable questions (answer in pool): {answerable_count}/{len(questions)}")

    # --- Hit@5: ALL questions ---
    print(f"\nhit@5 (all {len(questions)} questions):")
    for name in engine_names:
        hits = sum(1 for r in results[name] if r.get("hit"))
        print(f"  {name:12s}: {hits}/{len(questions)} ({100*hits/len(questions):.0f}%)")

    # --- Hit@5: ANSWERABLE only ---
    print(f"\nhit@5 (answerable only, n={answerable_count}):")
    for name in engine_names:
        answerable_results = [r for r in results[name] if r.get("answerable")]
        hits = sum(1 for r in answerable_results if r.get("hit"))
        total = len(answerable_results)
        pct = 100 * hits / total if total > 0 else 0
        print(f"  {name:12s}: {hits}/{total} ({pct:.0f}%)")

    # --- MRR: answerable only ---
    print(f"\nMean Reciprocal Rank (answerable questions):")
    mrr_values = {}
    for name in list(engines.keys()):
        ranks = [r["answer_rank"] for r in results[name] if r.get("answer_rank")]
        if ranks:
            mrr = sum(1.0 / r for r in ranks) / len(ranks)
            mean_rank = sum(ranks) / len(ranks)
            mrr_values[name] = mrr
            print(f"  {name:12s}: MRR={mrr:.4f}  mean_rank={mean_rank:.1f}  (n={len(ranks)})")
        else:
            mrr_values[name] = 0.0
            print(f"  {name:12s}: no answerable hits")

    # --- Per-question rank comparison (the real test) ---
    print(f"\nPer-question rank comparison (answerable only):")
    print(f"  {'Question':<45s} {'B-rank':>7s} {'U-rank':>7s} {'C-rank':>7s} {'Winner':>10s}")
    print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")

    b_wins = 0
    u_wins = 0
    c_wins = 0
    ties = 0

    for qi, q in enumerate(questions):
        b_r = results["behavioral"][qi].get("answer_rank")
        u_r = results["uniform"][qi].get("answer_rank")
        c_r = results["cognitive"][qi].get("answer_rank")

        if not results["behavioral"][qi].get("answerable"):
            continue

        b_str = f"@{b_r}" if b_r else "miss"
        u_str = f"@{u_r}" if u_r else "miss"
        c_str = f"@{c_r}" if c_r else "miss"

        # Determine winner (lower rank = better, miss = infinity)
        def rank_val(r):
            return r if r else 9999

        vals = {"B": rank_val(b_r), "U": rank_val(u_r), "C": rank_val(c_r)}
        best = min(vals.values())
        winners = [k for k, v in vals.items() if v == best]

        if len(winners) == 3:
            winner = "TIE"
            ties += 1
        elif "B" in winners and len(winners) == 1:
            winner = "BEHAVIORAL"
            b_wins += 1
        elif "U" in winners and len(winners) == 1:
            winner = "UNIFORM"
            u_wins += 1
        elif "C" in winners and len(winners) == 1:
            winner = "COGNITIVE"
            c_wins += 1
        else:
            winner = "+".join(winners)
            ties += 1

        qtext = q["question"][:43]
        print(f"  {qtext:<45s} {b_str:>7s} {u_str:>7s} {c_str:>7s} {winner:>10s}")

    print(f"\nWins: Behavioral={b_wins}  Uniform={u_wins}  Cognitive={c_wins}  Ties={ties}")

    # --- Activation diversity check ---
    all_acts = []
    for r in results["behavioral"]:
        for f in r.get("top_facts", []):
            all_acts.append(f.get("activation", 0.5))
    unique = len(set(round(a, 3) for a in all_acts))
    still_05 = sum(1 for a in all_acts if abs(a - 0.5) < 0.001)
    print(f"\nActivation diversity: {unique} unique values in {len(all_acts)} scores")
    print(f"Still-0.5 count: {still_05}/{len(all_acts)} ({'PROBLEM' if still_05 > len(all_acts)//2 else 'OK'})")
    print(f"Unenriched fallback count: {_unenriched_fallback_count} ({'PROBLEM' if _unenriched_fallback_count > 0 else 'OK'})")

    # --- Category distribution in top-5 per engine ---
    print(f"\nCategory distribution in top-5 results:")
    for ename in engines:
        cats = Counter()
        for r in results[ename]:
            for f in r.get("top_facts", []):
                cats[f.get("category", "?")] += 1
        top_cats = cats.most_common(5)
        cat_str = ", ".join(f"{c}:{n}" for c, n in top_cats)
        print(f"  {ename:12s}: {cat_str}")

    # --- THE VERDICT ---
    b_hits_ans = sum(1 for r in results["behavioral"] if r.get("answerable") and r.get("hit"))
    u_hits_ans = sum(1 for r in results["uniform"] if r.get("answerable") and r.get("hit"))
    c_hits_ans = sum(1 for r in results["cognitive"] if r.get("answerable") and r.get("hit"))
    b_mrr = mrr_values.get("behavioral", 0)
    u_mrr = mrr_values.get("uniform", 0)
    c_mrr = mrr_values.get("cognitive", 0)

    print(f"\n{'='*70}")
    print(f"  VERDICT (answerable questions only, n={answerable_count})")
    print(f"  hit@5:  B={b_hits_ans}  U={u_hits_ans}  C={c_hits_ans}")
    print(f"  MRR:    B={b_mrr:.4f}  U={u_mrr:.4f}  C={c_mrr:.4f}")
    print(f"  Wins:   B={b_wins}  U={u_wins}  C={c_wins}  Ties={ties}")

    if b_mrr > c_mrr and b_mrr > u_mrr:
        print(f"\n>>> BEHAVIORAL WINS on MRR. Thesis signal detected.")
        if b_wins > c_wins and b_wins > u_wins:
            print(f"  >>> BEHAVIORAL also wins most questions. Strong signal.")
        print(f"  >>> Proceed to full evaluation with confidence.")
    elif b_mrr > c_mrr:
        print(f"\n>>> BEHAVIORAL > COGNITIVE on MRR. Partial signal.")
        print(f"  >>> Uniform competitive. Investigate blend_weight or decay rates.")
    elif b_mrr == c_mrr == u_mrr == 0:
        print(f"\n>>> All MRR=0. Pool ceiling too low or answer matcher broken.")
        print(f"  >>> Not a thesis failure - evaluation methodology issue.")
    else:
        print(f"\n>>> BEHAVIORAL did not win. Investigate:")
        print(f"  >>> Check per-question rank comparison above for patterns.")
        print(f"  >>> Are specific categories hurting? Check category distribution.")
    print(f"{'='*70}")

    # Save
    suffix = f"_{results_suffix}" if results_suffix else ""
    out_path = ARTIFACTS_DIR / f"kill_gate_results_v4{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return {
        "alpha": alpha,
        "answerable_count": answerable_count,
        "mrr": {
            "behavioral": b_mrr,
            "uniform": u_mrr,
            "cognitive": c_mrr,
        },
        "hit_at_5_answerable": {
            "behavioral": b_hits_ans,
            "uniform": u_hits_ans,
            "cognitive": c_hits_ans,
        },
        "wins": {
            "behavioral": b_wins,
            "uniform": u_wins,
            "cognitive": c_wins,
            "ties": ties,
        },
        "results_path": str(out_path),
    }, question_pools


# ===========================================================================
# CLI
# ===========================================================================

def print_alpha_sweep_summary(summaries: list[dict]):
    """Print compact MRR comparison across alpha values."""
    if not summaries:
        return

    print(f"\n{'='*70}")
    print("ALPHA SWEEP SUMMARY (MRR on answerable questions)")
    print(f"{'='*70}")
    print(f"{'alpha':>7s} {'B-MRR':>10s} {'U-MRR':>10s} {'C-MRR':>10s} {'winner':>12s}")
    print(f"{'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for s in summaries:
        alpha = s.get("alpha", 0.0)
        mrr = s.get("mrr", {})
        b = float(mrr.get("behavioral", 0.0))
        u = float(mrr.get("uniform", 0.0))
        c = float(mrr.get("cognitive", 0.0))
        best = max({"behavioral": b, "uniform": u, "cognitive": c}.items(), key=lambda kv: kv[1])[0]
        print(f"{alpha:7.3f} {b:10.4f} {u:10.4f} {c:10.4f} {best:>12s}")

    print(f"{'='*70}")


async def main():
    args = sys.argv[1:]

    if not args or "--help" in args:
        print("Usage:")
        print("  python evaluate_v4.py --enrich-edges        # Classify edges (~30min)")
        print("  python evaluate_v4.py --evaluate             # Run kill gate")
        print("  python evaluate_v4.py --evaluate --alpha 0.3 # Custom blend weight")
        print("  python evaluate_v4.py --evaluate --sweep     # Run alpha sweep (0.2..0.8)")
        print("  python evaluate_v4.py --enrich-edges --evaluate  # Both")
        print("  python evaluate_v4.py --evaluate --no-routing  # Disable category routing (A/B baseline)")
        sys.exit(0)

    graphiti = get_graphiti_client()
    group_id = "full_234"
    did_work = False

    try:
        if "--enrich-edges" in args:
            did_work = True
            print("=" * 70)
            print("Phase A: Enriching EDGES with behavioral ontology classifications")
            print("=" * 70)
            await enrich_edges(graphiti.driver, group_id)

        do_sweep = "--sweep" in args
        do_evaluate = "--evaluate" in args or do_sweep
        if do_evaluate:
            did_work = True
            print("\n" + "=" * 70)
            print("Phase B: Kill Gate Evaluation")
            print("=" * 70)

            if do_sweep:
                summaries = []
                pools = None
                for alpha in ALPHA_SWEEP_VALUES:
                    suffix = f"alpha_{alpha:.3f}".replace(".", "_")
                    summary, pools = await run_evaluate(
                        graphiti, group_id, alpha=alpha,
                        results_suffix=suffix, prebuilt_pools=pools,
                    )
                    summaries.append(summary)
                print_alpha_sweep_summary(summaries)
            else:
                alpha = 0.5
                if "--alpha" in args:
                    idx = args.index("--alpha")
                    if idx + 1 < len(args):
                        alpha = float(args[idx + 1])
                await run_evaluate(graphiti, group_id, alpha=alpha)

        if not did_work:
            print("No action specified. Use --enrich-edges, --evaluate, or both.")
    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())