"""
evaluate_v3.py -- Hybrid kill gate that actually tests the decay engine thesis.

Problem with v1/v2: Graphiti's semantic search returns 5-10 results, most irrelevant.
The decay engine can only rerank what it receives. When the candidate pool is garbage,
reranking is meaningless.

Fix: Build a FAT candidate pool per question using multiple retrieval strategies:
    1. Graphiti semantic search (top 50)
    2. Direct Cypher keyword search on edge facts (question keywords)
    3. Direct Cypher keyword search on entity names/summaries
    4. Graph neighborhood traversal from matched entities

Then let all three engines rerank this merged pool. NOW we're testing whether
behavioral decay puts the right facts higher than uniform/cognitive.

Usage:
    python evaluate_v3.py
"""

import asyncio
import json
import os
import re
import sys
import time as time_module
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'LongMemEval' / 'data'
POC_ARTIFACTS_DIR = DATA_DIR / 'poc_artifacts'
POC_ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env():
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

load_env()

for var in ['OPENAI_API_KEY', 'XAI_API_KEY']:
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


def get_graphiti_client() -> Graphiti:
    xai_client = AsyncOpenAI(
        api_key=os.environ['XAI_API_KEY'],
        base_url="https://api.x.ai/v1",
    )
    llm_client = OpenAIClient(
        client=xai_client,
        config=LLMConfig(
            model="grok-4-1-fast-non-reasoning",
            small_model="grok-4-1-fast-non-reasoning",
        ),
    )
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig())
    return Graphiti(
        os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
        os.environ.get('NEO4J_USER', 'neo4j'),
        os.environ.get('NEO4J_PASSWORD', 'testpassword123'),
        llm_client=llm_client,
        embedder=embedder,
    )


# ===========================================================================
# Entity cache (same as v2)
# ===========================================================================

_entity_cache: dict[str, dict | None] = {}


def _attrs_to_fact_node(attrs: dict, fact_id: str) -> FactNode:
    raw_weights = attrs.get('membership_weights', '{}')
    if isinstance(raw_weights, str):
        weights = json.loads(raw_weights)
    else:
        weights = raw_weights or {}
    for cat in CATEGORIES:
        weights.setdefault(cat, 0.0)
    return FactNode(
        fact_id=fact_id,
        membership_weights=weights,
        primary_category=attrs.get('primary_category', 'OTHER'),
        last_updated_ts=attrs.get('last_updated_ts', 0.0) or 0.0,
        base_activation=1.0,
        future_anchor_ts=attrs.get('future_anchor_ts'),
        emotional_loading=attrs.get('emotional_loading', False) or False,
        emotional_loading_ts=attrs.get('emotional_loading_ts'),
        last_reactivation_ts=attrs.get('last_reactivation_ts'),
        access_count=attrs.get('access_count', 0) or 0,
    )


def compute_activation_from_uuids(
    source_uuid: str | None, target_uuid: str | None,
    ctx: TemporalContext, engine: DecayEngine
) -> float:
    """Compute decay activation from source/target entity UUIDs using cache."""
    activations = []
    for uuid in (source_uuid, target_uuid):
        if not uuid:
            continue
        attrs = _entity_cache.get(uuid)
        if attrs is None:
            continue
        fact_node = _attrs_to_fact_node(attrs, uuid)
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
        activations.append(engine.compute_activation(fact_node, fact_ctx))

    return sum(activations) / len(activations) if activations else 0.5


# ===========================================================================
# Candidate pool builder
# ===========================================================================

# Stopwords to filter out from keyword extraction
STOPWORDS = {
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'the', 'a', 'an', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'about', 'that',
    'this', 'it', 'its', 'and', 'but', 'or', 'if', 'not', 'no', 'so', 'up', 'out',
    'what', 'which', 'who', 'when', 'where', 'how', 'many', 'much', 'did', 'any',
    'some', 'all', 'most', 'more', 'also', 'very', 'just', 'than', 'then', 'now',
    'back', 'going', 'went', 'get', 'got', 'take', 'took', 'make', 'made',
    'currently', 'recently', 'previous', 'previous', 'last', 'first',
    'suggest', 'suggestions', 'recommend', 'wondering', 'planning', 'thinking',
    'looking', 'trying', 'decide', 'whether', 'know', 'tell', 'remember',
    'conversation', 'chat', 'mentioned', 'talked', 'discussed',
}


def extract_keywords(text: str, min_len: int = 3) -> list[str]:
    """Extract meaningful keywords from a question."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOPWORDS]


# A candidate edge: fact text + source/target UUIDs for activation lookup
class Candidate:
    __slots__ = ('uuid', 'fact', 'source_uuid', 'target_uuid', 'source')

    def __init__(self, uuid, fact, source_uuid=None, target_uuid=None, source='cypher'):
        self.uuid = uuid
        self.fact = fact
        self.source_uuid = source_uuid
        self.target_uuid = target_uuid
        self.source = source  # 'cypher' or 'graphiti'


async def build_candidate_pool(
    question: str, driver, graphiti, group_id: str
) -> list[Candidate]:
    """Build a fat candidate pool using multiple retrieval strategies."""

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
        for r in graphiti_results:
            fact_text = ''
            if hasattr(r, 'fact'):
                fact_text = str(r.fact)
            elif hasattr(r, 'name'):
                fact_text = str(r.name)

            uuid = getattr(r, 'uuid', None) or id(r)
            src = getattr(r, 'source_node_uuid', None)
            tgt = getattr(r, 'target_node_uuid', None)
            add(Candidate(str(uuid), fact_text, src, tgt, 'graphiti'))
    except Exception as e:
        print(f"    Graphiti search error: {e}")

    # --- Strategy 2: Direct Cypher keyword search on edge facts ---
    # Search for each keyword individually, then intersections
    for kw in keywords[:8]:  # limit to avoid huge queries
        try:
            result = await driver.execute_query(
                """
                MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
                WHERE e.group_id = $group_id
                  AND toLower(e.fact) CONTAINS $keyword
                RETURN e.uuid AS uuid, e.fact AS fact,
                       s.uuid AS source_uuid, t.uuid AS target_uuid
                LIMIT 20
                """,
                group_id=group_id,
                keyword=kw,
            )
            records = result.records if hasattr(result, 'records') else result
            for rec in records:
                d = rec.data() if hasattr(rec, 'data') else dict(rec)
                add(Candidate(d['uuid'], d['fact'], d['source_uuid'], d['target_uuid'], 'cypher_kw'))
        except:
            pass

    # --- Strategy 3: Multi-keyword intersection (top keywords together) ---
    if len(keywords) >= 2:
        for i in range(min(3, len(keywords))):
            for j in range(i+1, min(5, len(keywords))):
                kw1, kw2 = keywords[i], keywords[j]
                try:
                    result = await driver.execute_query(
                        """
                        MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
                        WHERE e.group_id = $group_id
                          AND toLower(e.fact) CONTAINS $kw1
                          AND toLower(e.fact) CONTAINS $kw2
                        RETURN e.uuid AS uuid, e.fact AS fact,
                               s.uuid AS source_uuid, t.uuid AS target_uuid
                        LIMIT 10
                        """,
                        group_id=group_id,
                        kw1=kw1,
                        kw2=kw2,
                    )
                    records = result.records if hasattr(result, 'records') else result
                    for rec in records:
                        d = rec.data() if hasattr(rec, 'data') else dict(rec)
                        add(Candidate(d['uuid'], d['fact'], d['source_uuid'], d['target_uuid'], 'cypher_intersect'))
                except:
                    pass

    # --- Strategy 4: Entity name match → neighborhood edges ---
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
                RETURN e.uuid AS uuid, e.fact AS fact,
                       startNode(e).uuid AS source_uuid, endNode(e).uuid AS target_uuid
                LIMIT 30
                """,
                group_id=group_id,
                keyword=kw,
            )
            records = result.records if hasattr(result, 'records') else result
            for rec in records:
                d = rec.data() if hasattr(rec, 'data') else dict(rec)
                add(Candidate(d['uuid'], d['fact'], d['source_uuid'], d['target_uuid'], 'cypher_neighbor'))
        except:
            pass

    return candidates


# ===========================================================================
# Reranking
# ===========================================================================

def rerank_candidates(
    candidates: list[Candidate],
    ctx: TemporalContext,
    engine: DecayEngine,
) -> list[tuple[Candidate, float]]:
    """Rerank candidates purely by decay activation.

    No Graphiti score blending — we want to isolate the decay engine's effect.
    Candidates from Cypher don't have Graphiti scores anyway.
    """
    scored = []
    for c in candidates:
        activation = compute_activation_from_uuids(
            c.source_uuid, c.target_uuid, ctx, engine
        )
        scored.append((c, activation))

    scored.sort(key=lambda x: -x[1])
    return scored


# ===========================================================================
# Answer matching (same as v2 + improvements)
# ===========================================================================

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[,$%]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def _answer_match(expected: str, fact_text: str) -> bool:
    expected_norm = _normalize(expected)
    fact_norm = _normalize(fact_text)

    # Direct substring
    if expected_norm in fact_norm:
        return True

    expected_words = expected_norm.split()

    # Single-word answers
    if len(expected_words) == 1:
        word = expected_words[0]
        if word.replace('.', '').isdigit():
            fact_numbers = re.findall(r'\d+(?:\.\d+)?', fact_norm)
            if word in fact_numbers:
                return True
        if re.search(r'\b' + re.escape(word) + r'\b', fact_norm):
            return True

    # Multi-word: token overlap
    tokens = [w for w in expected_words if len(w) > 2]
    if tokens:
        matches = sum(1 for w in tokens if w in fact_norm)
        if matches / len(tokens) >= 0.5:
            return True

    return False


# ===========================================================================
# Main evaluation
# ===========================================================================

async def run():
    poc_q_path = POC_ARTIFACTS_DIR / 'poc_questions.json'
    if not poc_q_path.exists():
        print("ERROR: poc_questions.json not found.")
        sys.exit(1)

    questions = json.load(open(poc_q_path, encoding='utf-8'))
    graphiti = get_graphiti_client()
    group_id = "poc_kill_gate"

    engines = {
        'behavioral': DecayEngine.default(),
        'uniform':    DecayEngine.uniform(),
        'cognitive':  DecayEngine.cognitive(),
    }

    ctx = build_temporal_context(
        last_session_ts=time_module.time() - 86400,
        session_message_count=0,
    )

    # Pre-load entity cache
    print("Pre-loading entity cache...")
    cache_result = await graphiti.driver.execute_query(
        """
        MATCH (n:Entity)
        WHERE n.fr_enriched = true AND n.group_id = $group_id
        RETURN n.uuid AS uuid,
               n.fr_primary_category AS primary_category,
               n.fr_membership_weights AS membership_weights,
               n.fr_last_updated_ts AS last_updated_ts,
               n.fr_access_count AS access_count,
               n.fr_emotional_loading AS emotional_loading,
               n.fr_emotional_loading_ts AS emotional_loading_ts,
               n.fr_future_anchor_ts AS future_anchor_ts,
               n.fr_last_reactivation_ts AS last_reactivation_ts,
               n.fr_confidence AS confidence
        """,
        group_id=group_id,
    )
    records = cache_result.records if hasattr(cache_result, 'records') else cache_result
    for rec in records:
        data = rec.data() if hasattr(rec, 'data') else dict(rec)
        uuid = data.pop('uuid')
        _entity_cache[uuid] = data
    print(f"Cached {len(_entity_cache)} enriched entities")

    # Also add a "no rerank" baseline — use Graphiti's original order
    engine_names = list(engines.keys()) + ['no_rerank']
    results = {name: [] for name in engine_names}

    print(f"\n{'='*70}")
    print(f"KILL GATE v3: Hybrid Retrieval + Decay Reranking")
    print(f"{'='*70}\n")

    pool_sizes = []

    for qi, q in enumerate(questions):
        question_text = q['question']
        expected_answer = q['answer']

        print(f"[{qi+1}/{len(questions)}] {question_text[:65]}...")
        print(f"  Answer: {expected_answer[:50]}")

        # Build fat candidate pool
        pool = await build_candidate_pool(question_text, graphiti.driver, graphiti, group_id)
        pool_sizes.append(len(pool))

        # Count how many candidates contain the answer
        answer_in_pool = sum(1 for c in pool if _answer_match(expected_answer, c.fact))
        sources = Counter(c.source for c in pool)

        print(f"  Pool: {len(pool)} candidates ({dict(sources)})")
        print(f"  Answer in pool: {answer_in_pool} facts")

        if not pool:
            for name in engine_names:
                results[name].append({
                    'question_id': q['question_id'],
                    'hit': False,
                    'reason': 'empty_pool',
                })
            continue

        # --- No-rerank baseline: check if answer is in pool at all ---
        # Just check if ANY fact in the pool matches (ceiling)
        no_rerank_hit = answer_in_pool > 0
        # For top-5: take first 5 from Graphiti results in pool
        graphiti_candidates = [c for c in pool if c.source == 'graphiti'][:5]
        nr_hit_at_5 = any(_answer_match(expected_answer, c.fact) for c in graphiti_candidates)

        results['no_rerank'].append({
            'question_id': q['question_id'],
            'question_type': q.get('question_type', 'unknown'),
            'hit': nr_hit_at_5,
            'answer_in_pool': answer_in_pool,
            'pool_size': len(pool),
            'top_facts': [{'text': c.fact[:100], 'source': c.source} for c in graphiti_candidates[:5]],
        })

        # --- Rerank with each decay engine ---
        for engine_name, engine in engines.items():
            reranked = rerank_candidates(pool, ctx, engine)

            hit = False
            top_facts = []
            for rank, (cand, activation) in enumerate(reranked[:5]):
                top_facts.append({
                    'text': cand.fact[:120],
                    'activation': round(activation, 4),
                    'source': cand.source,
                })
                if _answer_match(expected_answer, cand.fact):
                    hit = True

            # Also find rank of first correct answer in reranked list
            answer_rank = None
            for rank, (cand, _) in enumerate(reranked):
                if _answer_match(expected_answer, cand.fact):
                    answer_rank = rank + 1
                    break

            results[engine_name].append({
                'question_id': q['question_id'],
                'question_type': q.get('question_type', 'unknown'),
                'hit': hit,
                'answer_rank': answer_rank,
                'answer_in_pool': answer_in_pool,
                'pool_size': len(pool),
                'top_facts': top_facts,
            })

        # Per-question summary
        marks = {}
        for name in engine_names:
            marks[name] = '✅' if results[name][-1]['hit'] else '❌'
        ranks = {}
        for name in list(engines.keys()):
            r = results[name][-1].get('answer_rank')
            ranks[name] = f"@{r}" if r else "miss"

        print(f"  B:{marks['behavioral']} U:{marks['uniform']} C:{marks['cognitive']} "
              f"NR:{marks['no_rerank']}  |  "
              f"ranks: B={ranks['behavioral']} U={ranks['uniform']} C={ranks['cognitive']}")

    # ===================================================================
    # Summary
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"KILL GATE v3 RESULTS")
    print(f"{'='*70}\n")

    print(f"Pool stats: avg={sum(pool_sizes)/len(pool_sizes):.0f}, "
          f"min={min(pool_sizes)}, max={max(pool_sizes)}")

    # Hit@5
    print(f"\nhit@5:")
    for name in engine_names:
        hits = sum(1 for r in results[name] if r.get('hit', False))
        total = len(results[name])
        pct = 100 * hits / total if total > 0 else 0
        print(f"  {name:12s}: {hits}/{total} ({pct:.0f}%)")

    # Ceiling: how many questions have the answer anywhere in the pool?
    ceil = sum(1 for r in results['behavioral'] if r.get('answer_in_pool', 0) > 0)
    print(f"\n  Pool ceiling (answer exists in candidates): {ceil}/25")

    # Mean reciprocal rank (MRR) — only for questions where answer is in pool
    print(f"\nMean Reciprocal Rank (questions where answer in pool):")
    for name in list(engines.keys()):
        ranks_list = [r['answer_rank'] for r in results[name]
                      if r.get('answer_rank') is not None]
        if ranks_list:
            mrr = sum(1.0 / r for r in ranks_list) / len(ranks_list)
            mean_rank = sum(ranks_list) / len(ranks_list)
            print(f"  {name:12s}: MRR={mrr:.4f}  mean_rank={mean_rank:.1f}  "
                  f"(n={len(ranks_list)})")
        else:
            print(f"  {name:12s}: no hits")

    # By question type
    print(f"\nhit@5 by question type:")
    for qtype in sorted(set(q.get('question_type', 'unknown') for q in questions)):
        print(f"\n  {qtype}:")
        for name in engine_names:
            type_results = [r for r in results[name] if r.get('question_type') == qtype]
            hits = sum(1 for r in type_results if r.get('hit', False))
            total = len(type_results)
            pct = 100 * hits / total if total > 0 else 0
            print(f"    {name:12s}: {hits}/{total} ({pct:.0f}%)")

    # THE comparison
    b_hits = sum(1 for r in results['behavioral'] if r.get('hit', False))
    c_hits = sum(1 for r in results['cognitive'] if r.get('hit', False))
    u_hits = sum(1 for r in results['uniform'] if r.get('hit', False))
    nr_hits = sum(1 for r in results['no_rerank'] if r.get('hit', False))

    print(f"\n{'='*70}")
    print(f"  Behavioral: {b_hits}  |  Uniform: {u_hits}  |  Cognitive: {c_hits}  |  No-rerank: {nr_hits}")

    if b_hits > c_hits:
        print(f"\n  ✅ BEHAVIORAL ({b_hits}) > COGNITIVE ({c_hits})")
        if b_hits > u_hits:
            print(f"  ✅ BEHAVIORAL ({b_hits}) > UNIFORM ({u_hits})")
            print(f"  THESIS HOLDS. Behavioral ontology improves reranking.")
        else:
            print(f"  ⚠️  BEHAVIORAL ({b_hits}) ≤ UNIFORM ({u_hits})")
            print(f"  Beats cognitive but not uniform. Check blend_weight or decay params.")
    elif b_hits == c_hits:
        print(f"\n  ⚠️  BEHAVIORAL ({b_hits}) = COGNITIVE ({c_hits})")
        print(f"  Check MRR above — behavioral might rank correct answers higher even if hit@5 ties.")
    else:
        print(f"\n  ❌ BEHAVIORAL ({b_hits}) < COGNITIVE ({c_hits})")
        print(f"  Investigate decay parameters.")

    # MRR comparison (more granular than hit@5)
    b_ranks = [r['answer_rank'] for r in results['behavioral'] if r.get('answer_rank')]
    c_ranks = [r['answer_rank'] for r in results['cognitive'] if r.get('answer_rank')]
    u_ranks = [r['answer_rank'] for r in results['uniform'] if r.get('answer_rank')]
    if b_ranks and c_ranks:
        b_mrr = sum(1/r for r in b_ranks) / len(b_ranks)
        c_mrr = sum(1/r for r in c_ranks) / len(c_ranks)
        u_mrr = sum(1/r for r in u_ranks) / len(u_ranks) if u_ranks else 0
        print(f"\n  MRR comparison: B={b_mrr:.4f} vs C={c_mrr:.4f} vs U={u_mrr:.4f}")
        if b_mrr > c_mrr:
            print(f"  ✅ Behavioral MRR > Cognitive MRR (even if hit@5 ties)")

    print(f"{'='*70}")

    # Save
    out_path = POC_ARTIFACTS_DIR / 'kill_gate_results_v3.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    await graphiti.close()


if __name__ == '__main__':
    asyncio.run(run())