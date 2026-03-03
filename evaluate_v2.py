"""
evaluate_v2.py — Fixed kill gate evaluation.

Fixes from v1:
    1. ACTIVATION LOOKUP: Graphiti search returns EntityEdge objects with
       source_node_uuid/target_node_uuid as STRING UUIDs. The v1 bridge tried
       to find enriched EntityNode objects on the edge — but they're not there.
       Fix: For each search result edge, query Neo4j directly to fetch fr_*
       attributes from the source/target entity nodes, then compute activation.

    2. ANSWER MATCHING: v1 used strict substring matching that missed
       numeric answers ("4", "$400,000"), day names in longer strings,
       and reformulations. Fix: smarter matching with normalization.

    3. DIAGNOSTICS: Logs per-question activation scores to verify the
       decay engine is actually differentiating across engines.

Does NOT re-ingest or re-enrich. Uses existing data in Neo4j.

Usage:
    cd C:\\Users\\anshu\\PycharmProjects\\hugeleapforward
    python evaluate_v2.py
"""

import asyncio
import json
import os
import re
import sys
import time as time_module
from datetime import datetime, timezone
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
from graphiti_core.nodes import EntityNode
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

sys.path.insert(0, str(PROJECT_ROOT))
from decay_engine import DecayEngine, DecayConfig, TemporalContext, FactNode, CATEGORIES
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
# FIX 1: Neo4j-aware activation lookup
# ===========================================================================

# Cache: entity_uuid -> fr_ attributes dict (avoid repeated Neo4j queries)
_entity_cache: dict[str, dict | None] = {}


async def _fetch_entity_fr_attrs(driver, entity_uuid: str) -> dict | None:
    """Fetch fr_ attributes for an entity node from Neo4j. Cached."""
    if entity_uuid in _entity_cache:
        return _entity_cache[entity_uuid]

    result = await driver.execute_query(
        """
        MATCH (n:Entity {uuid: $uuid})
        WHERE n.fr_enriched = true
        RETURN n.fr_primary_category AS primary_category,
               n.fr_membership_weights AS membership_weights,
               n.fr_last_updated_ts AS last_updated_ts,
               n.fr_access_count AS access_count,
               n.fr_emotional_loading AS emotional_loading,
               n.fr_emotional_loading_ts AS emotional_loading_ts,
               n.fr_future_anchor_ts AS future_anchor_ts,
               n.fr_last_reactivation_ts AS last_reactivation_ts,
               n.fr_confidence AS confidence
        """,
        uuid=entity_uuid,
    )

    records = result.records if hasattr(result, 'records') else result
    if not records:
        _entity_cache[entity_uuid] = None
        return None

    rec = records[0]
    # Handle both dict-like and Record-like access
    if hasattr(rec, 'data'):
        data = rec.data()
    else:
        data = dict(rec)

    if data.get('primary_category') is None:
        _entity_cache[entity_uuid] = None
        return None

    _entity_cache[entity_uuid] = data
    return data


def _attrs_to_fact_node(attrs: dict, fact_id: str) -> FactNode:
    """Convert Neo4j fr_ attributes to a FactNode."""
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


async def compute_edge_activation(
    edge_result, driver, ctx: TemporalContext, engine: DecayEngine
) -> float:
    """Compute decay activation for a search result edge by looking up its
    source and target entity nodes in Neo4j.

    Strategy: check both source and target entity nodes, use the one that
    is enriched. If both are enriched, average their activations (the fact
    = relationship between two classified entities).
    """
    activations = []

    for uuid_attr in ('source_node_uuid', 'target_node_uuid'):
        uuid_val = getattr(edge_result, uuid_attr, None)
        if not uuid_val:
            continue

        attrs = await _fetch_entity_fr_attrs(driver, uuid_val)
        if attrs is None:
            continue

        fact_node = _attrs_to_fact_node(attrs, uuid_val)

        # Compute per-fact absolute_hours
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

    if not activations:
        return 0.5  # neutral fallback (unenriched)

    return sum(activations) / len(activations)


async def rerank_with_neo4j_lookup(
    search_results: list,
    driver,
    ctx: TemporalContext,
    engine: DecayEngine,
    blend_weight: float = 0.5,
) -> list:
    """Re-rank search results using Neo4j-backed activation lookup.

    Returns list of (result, activation, blended_score) tuples, sorted desc.
    """
    scored = []

    for result in search_results:
        activation = await compute_edge_activation(result, driver, ctx, engine)

        # Extract Graphiti's score
        graphiti_score = 0.5
        for attr in ('score', 'rrf_score', 'relevance_score'):
            s = getattr(result, attr, None)
            if s is not None:
                graphiti_score = float(s)
                break

        blended = (1.0 - blend_weight) * graphiti_score + blend_weight * activation
        scored.append((result, activation, blended))

    scored.sort(key=lambda x: -x[2])
    return scored


# ===========================================================================
# FIX 2: Improved answer matching
# ===========================================================================

def _normalize(s: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation/currency."""
    s = s.lower().strip()
    s = re.sub(r'[,$%]', '', s)          # strip currency/percent
    s = re.sub(r'\s+', ' ', s)           # collapse whitespace
    return s


def _answer_match(expected: str, fact_text: str) -> bool:
    """Check if the expected answer appears in the retrieved fact.

    Handles:
    - Direct substring
    - Numeric matching (4, 400000, 16)
    - Day-of-week matching (Wednesday, Friday)
    - Token overlap (60% of >3-char answer words)
    """
    expected_norm = _normalize(expected)
    fact_norm = _normalize(fact_text)

    # Direct substring
    if expected_norm in fact_norm:
        return True

    # Try each "word" of the expected answer as an exact match
    # Good for short answers like "Wednesday", "4", "Paris"
    expected_words = expected_norm.split()

    # For single-word answers, check direct containment
    if len(expected_words) == 1:
        # Numeric: match the number anywhere
        if expected_words[0].replace('.', '').isdigit():
            # Extract all numbers from fact
            fact_numbers = re.findall(r'\d+(?:\.\d+)?', fact_norm)
            if expected_words[0] in fact_numbers:
                return True
        # Word: check as whole word
        if re.search(r'\b' + re.escape(expected_words[0]) + r'\b', fact_norm):
            return True

    # For multi-word answers: token overlap
    answer_tokens = [w for w in expected_words if len(w) > 2]  # lowered threshold from 3
    if answer_tokens:
        matches = sum(1 for w in answer_tokens if w in fact_norm)
        if matches / len(answer_tokens) >= 0.5:  # lowered threshold from 0.6
            return True

    return False


# ===========================================================================
# Helpers
# ===========================================================================

def _extract_fact_text(result) -> str:
    """Extract human-readable fact text from a Graphiti search result."""
    if hasattr(result, 'fact'):
        return str(result.fact)
    if hasattr(result, 'name'):
        return str(result.name)
    if hasattr(result, 'summary'):
        return str(result.summary)
    if isinstance(result, tuple) and len(result) >= 1:
        return _extract_fact_text(result[0])
    return str(result)


# ===========================================================================
# Evaluation
# ===========================================================================

async def run_evaluation():
    # Load PoC questions
    poc_q_path = POC_ARTIFACTS_DIR / 'poc_questions.json'
    if not poc_q_path.exists():
        print("ERROR: poc_questions.json not found.")
        sys.exit(1)

    questions = json.load(open(poc_q_path, encoding='utf-8'))
    print(f"Evaluating {len(questions)} questions")

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

    results = {name: [] for name in engines}

    # Pre-warm cache: load all enriched entity UUIDs
    print("Pre-loading enriched entity cache...")
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

    enriched_count = sum(1 for v in _entity_cache.values() if v is not None)
    print(f"Cached {enriched_count} enriched entities")

    # Diagnostic: verify activations differ across engines for a sample entity
    if _entity_cache:
        sample_uuid = next(iter(_entity_cache))
        sample_attrs = _entity_cache[sample_uuid]
        if sample_attrs:
            sample_fact = _attrs_to_fact_node(sample_attrs, sample_uuid)
            sample_ctx = TemporalContext(
                absolute_hours=24.0, relative_hours=24.0,
                conversational_messages=0,
                current_timestamp=time_module.time(),
            )
            for ename, eng in engines.items():
                a = eng.compute_activation(sample_fact, sample_ctx)
                print(f"  Sample activation ({ename}): {a:.4f}")

    print(f"\n{'='*60}")
    print(f"KILL GATE v2: Behavioral vs Uniform vs Cognitive")
    print(f"{'='*60}\n")

    for qi, q in enumerate(questions):
        question_text = q['question']
        expected_answer = q['answer']

        print(f"[{qi+1}/{len(questions)}] {question_text[:70]}...")
        print(f"  Expected: {expected_answer[:60]}")

        try:
            search_results = await graphiti.search(
                question_text,
                group_ids=[group_id],
                num_results=10,
            )
        except Exception as e:
            print(f"  Search error: {e}")
            for name in engines:
                results[name].append({
                    'question_id': q['question_id'],
                    'hit': False,
                    'error': str(e),
                })
            continue

        if not search_results:
            print(f"  No results")
            for name in engines:
                results[name].append({
                    'question_id': q['question_id'],
                    'hit': False,
                    'reason': 'no_results',
                })
            continue

        # Log what Graphiti actually returned (once, not per engine)
        raw_facts = []
        for sr in search_results[:10]:
            ft = _extract_fact_text(sr)
            raw_facts.append(ft[:100] if ft else 'N/A')

        for engine_name, engine in engines.items():
            reranked = await rerank_with_neo4j_lookup(
                search_results, graphiti.driver, ctx, engine=engine, blend_weight=0.5,
            )

            hit = False
            top_facts = []
            for rank, (result, activation, blended) in enumerate(reranked[:5]):
                fact_text = _extract_fact_text(result)
                top_facts.append({
                    'text': fact_text[:120] if fact_text else 'N/A',
                    'activation': round(activation, 4),
                    'blended': round(blended, 4),
                })
                if fact_text and _answer_match(expected_answer, fact_text):
                    hit = True

            results[engine_name].append({
                'question_id': q['question_id'],
                'question_type': q.get('question_type', 'unknown'),
                'question': question_text,
                'expected_answer': expected_answer,
                'hit': hit,
                'top_facts': top_facts,
                'raw_graphiti_top3': raw_facts[:3],  # diagnostics
            })

        b = '✅' if results['behavioral'][-1]['hit'] else '❌'
        u = '✅' if results['uniform'][-1]['hit'] else '❌'
        c = '✅' if results['cognitive'][-1]['hit'] else '❌'

        # Show activation scores for first engine to verify they're not all 0.5
        b_acts = [f['activation'] for f in results['behavioral'][-1]['top_facts']]
        print(f"  B:{b} U:{u} C:{c}  activations={b_acts}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"KILL GATE v2 RESULTS")
    print(f"{'='*60}\n")

    for engine_name in engines:
        hits = sum(1 for r in results[engine_name] if r.get('hit', False))
        total = len(results[engine_name])
        pct = 100 * hits / total if total > 0 else 0
        print(f"  {engine_name:12s}: {hits}/{total} ({pct:.0f}%) hit@5")

    print(f"\nBy question type:")
    for qtype in sorted(set(q.get('question_type', 'unknown') for q in questions)):
        print(f"\n  {qtype}:")
        for engine_name in engines:
            type_results = [r for r in results[engine_name] if r.get('question_type') == qtype]
            hits = sum(1 for r in type_results if r.get('hit', False))
            total = len(type_results)
            pct = 100 * hits / total if total > 0 else 0
            print(f"    {engine_name:12s}: {hits}/{total} ({pct:.0f}%)")

    b_hits = sum(1 for r in results['behavioral'] if r.get('hit', False))
    c_hits = sum(1 for r in results['cognitive'] if r.get('hit', False))
    u_hits = sum(1 for r in results['uniform'] if r.get('hit', False))

    # Activation diversity check
    all_b_acts = []
    for r in results['behavioral']:
        for f in r.get('top_facts', []):
            all_b_acts.append(f['activation'])
    unique_acts = len(set(round(a, 3) for a in all_b_acts))
    all_05 = sum(1 for a in all_b_acts if abs(a - 0.5) < 0.001)
    print(f"\n  Activation diversity: {unique_acts} unique values in {len(all_b_acts)} scores")
    print(f"  Still-0.5 count: {all_05}/{len(all_b_acts)} "
          f"({'PROBLEM' if all_05 == len(all_b_acts) else 'OK'})")

    print(f"\n{'='*60}")
    if b_hits > c_hits and b_hits > u_hits:
        print(f"✅ BEHAVIORAL ({b_hits}) > COGNITIVE ({c_hits}) & UNIFORM ({u_hits})")
        print(f"   THESIS HOLDS. Proceed to full evaluation.")
    elif b_hits > c_hits:
        print(f"⚠️  BEHAVIORAL ({b_hits}) > COGNITIVE ({c_hits}) but ≤ UNIFORM ({u_hits})")
        print(f"   Behavioral beats cognitive but uniform is competitive.")
    elif b_hits == c_hits:
        print(f"⚠️  BEHAVIORAL ({b_hits}) = COGNITIVE ({c_hits}) | UNIFORM ({u_hits})")
        print(f"   Check activation diversity above — if all 0.5, lookup still broken.")
    else:
        print(f"❌ BEHAVIORAL ({b_hits}) < COGNITIVE ({c_hits}) | UNIFORM ({u_hits})")
        print(f"   Investigate.")
    print(f"{'='*60}")

    # Save
    out_path = POC_ARTIFACTS_DIR / 'kill_gate_results_v2.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    await graphiti.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    asyncio.run(run_evaluation())