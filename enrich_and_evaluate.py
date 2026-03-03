"""
enrich_and_evaluate.py — Recovery script after ingest crash.

All 25 questions are safe in Neo4j. This script:
    1. Loads all entity nodes from the poc_kill_gate group
    2. Classifies each with your v7 behavioral ontology (via Grok)
    3. Writes ONLY fr_ attributes via direct Cypher (bypasses node.save() embedding bug)
    4. Runs the kill gate evaluation

Usage:
    cd C:\\Users\\anshu\\PycharmProjects\\hugeleapforward
    python enrich_and_evaluate.py
"""

import asyncio
import json
import os
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
from decay_engine import DecayEngine, TemporalContext, CATEGORIES
from graphiti_bridge import (
    enrich_entity_node, entity_to_fact_node, is_enriched,
    build_temporal_context, rerank_by_activation,
    ATTR_PRIMARY_CATEGORY, ATTR_MEMBERSHIP_WEIGHTS,
    ATTR_LAST_UPDATED_TS, ATTR_ACCESS_COUNT, ATTR_CONFIDENCE,
    ATTR_ENRICHED, ATTR_EMOTIONAL_LOADING, ATTR_EMOTIONAL_LOADING_TS,
    ATTR_FUTURE_ANCHOR_TS, ATTR_LAST_REACTIVATION_TS,
)


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


# ---------------------------------------------------------------------------
# Classifier (same as ingest_poc.py)
# ---------------------------------------------------------------------------

async def classify_entity(entity_name: str, entity_summary: str,
                          xai_client: AsyncOpenAI) -> dict:
    system = """You are a memory classification system. Given an entity from a knowledge graph, classify it into behavioral categories.

## Categories
1. OBLIGATIONS — Tasks, deadlines, appointments, promises.
2. RELATIONAL_BONDS — Relationships: family, partners, friends, colleagues.
3. HEALTH_WELLBEING — Physical/mental health, medications, diagnoses, fitness.
4. IDENTITY_SELF_CONCEPT — Core stable traits: ethnicity, name, occupation, heritage, values.
5. HOBBIES_RECREATION — Active leisure with skill/equipment investment.
6. PREFERENCES_HABITS — Current tastes, media, food choices, subscriptions, routines.
7. INTELLECTUAL_INTERESTS — Curiosities, learning goals without a deliverable.
8. LOGISTICAL_CONTEXT — Transient scheduling, one-time locations, errands.
9. PROJECTS_ENDEAVORS — Ongoing works with milestones: startups, research, creative projects.
10. FINANCIAL_MATERIAL — Budget, income, expenses, purchases, devices, assets.
11. OTHER — Genuinely does not fit 1-10.

## Rules
- Classify by what the entity IS, not conversational context.
- Soft membership weights summing to 1.0. Primary gets highest weight (>= 0.3).
- Detect emotional loading if present.

## Output (strict JSON)
{"primary_category": "CATEGORY", "weights": {"OBLIGATIONS": 0.0, "RELATIONAL_BONDS": 0.0, "HEALTH_WELLBEING": 0.0, "IDENTITY_SELF_CONCEPT": 0.0, "HOBBIES_RECREATION": 0.0, "PREFERENCES_HABITS": 0.0, "INTELLECTUAL_INTERESTS": 0.0, "LOGISTICAL_CONTEXT": 0.0, "PROJECTS_ENDEAVORS": 0.0, "FINANCIAL_MATERIAL": 0.0, "OTHER": 0.0}, "emotional_loading": {"detected": false, "type": null, "intensity": null}, "confidence": "high"}"""

    prompt = f"""Classify this entity/fact into behavioral categories.

**Entity name:** {entity_name}
**Entity summary:** {entity_summary}

Respond with JSON only."""

    try:
        resp = await xai_client.chat.completions.create(
            model="grok-4-1-fast-non-reasoning",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"  Classification error for {entity_name}: {e}")
        return {
            'primary_category': 'OTHER',
            'weights': {'OTHER': 1.0},
            'emotional_loading': {'detected': False},
            'confidence': 'low',
        }


# ---------------------------------------------------------------------------
# Direct Cypher update (bypasses node.save() — no embedding touch)
# ---------------------------------------------------------------------------

async def write_enrichment_to_neo4j(driver, node_uuid: str, classification: dict):
    """Write ONLY fr_ attributes to Neo4j via direct Cypher. Never touches embeddings."""

    now_ts = datetime.now(timezone.utc).timestamp()

    emo = classification.get('emotional_loading', {})
    emo_detected = isinstance(emo, dict) and emo.get('detected', False)

    weights_json = json.dumps(classification.get('weights', {}))

    await driver.execute_query(
        """
        MATCH (n:Entity {uuid: $uuid})
        SET n.fr_primary_category = $primary_category,
            n.fr_membership_weights = $membership_weights,
            n.fr_last_updated_ts = $last_updated_ts,
            n.fr_access_count = 0,
            n.fr_confidence = $confidence,
            n.fr_enriched = true,
            n.fr_emotional_loading = $emotional_loading,
            n.fr_emotional_loading_ts = $emotional_loading_ts,
            n.fr_future_anchor_ts = null,
            n.fr_last_reactivation_ts = null
        """,
        uuid=node_uuid,
        primary_category=classification.get('primary_category', 'OTHER'),
        membership_weights=weights_json,
        last_updated_ts=now_ts,
        confidence=classification.get('confidence', 'unknown'),
        emotional_loading=emo_detected,
        emotional_loading_ts=now_ts if emo_detected else None,
    )


# ---------------------------------------------------------------------------
# Phase 1: Enrich
# ---------------------------------------------------------------------------

async def run_enrichment(graphiti):
    group_id = "poc_kill_gate"

    xai_client = AsyncOpenAI(
        api_key=os.environ['XAI_API_KEY'],
        base_url="https://api.x.ai/v1",
    )

    # Fetch all entity nodes
    print("Loading entity nodes from Neo4j...")
    entity_nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id])
    print(f"Found {len(entity_nodes)} entity nodes")

    # Check how many are already enriched (in case of partial run)
    already_enriched = sum(1 for n in entity_nodes
                          if n.attributes and n.attributes.get(ATTR_ENRICHED, False))
    print(f"Already enriched: {already_enriched}")
    to_classify = [n for n in entity_nodes
                   if not (n.attributes and n.attributes.get(ATTR_ENRICHED, False))]
    print(f"Need classification: {len(to_classify)}")

    if not to_classify:
        print("All nodes already enriched!")
        return entity_nodes

    enrichment_results = []
    start = time_module.time()

    # Process in batches of 10 (parallel classification, serial Neo4j writes)
    for i in range(0, len(to_classify), 10):
        batch = to_classify[i:i+10]

        # Classify in parallel
        classifications = await asyncio.gather(*[
            classify_entity(n.name, n.summary, xai_client)
            for n in batch
        ])

        # Write to Neo4j (serial to avoid contention)
        for node, classification in zip(batch, classifications):
            await write_enrichment_to_neo4j(graphiti.driver, node.uuid, classification)

            enrichment_results.append({
                'uuid': node.uuid,
                'name': node.name,
                'primary_category': classification.get('primary_category', 'OTHER'),
            })

        done = min(i + 10, len(to_classify))
        if done % 100 == 0 or done == len(to_classify):
            elapsed = time_module.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(to_classify) - done) / rate if rate > 0 else 0
            print(f"  Classified {done}/{len(to_classify)} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time_module.time() - start
    print(f"\nEnrichment complete: {len(enrichment_results)} nodes in {elapsed:.0f}s")

    # Category distribution
    cats = Counter(r['primary_category'] for r in enrichment_results)
    print(f"\nCategory distribution:")
    for cat, count in cats.most_common():
        pct = 100 * count / len(enrichment_results)
        print(f"  {cat}: {count} ({pct:.0f}%)")

    # Save
    with open(POC_ARTIFACTS_DIR / 'enrichment_results.json', 'w') as f:
        json.dump(enrichment_results, f, indent=2)

    # Re-fetch nodes with updated attributes
    entity_nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id])
    return entity_nodes


# ---------------------------------------------------------------------------
# Phase 2: Evaluate — the kill gate
# ---------------------------------------------------------------------------

def make_fact_node_from_neo4j_attrs(node):
    """Convert EntityNode with fr_ attributes stored as Neo4j properties."""
    attrs = node.attributes or {}

    if not attrs.get('fr_enriched', False):
        return None

    raw_weights = attrs.get('fr_membership_weights', '{}')
    if isinstance(raw_weights, str):
        weights = json.loads(raw_weights)
    else:
        weights = raw_weights

    for cat in CATEGORIES:
        weights.setdefault(cat, 0.0)

    from decay_engine import FactNode
    return FactNode(
        fact_id=node.uuid,
        membership_weights=weights,
        primary_category=attrs.get('fr_primary_category', 'OTHER'),
        last_updated_ts=attrs.get('fr_last_updated_ts', 0.0),
        base_activation=1.0,
        future_anchor_ts=attrs.get('fr_future_anchor_ts'),
        emotional_loading=attrs.get('fr_emotional_loading', False),
        emotional_loading_ts=attrs.get('fr_emotional_loading_ts'),
        last_reactivation_ts=attrs.get('fr_last_reactivation_ts'),
        access_count=attrs.get('fr_access_count', 0),
    )


async def run_evaluation(graphiti):
    # Load PoC questions
    poc_q_path = POC_ARTIFACTS_DIR / 'poc_questions.json'
    if not poc_q_path.exists():
        print("ERROR: poc_questions.json not found.")
        sys.exit(1)

    questions = json.load(open(poc_q_path, encoding='utf-8'))
    print(f"\nEvaluating {len(questions)} questions")

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

    print(f"\n{'='*60}")
    print(f"KILL GATE: Behavioral vs Uniform vs Cognitive")
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

        for engine_name, engine in engines.items():
            reranked = rerank_by_activation(
                search_results, ctx, engine=engine, blend_weight=0.5,
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
                'hit': hit,
                'top_facts': top_facts,
            })

        b = '✅' if results['behavioral'][-1]['hit'] else '❌'
        u = '✅' if results['uniform'][-1]['hit'] else '❌'
        c = '✅' if results['cognitive'][-1]['hit'] else '❌'
        print(f"  Behavioral: {b}  |  Uniform: {u}  |  Cognitive: {c}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"KILL GATE RESULTS")
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

    print(f"\n{'='*60}")
    if b_hits > c_hits and b_hits > u_hits:
        print(f"✅ BEHAVIORAL ({b_hits}) > COGNITIVE ({c_hits}) & UNIFORM ({u_hits})")
        print(f"   THESIS HOLDS. Proceed to full evaluation.")
    elif b_hits > c_hits:
        print(f"⚠️  BEHAVIORAL ({b_hits}) > COGNITIVE ({c_hits}) but ≤ UNIFORM ({u_hits})")
        print(f"   Behavioral beats cognitive but uniform is competitive. Check blend_weight.")
    elif b_hits == c_hits:
        print(f"⚠️  BEHAVIORAL ({b_hits}) = COGNITIVE ({c_hits}) | UNIFORM ({u_hits})")
        print(f"   Ambiguous. Tune decay rates or test on more questions.")
    else:
        print(f"❌ BEHAVIORAL ({b_hits}) < COGNITIVE ({c_hits}) | UNIFORM ({u_hits})")
        print(f"   Investigate classifier accuracy, decay params, blend_weight.")
    print(f"{'='*60}")

    with open(POC_ARTIFACTS_DIR / 'kill_gate_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {POC_ARTIFACTS_DIR / 'kill_gate_results.json'}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_fact_text(result) -> str:
    if hasattr(result, 'fact'):
        return str(result.fact)
    if hasattr(result, 'name'):
        return str(result.name)
    if hasattr(result, 'summary'):
        return str(result.summary)
    if isinstance(result, tuple) and len(result) >= 1:
        return _extract_fact_text(result[0])
    return str(result)


def _answer_match(expected: str, fact_text: str) -> bool:
    expected_lower = expected.lower().strip()
    fact_lower = fact_text.lower().strip()

    if expected_lower in fact_lower:
        return True

    answer_words = [w for w in expected_lower.split() if len(w) > 3]
    if answer_words:
        matches = sum(1 for w in answer_words if w in fact_lower)
        if matches / len(answer_words) >= 0.6:
            return True

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    graphiti = get_graphiti_client()

    print("Phase 1: Enriching entity nodes...")
    await run_enrichment(graphiti)

    print("\nPhase 2: Kill gate evaluation...")
    await run_evaluation(graphiti)

    await graphiti.close()


if __name__ == '__main__':
    asyncio.run(main())