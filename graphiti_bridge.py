"""
graphiti_bridge.py — Wires decay_engine into Graphiti without modifying Graphiti source.

Three integration points:
    1. INGESTION:  enrich_entity_node()  — after Graphiti extracts an entity, stamp it
                                            with behavioral ontology fields via attributes dict
    2. RETRIEVAL:  entity_to_fact_node()  — convert Graphiti EntityNode → decay_engine FactNode
    3. RANKING:    rerank_by_activation() — post-process Graphiti search results with decay scores

Placement:
    C:\\Users\\anshu\\PycharmProjects\\hugeleapforward\\graphiti\\graphiti_bridge.py
    (sibling to your graphiti_core/ fork, NOT inside it)

Usage example — ingestion:
    from graphiti_bridge import enrich_entity_node

    # After Graphiti's add_episode extracts entity nodes...
    for node in extracted_nodes:
        classification = await classify_fact(node.summary)  # your existing classifier
        enrich_entity_node(node, classification)
        await node.save(driver)

Usage example — retrieval:
    from graphiti_bridge import rerank_by_activation, build_temporal_context

    # After Graphiti's search returns results...
    results = await graphiti.search("What's my mom's condition?")
    ctx = build_temporal_context(last_session_ts=user.last_session_ts)
    reranked = rerank_by_activation(results, ctx)
"""

import time
from datetime import datetime, timezone
from typing import Any, Optional

from decay_engine import (
    DecayEngine,
    DecayConfig,
    FactNode,
    TemporalContext,
    CATEGORIES,
)

# ============================================================================
# Constants — attribute keys stored in EntityNode.attributes
# ============================================================================

# These keys get flattened into Neo4j node properties automatically
# via EntityNode.save() → entity_data.update(self.attributes)
ATTR_PRIMARY_CATEGORY = 'fr_primary_category'
ATTR_MEMBERSHIP_WEIGHTS = 'fr_membership_weights'  # JSON string in Neo4j
ATTR_FUTURE_ANCHOR_TS = 'fr_future_anchor_ts'  # float (Unix timestamp) or None
ATTR_EMOTIONAL_LOADING = 'fr_emotional_loading'  # bool
ATTR_EMOTIONAL_LOADING_TS = 'fr_emotional_loading_ts'  # float or None
ATTR_LAST_UPDATED_TS = 'fr_last_updated_ts'  # float (Unix timestamp)
ATTR_ACCESS_COUNT = 'fr_access_count'  # int
ATTR_LAST_REACTIVATION_TS = 'fr_last_reactivation_ts'  # float or None
ATTR_CONFIDENCE = 'fr_confidence'  # str: high/medium/low
ATTR_ENRICHED = 'fr_enriched'  # bool — sentinel to check if node has been classified


# Prefix ensures no collision with Graphiti's own attributes or user-defined entity types
# "fr_" = Fortunate Recall


# ============================================================================
# 1. INGESTION: Stamp behavioral ontology onto EntityNode.attributes
# ============================================================================

def enrich_entity_node(entity_node, classification: dict,
                       reference_time: Optional[datetime] = None):
    """Stamp a Graphiti EntityNode with behavioral ontology fields.

    Call this AFTER Graphiti extracts the entity, BEFORE node.save().

    Args:
        entity_node: graphiti_core EntityNode instance
        classification: Output from your classifier (classify_facts.py format):
            {
                'primary_category': 'HEALTH_WELLBEING',
                'weights': {'HEALTH_WELLBEING': 0.7, 'RELATIONAL_BONDS': 0.2, ...},
                'emotional_loading': {'detected': True, 'type': 'anxiety', 'intensity': 'high'},
                'confidence': 'high',
            }
        reference_time: When this fact was stated. Defaults to now.

    The fields are stored in entity_node.attributes, which Graphiti automatically
    persists as top-level Neo4j node properties. No schema changes needed.
    """
    import json as _json

    now_ts = (reference_time or datetime.now(timezone.utc)).timestamp()

    attrs = entity_node.attributes
    if attrs is None:
        attrs = {}
        entity_node.attributes = attrs

    # Core behavioral ontology fields
    attrs[ATTR_PRIMARY_CATEGORY] = classification.get('primary_category', 'OTHER')
    attrs[ATTR_MEMBERSHIP_WEIGHTS] = _json.dumps(classification.get('weights', {}))
    attrs[ATTR_LAST_UPDATED_TS] = now_ts
    attrs[ATTR_ACCESS_COUNT] = 0
    attrs[ATTR_CONFIDENCE] = classification.get('confidence', 'unknown')
    attrs[ATTR_ENRICHED] = True

    # Emotional loading
    emo = classification.get('emotional_loading', {})
    if isinstance(emo, dict) and emo.get('detected', False):
        attrs[ATTR_EMOTIONAL_LOADING] = True
        attrs[ATTR_EMOTIONAL_LOADING_TS] = now_ts
    else:
        attrs[ATTR_EMOTIONAL_LOADING] = False
        attrs[ATTR_EMOTIONAL_LOADING_TS] = None

    # Future anchor (caller sets this separately for deadline-bearing facts)
    if ATTR_FUTURE_ANCHOR_TS not in attrs:
        attrs[ATTR_FUTURE_ANCHOR_TS] = None

    # Reactivation tracking
    attrs[ATTR_LAST_REACTIVATION_TS] = None


def set_future_anchor(entity_node, deadline: datetime):
    """Set a future anchor timestamp for anticipatory activation.

    Call for OBLIGATIONS and PROJECTS_ENDEAVORS facts that have deadlines.

    Args:
        entity_node: Graphiti EntityNode (must be enriched first)
        deadline: The deadline/event datetime
    """
    if entity_node.attributes is None:
        entity_node.attributes = {}
    entity_node.attributes[ATTR_FUTURE_ANCHOR_TS] = deadline.timestamp()


def is_enriched(entity_node) -> bool:
    """Check if a node has been stamped with behavioral ontology fields."""
    if entity_node.attributes is None:
        return False
    return entity_node.attributes.get(ATTR_ENRICHED, False)


# ============================================================================
# 2. RETRIEVAL: Convert EntityNode → FactNode for decay computation
# ============================================================================

def entity_to_fact_node(entity_node) -> Optional[FactNode]:
    """Convert a Graphiti EntityNode to a decay_engine FactNode.

    Returns None if the node hasn't been enriched with behavioral ontology.
    """
    import json as _json

    attrs = entity_node.attributes or {}

    if not attrs.get(ATTR_ENRICHED, False):
        return None

    # Parse membership weights from JSON string (Neo4j stores as string)
    raw_weights = attrs.get(ATTR_MEMBERSHIP_WEIGHTS, '{}')
    if isinstance(raw_weights, str):
        weights = _json.loads(raw_weights)
    else:
        weights = raw_weights

    # Ensure all categories present
    for cat in CATEGORIES:
        weights.setdefault(cat, 0.0)

    return FactNode(
        fact_id=entity_node.uuid,
        membership_weights=weights,
        primary_category=attrs.get(ATTR_PRIMARY_CATEGORY, 'OTHER'),
        last_updated_ts=attrs.get(ATTR_LAST_UPDATED_TS, 0.0),
        base_activation=1.0,
        future_anchor_ts=attrs.get(ATTR_FUTURE_ANCHOR_TS),
        emotional_loading=attrs.get(ATTR_EMOTIONAL_LOADING, False),
        emotional_loading_ts=attrs.get(ATTR_EMOTIONAL_LOADING_TS),
        last_reactivation_ts=attrs.get(ATTR_LAST_REACTIVATION_TS),
        access_count=attrs.get(ATTR_ACCESS_COUNT, 0),
    )


# ============================================================================
# 3. RANKING: Build temporal context and re-rank search results
# ============================================================================

def build_temporal_context(
        last_session_ts: Optional[float] = None,
        session_message_count: int = 0,
        now: Optional[float] = None,
) -> TemporalContext:
    """Build a TemporalContext for the current retrieval moment.

    Args:
        last_session_ts: Unix timestamp of when user's previous session started.
                         If None, defaults to 24h ago (new user assumption).
        session_message_count: Messages exchanged in current session so far.
        now: Current Unix timestamp. Defaults to time.time().

    Returns:
        TemporalContext ready for decay_engine.compute_activation()

    Note: absolute_hours is computed per-fact inside rerank_by_activation,
    since each fact has a different last_updated_ts.
    """
    current_ts = now or time.time()

    if last_session_ts is None:
        last_session_ts = current_ts - 86400.0  # 24h ago default

    relative_hours = (current_ts - last_session_ts) / 3600.0

    return TemporalContext(
        absolute_hours=0.0,  # placeholder — computed per-fact in rerank
        relative_hours=max(0.0, relative_hours),
        conversational_messages=session_message_count,
        current_timestamp=current_ts,
    )


def rerank_by_activation(
        search_results: list,
        ctx: TemporalContext,
        engine: Optional[DecayEngine] = None,
        blend_weight: float = 0.5,
) -> list:
    """Re-rank Graphiti search results using behavioral decay activation.

    Takes Graphiti's hybrid retrieval results (semantic + BM25 + graph distance)
    and blends in temporal activation from the decay engine.

    Args:
        search_results: List of Graphiti search results. Each result should have:
            - .fact or .edge: the EntityEdge with source/target EntityNodes
            - .score: Graphiti's retrieval relevance score (RRF, etc.)
            Exact structure depends on whether you use graphiti.search() or graphiti._search()
        ctx: TemporalContext from build_temporal_context() (absolute_hours=0, filled per-fact)
        engine: DecayEngine instance. Defaults to DecayEngine.default().
        blend_weight: How much to weight activation vs Graphiti score.
            0.0 = pure Graphiti ranking (ablation baseline)
            1.0 = pure activation ranking
            0.5 = equal blend (default)

    Returns:
        Re-ranked list of (result, activation, blended_score) tuples, sorted descending.
    """
    if engine is None:
        engine = DecayEngine.default()

    scored = []

    for result in search_results:
        # Extract the entity node(s) from the search result
        # Graphiti search returns edges (facts) with source/target nodes
        # We compute activation on the EDGE's attributes (facts are edges in Graphiti)
        # or on the source/target nodes depending on where we stored the classification

        activation = _compute_result_activation(result, ctx, engine)

        # Get Graphiti's original score (normalized to 0-1 range)
        graphiti_score = _extract_graphiti_score(result)

        # Blend: higher = more relevant
        blended = (1.0 - blend_weight) * graphiti_score + blend_weight * activation

        scored.append((result, activation, blended))

    # Sort by blended score descending
    scored.sort(key=lambda x: -x[2])
    return scored


def _compute_result_activation(result, ctx: TemporalContext, engine: DecayEngine) -> float:
    """Compute activation for a single search result.

    Handles both node-level and edge-level results from Graphiti's search.
    """
    # Try to get entity node from the result
    # Graphiti's search() returns EntityEdge objects; _search() returns richer objects
    node = None

    # Pattern 1: result is an EntityEdge with source_node/target_node
    if hasattr(result, 'source_node') and result.source_node is not None:
        if is_enriched(result.source_node):
            node = result.source_node
    if node is None and hasattr(result, 'target_node') and result.target_node is not None:
        if is_enriched(result.target_node):
            node = result.target_node

    # Pattern 2: result is an EntityNode directly (from node search)
    if node is None and hasattr(result, 'attributes'):
        if is_enriched(result):
            node = result

    # Pattern 3: result has a .fact or .edge attribute
    if node is None:
        for attr_name in ('fact', 'edge', 'node'):
            inner = getattr(result, attr_name, None)
            if inner is not None and hasattr(inner, 'attributes') and is_enriched(inner):
                node = inner
                break

    if node is None:
        # Not enriched — return neutral activation (doesn't penalize unenriched nodes)
        return 0.5

    fact_node = entity_to_fact_node(node)
    if fact_node is None:
        return 0.5

    # Compute per-fact absolute_hours (each fact has its own last_updated_ts)
    if ctx.current_timestamp and fact_node.last_updated_ts > 0:
        absolute_hours = (ctx.current_timestamp - fact_node.last_updated_ts) / 3600.0
    else:
        absolute_hours = 0.0

    # Build fact-specific context with correct absolute_hours
    fact_ctx = TemporalContext(
        absolute_hours=max(0.0, absolute_hours),
        relative_hours=ctx.relative_hours,
        conversational_messages=ctx.conversational_messages,
        current_timestamp=ctx.current_timestamp,
    )

    return engine.compute_activation(fact_node, fact_ctx)


def _extract_graphiti_score(result) -> float:
    """Extract Graphiti's relevance score from a search result.

    Graphiti's search methods return results with different score attributes
    depending on the search type (RRF, semantic, BM25, etc.).
    """
    # Try common score attribute names
    for attr in ('score', 'rrf_score', 'relevance_score', 'similarity_score'):
        score = getattr(result, attr, None)
        if score is not None:
            return float(score)

    # If result is a tuple/list (some search methods return (entity, score) pairs)
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        try:
            return float(result[-1])
        except (TypeError, ValueError):
            pass

    return 0.5  # neutral default


# ============================================================================
# 4. EDGE ENRICHMENT: For fact-level classification (edges = facts in Graphiti)
# ============================================================================

def enrich_entity_edge(entity_edge, classification: dict,
                       reference_time: Optional[datetime] = None):
    """Stamp a Graphiti EntityEdge with behavioral ontology fields.

    In Graphiti, facts are stored as EDGES between entity nodes.
    "Kendra loves Adidas shoes" → EntityEdge between Kendra and Adidas shoes.

    The behavioral classification should be on the FACT (edge), not just the entities.
    This function works identically to enrich_entity_node but on an edge.

    Graphiti edges don't have an `attributes` dict by default, so we store
    the classification in edge properties that get persisted to Neo4j.
    """
    import json as _json

    now_ts = (reference_time or datetime.now(timezone.utc)).timestamp()

    # EntityEdge doesn't have .attributes — check if it exists or use a dict
    if not hasattr(entity_edge, 'attributes'):
        # Store in a way that's accessible but won't break Graphiti's save
        if not hasattr(entity_edge, '_fr_metadata'):
            entity_edge._fr_metadata = {}
        attrs = entity_edge._fr_metadata
    else:
        attrs = entity_edge.attributes
        if attrs is None:
            attrs = {}
            entity_edge.attributes = attrs

    attrs[ATTR_PRIMARY_CATEGORY] = classification.get('primary_category', 'OTHER')
    attrs[ATTR_MEMBERSHIP_WEIGHTS] = _json.dumps(classification.get('weights', {}))
    attrs[ATTR_LAST_UPDATED_TS] = now_ts
    attrs[ATTR_ACCESS_COUNT] = 0
    attrs[ATTR_CONFIDENCE] = classification.get('confidence', 'unknown')
    attrs[ATTR_ENRICHED] = True

    emo = classification.get('emotional_loading', {})
    if isinstance(emo, dict) and emo.get('detected', False):
        attrs[ATTR_EMOTIONAL_LOADING] = True
        attrs[ATTR_EMOTIONAL_LOADING_TS] = now_ts
    else:
        attrs[ATTR_EMOTIONAL_LOADING] = False
        attrs[ATTR_EMOTIONAL_LOADING_TS] = None

    attrs[ATTR_FUTURE_ANCHOR_TS] = None
    attrs[ATTR_LAST_REACTIVATION_TS] = None


# ============================================================================
# 5. CONVENIENCE: Record access for frequency-based parameter evolution
# ============================================================================

def record_access(entity_node, driver=None):
    """Record that a fact was accessed during retrieval.

    Updates access_count and last_updated_ts in attributes.
    Caller should persist via node.save(driver) if driver is provided.

    This feeds into future per-user parameter evolution:
    frequently accessed facts get slightly boosted base activation.
    """
    attrs = entity_node.attributes or {}
    attrs[ATTR_ACCESS_COUNT] = attrs.get(ATTR_ACCESS_COUNT, 0) + 1
    attrs[ATTR_LAST_UPDATED_TS] = time.time()
    entity_node.attributes = attrs


# ============================================================================
# 6. DIAGNOSTICS: Inspect behavioral ontology state of a node
# ============================================================================

def inspect_node(entity_node, engine: Optional[DecayEngine] = None) -> dict:
    """Get a diagnostic report for a node's behavioral ontology state.

    Useful for debugging and paper figures.
    """
    import json as _json

    if not is_enriched(entity_node):
        return {'enriched': False, 'node_uuid': entity_node.uuid, 'name': entity_node.name}

    if engine is None:
        engine = DecayEngine.default()

    fact_node = entity_to_fact_node(entity_node)
    if fact_node is None:
        return {'enriched': False, 'node_uuid': entity_node.uuid, 'name': entity_node.name}

    attrs = entity_node.attributes or {}
    weights = _json.loads(attrs.get(ATTR_MEMBERSHIP_WEIGHTS, '{}'))

    # Current activation at this moment
    now = time.time()
    absolute_hours = (now - fact_node.last_updated_ts) / 3600.0
    ctx = TemporalContext(
        absolute_hours=max(0.0, absolute_hours),
        relative_hours=24.0,  # assume 24h since last session for diagnostics
        conversational_messages=0,
        current_timestamp=now,
    )
    activation = engine.compute_activation(fact_node, ctx)
    category_report = engine.category_report(fact_node)

    return {
        'enriched': True,
        'node_uuid': entity_node.uuid,
        'name': entity_node.name,
        'primary_category': attrs.get(ATTR_PRIMARY_CATEGORY),
        'weights_top3': sorted(
            [(k, v) for k, v in weights.items() if v > 0],
            key=lambda x: -x[1]
        )[:3],
        'current_activation': round(activation, 4),
        'hours_since_update': round(absolute_hours, 1),
        'access_count': attrs.get(ATTR_ACCESS_COUNT, 0),
        'emotional_loading': attrs.get(ATTR_EMOTIONAL_LOADING, False),
        'has_future_anchor': attrs.get(ATTR_FUTURE_ANCHOR_TS) is not None,
        'confidence': attrs.get(ATTR_CONFIDENCE, 'unknown'),
        'category_report': category_report,
    }


# ============================================================================
# Tests
# ============================================================================

def _run_bridge_tests():
    """Verify the bridge works with mock EntityNode-like objects."""
    import json as _json

    class MockEntityNode:
        """Mimics Graphiti EntityNode for testing."""

        def __init__(self, uuid, name):
            self.uuid = uuid
            self.name = name
            self.attributes = {}

    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            failed += 1
            print(f"  ❌ {name}")

    # Test 1: Enrichment stamps attributes correctly
    print("\n[Test 1] Enrich entity node")
    node = MockEntityNode("uuid-1", "Mom's diabetes")
    classification = {
        'primary_category': 'HEALTH_WELLBEING',
        'weights': {
            'HEALTH_WELLBEING': 0.7,
            'RELATIONAL_BONDS': 0.2,
            'IDENTITY_SELF_CONCEPT': 0.1,
        },
        'emotional_loading': {'detected': True, 'type': 'concern', 'intensity': 'high'},
        'confidence': 'high',
    }
    enrich_entity_node(node, classification)
    check("Node is enriched", is_enriched(node))
    check("Primary category stored", node.attributes[ATTR_PRIMARY_CATEGORY] == 'HEALTH_WELLBEING')
    check("Weights stored as JSON string", isinstance(node.attributes[ATTR_MEMBERSHIP_WEIGHTS], str))
    check("Emotional loading detected", node.attributes[ATTR_EMOTIONAL_LOADING] is True)
    check("Timestamp set", node.attributes[ATTR_LAST_UPDATED_TS] > 0)

    # Test 2: Conversion to FactNode
    print("\n[Test 2] Convert to FactNode")
    fact = entity_to_fact_node(node)
    check("FactNode created", fact is not None)
    check("Primary category matches", fact.primary_category == 'HEALTH_WELLBEING')
    check("Weights parsed correctly", fact.membership_weights['HEALTH_WELLBEING'] == 0.7)
    check("Emotional loading carried", fact.emotional_loading is True)
    check("All 11 categories present", len(fact.membership_weights) == 11)

    # Test 3: Activation computation through bridge
    print("\n[Test 3] Compute activation through bridge")
    engine = DecayEngine.default()
    ctx = TemporalContext(
        absolute_hours=24.0,
        relative_hours=24.0,
        conversational_messages=0,
        current_timestamp=fact.last_updated_ts + 24 * 3600,
    )
    activation = engine.compute_activation(fact, ctx)
    check(f"Activation is positive: {activation:.4f}", activation > 0)
    check(f"Activation < 1.0 (decayed): {activation:.4f}", activation < 1.0)

    # Test 4: Unenriched node returns None
    print("\n[Test 4] Unenriched node handling")
    plain_node = MockEntityNode("uuid-2", "Some entity")
    check("Not enriched", not is_enriched(plain_node))
    check("Conversion returns None", entity_to_fact_node(plain_node) is None)

    # Test 5: Future anchor
    print("\n[Test 5] Future anchor")
    deadline_node = MockEntityNode("uuid-3", "Project deadline March 15")
    enrich_entity_node(deadline_node, {
        'primary_category': 'OBLIGATIONS',
        'weights': {'OBLIGATIONS': 0.8, 'PROJECTS_ENDEAVORS': 0.2},
        'emotional_loading': {'detected': False},
        'confidence': 'high',
    })
    set_future_anchor(deadline_node, datetime(2026, 3, 15, tzinfo=timezone.utc))
    fact_d = entity_to_fact_node(deadline_node)
    check("Future anchor set", fact_d.future_anchor_ts is not None)
    check("Future anchor is March 15", fact_d.future_anchor_ts > 0)

    # Test 6: Inspect node
    print("\n[Test 6] Inspect node diagnostics")
    report = inspect_node(node)
    check("Report shows enriched", report['enriched'] is True)
    check("Report has activation", 'current_activation' in report)
    check("Report has category breakdown", 'category_report' in report)

    # Test 7: Record access
    print("\n[Test 7] Record access")
    record_access(node)
    check("Access count incremented", node.attributes[ATTR_ACCESS_COUNT] == 1)
    record_access(node)
    check("Access count = 2", node.attributes[ATTR_ACCESS_COUNT] == 2)

    print(f"\n{'=' * 60}")
    print(f"Bridge tests: {passed}/{passed + failed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == '__main__':
    _run_bridge_tests()