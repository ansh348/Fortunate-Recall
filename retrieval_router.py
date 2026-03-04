"""
retrieval_router.py — Category-aware retrieval routing orchestrator.

Orchestrates: classify query → category-filtered retrieval → edge-cache
population → Candidate tuple conversion.

Plugs into evaluate_v4.build_candidate_pool() as Strategy 5.
"""

from dataclasses import dataclass

import numpy as np

from query_classifier import classify_query, clear_classifier_cache, QueryClassification
from category_retrieval import fetch_category_edges, ensure_in_edge_cache

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RoutingConfig:
    """Configuration for category-aware retrieval routing."""
    enable_routing: bool = True           # master toggle for A/B testing
    per_category_limit: int = 50          # max edges per category from Cypher
    routing_categories_max: int = 5       # top N categories from classifier
    min_classifier_score: float = 0.15    # threshold to include a category
    include_global: bool = True           # global search always runs alongside
    baseline_graphiti_score: float = 0.35 # score assigned to category-routed candidates


DEFAULT_ROUTING_CONFIG = RoutingConfig()


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

async def route_and_retrieve(
    query: str,
    driver,
    group_id: str,
    xai_client,
    edge_cache: dict[str, dict],
    config: RoutingConfig = DEFAULT_ROUTING_CONFIG,
    embedder=None,
) -> list[tuple[str, str, str, float]]:
    """Run category-aware retrieval routing for a query.

    Pipeline:
        1. Classify query → 2-3 behavioral categories
        2. Fetch category-filtered edges from Neo4j
        3. Populate edge_cache for activation computation
        4. Return (uuid, fact, source, graphiti_score) tuples

    Args:
        query: User's question text.
        driver: Neo4j async driver.
        group_id: Graph group ID.
        xai_client: AsyncOpenAI client for XAI Grok.
        edge_cache: Reference to evaluate_v4._edge_cache (mutated in place).
        config: Routing configuration.
        embedder: OpenAIEmbedder for computing query embedding (cosine similarity).

    Returns:
        List of (uuid, fact, "category_routed", score) tuples.
        Empty list if routing is disabled or no categories qualify.
    """
    if not config.enable_routing:
        return []

    # Step 1: Classify query
    classification = await classify_query(query, xai_client)

    # Step 2: Select categories above score threshold
    categories_to_query = [
        m.category
        for m in classification.matches[:config.routing_categories_max]
        if m.score >= config.min_classifier_score
    ]

    if not categories_to_query:
        return []

    # Step 3: Fetch category-filtered edges
    category_candidates = await fetch_category_edges(
        driver=driver,
        group_id=group_id,
        categories=categories_to_query,
        per_category_limit=config.per_category_limit,
    )

    # Step 4: Populate edge cache
    ensure_in_edge_cache(category_candidates, edge_cache)

    # Step 5: Compute query embedding for cosine similarity scoring
    query_vec = None
    if embedder is not None:
        try:
            query_vec = await embedder.create(input_data=[query.replace('\n', ' ')])
        except Exception:
            query_vec = None

    # Step 6: Convert to tuples with real cosine similarity scores
    results = []
    for c in category_candidates:
        if query_vec is not None and c.fact_embedding is not None:
            score = _cosine_sim(query_vec, c.fact_embedding)
        else:
            score = config.baseline_graphiti_score
        results.append((c.uuid, c.fact, "category_routed", score))
    return results


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def reset_routing_state():
    """Clear all routing caches.  Call at the start of each evaluation run."""
    clear_classifier_cache()
