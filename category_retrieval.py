"""
category_retrieval.py — Category-filtered edge retrieval from Neo4j.

Fetches edges whose fr_primary_category matches the query classifier's
output.  Returns CategoryCandidate objects that the router converts to
evaluate_v4.Candidate for pool merging.
"""

import asyncio
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CategoryCandidate:
    """Raw result from category-filtered Cypher query."""
    uuid: str
    fact: str
    primary_category: str
    membership_weights: str   # JSON string as stored in Neo4j
    created_at_ts: float      # Unix timestamp for recency
    confidence: float
    is_world_knowledge: bool
    fact_embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Cypher — one query per category, run concurrently
# ---------------------------------------------------------------------------
CATEGORY_FILTER_CYPHER = """
MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
WHERE e.group_id = $group_id
  AND e.fr_enriched = true
  AND e.fr_primary_category = $category
  AND e.expired_at IS NULL
  AND (e.fr_is_world_knowledge IS NULL OR e.fr_is_world_knowledge = false)
RETURN e.uuid AS uuid,
       e.fact AS fact,
       e.fr_primary_category AS primary_category,
       e.fr_membership_weights AS membership_weights,
       e.created_at AS created_at,
       e.fr_confidence AS confidence,
       e.fr_is_world_knowledge AS is_world_knowledge,
       e.fact_embedding AS fact_embedding
ORDER BY e.created_at DESC
LIMIT $per_category_limit
"""


# ---------------------------------------------------------------------------
# Timestamp conversion (mirrors evaluate_v4._to_unix_ts)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

async def fetch_category_edges(
    driver,
    group_id: str,
    categories: list[str],
    per_category_limit: int = 10,
) -> list[CategoryCandidate]:
    """Fetch edges filtered by fr_primary_category for given categories.

    Runs one Cypher query per category concurrently.  Results are
    deduplicated by UUID across categories.

    Args:
        driver: Neo4j async driver instance.
        group_id: Graph group ID (e.g., "full_234").
        categories: Category names to query.
        per_category_limit: Max edges per category.

    Returns:
        Deduplicated list of CategoryCandidate objects.
    """
    seen_uuids: set[str] = set()
    results: list[CategoryCandidate] = []

    async def _fetch_one(category: str) -> list[dict]:
        try:
            result = await driver.execute_query(
                CATEGORY_FILTER_CYPHER,
                group_id=group_id,
                category=category,
                per_category_limit=per_category_limit,
            )
            records = result.records if hasattr(result, "records") else result
            return [rec.data() if hasattr(rec, "data") else dict(rec) for rec in records]
        except Exception:
            return []

    batches = await asyncio.gather(*[_fetch_one(cat) for cat in categories], return_exceptions=True)

    for batch in batches:
        if isinstance(batch, Exception):
            continue
        for d in batch:
            uuid = d["uuid"]
            if uuid in seen_uuids:
                continue
            seen_uuids.add(uuid)
            raw_emb = d.get("fact_embedding")
            fact_embedding = list(raw_emb) if raw_emb is not None else None
            results.append(CategoryCandidate(
                uuid=uuid,
                fact=d.get("fact", ""),
                primary_category=d.get("primary_category", "OTHER"),
                membership_weights=d.get("membership_weights", "{}"),
                created_at_ts=_to_unix_ts(d.get("created_at")),
                confidence=float(d.get("confidence", 0.0) or 0.0),
                is_world_knowledge=bool(d.get("is_world_knowledge", False)),
                fact_embedding=fact_embedding,
            ))

    return results


# ---------------------------------------------------------------------------
# Edge-cache population helper
# ---------------------------------------------------------------------------

def ensure_in_edge_cache(
    candidates: list[CategoryCandidate],
    edge_cache: dict[str, dict],
):
    """Populate _edge_cache for category-routed edges not already cached.

    Without this, compute_edge_activation_v4() would return the 0.5
    unenriched fallback, making category routing pointless.
    """
    for c in candidates:
        if c.uuid not in edge_cache:
            edge_cache[c.uuid] = {
                "primary_category": c.primary_category,
                "membership_weights": c.membership_weights,
                "confidence": c.confidence,
                "created_at_ts": c.created_at_ts,
                "is_world_knowledge": c.is_world_knowledge,
            }
