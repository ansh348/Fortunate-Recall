"""
diagnose_retrieval.py — Diagnose WHY E3/E4 edges are not retrieved.

For each of the ~40 E3+E4 failing questions (correct edges exist in Neo4j
but are not retrieved into the candidate pool), this script:

1. Finds the correct edge(s) using answer_keywords from KEYWORD_REGISTRY
2. Computes cosine similarity between question embedding and edge embedding
3. Tests each retrieval strategy independently (graphiti semantic, keyword,
   multi-keyword, entity neighborhood, category routing)
4. Classifies the failure into R1-R6 subcategories

Outputs:
  - retrieval_diagnosis_report.md  (human-readable)
  - retrieval_diagnosis.json       (machine-readable)

Usage: python diagnose_retrieval.py
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

PROJECT_ROOT = Path(__file__).parent
LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
QUESTIONS_PATH = LIFEMEMEVAL_DIR / "lifemembench_questions.json"

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

from neo4j import AsyncGraphDatabase
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
from extraction_audit import KEYWORD_REGISTRY
from query_classifier import classify_query, CATEGORIES

# ---------------------------------------------------------------------------
# Persona mapping (mirrors evaluate_lifemembench.py)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Stopwords + extract_keywords (copied from evaluate_lifemembench.py)
# ---------------------------------------------------------------------------
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
    "know", "tell", "does", "doing", "been", "there", "their", "they", "them",
    "her", "him", "his", "she", "he",
}


def extract_keywords(text: str, min_len: int = 3) -> list[str]:
    """Current production extract_keywords (alpha-only, min_len=3)."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOPWORDS]


# ---------------------------------------------------------------------------
# E3+E4 question set — filter KEYWORD_REGISTRY to questions with answer edges
# ---------------------------------------------------------------------------

def load_e3_e4_questions() -> dict:
    """Load E3+E4 questions: those with non-empty answer_keywords."""
    e3_e4 = {}
    for qid, reg in KEYWORD_REGISTRY.items():
        if reg["answer_keywords"]:  # E3/E4 have correct edges in Neo4j
            e3_e4[qid] = reg
    return e3_e4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CorrectEdge:
    uuid: str
    fact: str
    fact_embedding: list[float] | None
    enriched: bool
    category: str | None
    created_at_ts: float
    matched_keywords: list[str]


@dataclass
class StrategyResult:
    """Result of testing one retrieval strategy."""
    found: bool
    rank: int | None = None  # 1-indexed rank within strategy results
    total_results: int = 0
    details: str = ""


@dataclass
class DiagnosisResult:
    question_id: str
    persona: str
    question_text: str
    correct_answer: str
    attack_vector: str
    correct_edges: list[CorrectEdge]
    cosine_sim: float | None
    strategies: dict[str, StrategyResult]
    classification: str  # R1-R6 or E2
    explanation: str
    predicted_categories: list[str]
    edge_category: str | None


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_sim(a, b) -> float:
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(dot / norm)


# ---------------------------------------------------------------------------
# Neo4j: find correct edges
# ---------------------------------------------------------------------------

async def find_correct_edges(
    driver, group_id: str, answer_keywords: list[str],
) -> list[CorrectEdge]:
    """Find edges matching answer_keywords in a persona's graph."""
    edges = []
    for kw in answer_keywords:
        try:
            result = await driver.execute_query(
                """
                MATCH (s)-[e:RELATES_TO]->(t)
                WHERE e.group_id = $gid
                  AND toLower(e.fact) CONTAINS toLower($kw)
                RETURN e.uuid AS uuid, e.fact AS fact,
                       e.fact_embedding AS fact_embedding,
                       e.fr_enriched AS enriched,
                       e.fr_primary_category AS category,
                       e.created_at AS created_at
                """,
                gid=group_id,
                kw=kw,
            )
            records = result.records if hasattr(result, "records") else result
            for rec in records:
                d = rec.data() if hasattr(rec, "data") else dict(rec)
                uuid = d["uuid"]
                # Avoid duplicates — add keyword to existing
                existing = next((e for e in edges if e.uuid == uuid), None)
                if existing is not None:
                    if kw not in existing.matched_keywords:
                        existing.matched_keywords.append(kw)
                    continue
                raw_emb = d.get("fact_embedding")
                fact_embedding = list(raw_emb) if raw_emb is not None else None
                created_at = d.get("created_at")
                ts = 0.0
                if created_at is not None:
                    if hasattr(created_at, "to_native"):
                        ts = created_at.to_native().timestamp()
                    elif hasattr(created_at, "timestamp"):
                        ts = created_at.timestamp()
                edges.append(CorrectEdge(
                    uuid=uuid,
                    fact=d.get("fact", ""),
                    fact_embedding=fact_embedding,
                    enriched=bool(d.get("enriched")),
                    category=d.get("category"),
                    created_at_ts=ts,
                    matched_keywords=[kw],
                ))
        except Exception as e:
            print(f"    WARNING: Neo4j query failed for kw='{kw}': {e}")

    return edges


# ---------------------------------------------------------------------------
# Strategy testers
# ---------------------------------------------------------------------------

async def test_graphiti_search(
    graphiti: Graphiti,
    question: str,
    group_id: str,
    correct_uuids: set[str],
    num_results: int = 50,
) -> StrategyResult:
    """Test Graphiti semantic search with given pool size.

    Uses search_() with a custom config to avoid mutating the global
    EDGE_HYBRID_SEARCH_RRF config object.
    """
    try:
        config = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                reranker=EdgeReranker.rrf,
                sim_min_score=0.0,  # no filtering — want to see everything
            ),
            limit=num_results,
        )
        results = await graphiti.search_(
            question, config=config, group_ids=[group_id],
        )
        result_uuids = [str(e.uuid) for e in results.edges]
        total = len(result_uuids)

        for rank, uuid in enumerate(result_uuids):
            if uuid in correct_uuids:
                return StrategyResult(
                    found=True, rank=rank + 1, total_results=total,
                    details=f"Found at rank {rank + 1}/{total}",
                )
        return StrategyResult(
            found=False, total_results=total,
            details=f"Not found in top-{num_results} ({total} returned)",
        )
    except Exception as e:
        return StrategyResult(found=False, details=f"ERROR: {e}")


async def test_cypher_keyword(
    driver, group_id: str, question: str, correct_uuids: set[str],
) -> StrategyResult:
    """Test Cypher keyword search (Strategy 2 from build_candidate_pool)."""
    keywords = extract_keywords(question)
    found_uuids = set()
    for kw in keywords[:8]:
        try:
            result = await driver.execute_query(
                """
                MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
                WHERE e.group_id = $group_id
                  AND toLower(e.fact) CONTAINS $keyword
                RETURN e.uuid AS uuid
                LIMIT 20
                """,
                group_id=group_id,
                keyword=kw,
            )
            records = result.records if hasattr(result, "records") else result
            for rec in records:
                d = rec.data() if hasattr(rec, "data") else dict(rec)
                found_uuids.add(d["uuid"])
        except Exception:
            pass

    matched = correct_uuids & found_uuids
    if matched:
        return StrategyResult(
            found=True, total_results=len(found_uuids),
            details=f"Found {len(matched)} correct edge(s) via keywords: {keywords[:8]}",
        )
    return StrategyResult(
        found=False, total_results=len(found_uuids),
        details=f"Keywords {keywords[:8]} did not match correct edges",
    )


async def test_cypher_intersect(
    driver, group_id: str, question: str, correct_uuids: set[str],
) -> StrategyResult:
    """Test multi-keyword intersection (Strategy 3)."""
    keywords = extract_keywords(question)
    if len(keywords) < 2:
        return StrategyResult(found=False, details="< 2 keywords extracted")

    found_uuids = set()
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
                    RETURN e.uuid AS uuid
                    LIMIT 10
                    """,
                    group_id=group_id,
                    kw1=kw1,
                    kw2=kw2,
                )
                records = result.records if hasattr(result, "records") else result
                for rec in records:
                    d = rec.data() if hasattr(rec, "data") else dict(rec)
                    found_uuids.add(d["uuid"])
            except Exception:
                pass

    matched = correct_uuids & found_uuids
    if matched:
        return StrategyResult(
            found=True, total_results=len(found_uuids),
            details=f"Found {len(matched)} correct edge(s) via keyword pairs",
        )
    return StrategyResult(
        found=False, total_results=len(found_uuids),
        details="No keyword pair matched correct edges",
    )


async def test_entity_neighborhood(
    driver, group_id: str, question: str, correct_uuids: set[str],
) -> StrategyResult:
    """Test entity name neighborhood search (Strategy 4)."""
    keywords = extract_keywords(question)
    found_uuids = set()
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
                RETURN e.uuid AS uuid
                LIMIT 30
                """,
                group_id=group_id,
                keyword=kw,
            )
            records = result.records if hasattr(result, "records") else result
            for rec in records:
                d = rec.data() if hasattr(rec, "data") else dict(rec)
                found_uuids.add(d["uuid"])
        except Exception:
            pass

    matched = correct_uuids & found_uuids
    if matched:
        return StrategyResult(
            found=True, total_results=len(found_uuids),
            details=f"Found {len(matched)} correct edge(s) via entity neighborhood",
        )
    return StrategyResult(
        found=False, total_results=len(found_uuids),
        details="Entity neighborhood did not contain correct edges",
    )


async def test_category_routing(
    driver, group_id: str, question: str,
    correct_edges: list[CorrectEdge],
    xai_client: AsyncOpenAI,
) -> tuple[StrategyResult, list[str]]:
    """Test category routing (Strategy 5).

    Returns (result, predicted_categories).
    """
    try:
        classification = await classify_query(question, xai_client)
    except Exception as e:
        return StrategyResult(found=False, details=f"Classifier error: {e}"), []

    predicted_cats = [m.category for m in classification.matches if m.score >= 0.3]

    # Check if any correct edge's category is in predicted categories
    edge_cats = {e.category for e in correct_edges if e.category}
    cat_match = edge_cats & set(predicted_cats)

    if not edge_cats:
        return StrategyResult(
            found=False,
            details=f"No correct edges have categories. Predicted: {predicted_cats}",
        ), predicted_cats

    if not cat_match:
        return StrategyResult(
            found=False,
            details=f"Edge categories {edge_cats} not in predicted {predicted_cats}",
        ), predicted_cats

    # Category matches — check if edge is in the top-10 most recent per category
    found_uuids = set()
    for cat in cat_match:
        try:
            result = await driver.execute_query(
                """
                MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
                WHERE e.group_id = $group_id
                  AND e.fr_enriched = true
                  AND e.fr_primary_category = $category
                  AND e.expired_at IS NULL
                  AND (e.fr_is_world_knowledge IS NULL OR e.fr_is_world_knowledge = false)
                RETURN e.uuid AS uuid
                ORDER BY e.created_at DESC
                LIMIT 10
                """,
                group_id=group_id,
                category=cat,
            )
            records = result.records if hasattr(result, "records") else result
            for rec in records:
                d = rec.data() if hasattr(rec, "data") else dict(rec)
                found_uuids.add(d["uuid"])
        except Exception:
            pass

    correct_uuids = {e.uuid for e in correct_edges}
    matched = correct_uuids & found_uuids
    if matched:
        return StrategyResult(
            found=True, total_results=len(found_uuids),
            details=f"Category match {cat_match}, edge in top-10 most recent",
        ), predicted_cats

    return StrategyResult(
        found=False, total_results=len(found_uuids),
        details=f"Category matches {cat_match} but edge not in top-10 most recent",
    ), predicted_cats


# ---------------------------------------------------------------------------
# Classification waterfall (R1-R6)
# ---------------------------------------------------------------------------

def classify_failure(
    cosine_sim_val: float | None,
    strategies: dict[str, StrategyResult],
    correct_edges: list[CorrectEdge],
    predicted_categories: list[str],
) -> tuple[str, str]:
    """Classify a retrieval failure into R1-R6.

    R1: POOL_TOO_SMALL — in top-200 but not top-50, or found but ranked low
    R2: SEMANTIC_MISMATCH — cosine similarity < 0.6
    R3: KEYWORD_MISS — no keyword from extract_keywords matches edge fact
    R4: NOT_ENRICHED — edge has fr_enriched=false
    R5: WRONG_CATEGORY — edge category not in classifier's predicted categories
    R6: ROUTING_EXCLUSION — category matches but edge not in top-10 most recent
    """
    s1_50 = strategies.get("graphiti_top50", StrategyResult(found=False))
    s1_200 = strategies.get("graphiti_top200", StrategyResult(found=False))
    s2 = strategies.get("cypher_kw", StrategyResult(found=False))
    s5 = strategies.get("category_routing", StrategyResult(found=False))

    any_found = any(s.found for s in strategies.values())

    # R4: NOT_ENRICHED
    if correct_edges and not any(e.enriched for e in correct_edges):
        return "R4", "Edge has fr_enriched=false, invisible to category routing and edge cache"

    # R1: POOL_TOO_SMALL — in 200 but not 50
    if s1_200.found and not s1_50.found:
        rank = s1_200.rank or "?"
        return "R1", f"In graphiti top-200 (rank {rank}) but not top-50"

    # R1: found in top-50 but at a high rank (likely reranked out of top-5)
    if s1_50.found and s1_50.rank and s1_50.rank > 5:
        return "R1", f"In graphiti top-50 at rank {s1_50.rank} — likely reranked out of top-5"

    # R5: WRONG_CATEGORY
    edge_cats = {e.category for e in correct_edges if e.category and e.enriched}
    if edge_cats and predicted_categories:
        if not edge_cats & set(predicted_categories):
            return "R5", f"Edge category {edge_cats} not in predicted {predicted_categories}"

    # R6: ROUTING_EXCLUSION — category matches but not in top-10
    if not s5.found and edge_cats and set(predicted_categories) & edge_cats:
        return "R6", "Category matches but edge not in top-10 most recent per category"

    # R2: SEMANTIC_MISMATCH — low cosine similarity
    if cosine_sim_val is not None and cosine_sim_val < 0.6 and not s1_50.found:
        return "R2", f"Cosine similarity {cosine_sim_val:.3f} < 0.6 threshold"

    # R3: KEYWORD_MISS
    if not s2.found and not s1_50.found:
        return "R3", "No keyword from extract_keywords() matches edge fact, and not in semantic top-50"

    # Found by some strategy but still failing — ranking issue
    if any_found:
        found_strats = [name for name, s in strategies.items() if s.found]
        return "R1", f"Found by {found_strats} but likely ranked too low after reranking"

    # Default
    sim_str = f"{cosine_sim_val:.3f}" if cosine_sim_val is not None else "N/A"
    return "R2", f"Not found by any strategy. Cosine sim: {sim_str}"


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


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[DiagnosisResult], timestamp: str) -> str:
    lines = []
    lines.append("# Retrieval Diagnosis Report\n")
    lines.append(f"*Generated: {timestamp}*\n")

    # Executive summary
    counts = Counter(r.classification for r in results)
    total = len(results)

    lines.append("## Executive Summary\n")
    lines.append(f"- **Total E3+E4 questions diagnosed:** {total}")
    for code in ["R1", "R2", "R3", "R4", "R5", "R6", "E2"]:
        c = counts.get(code, 0)
        if c == 0 and code == "E2":
            continue
        pct = (c / total * 100) if total else 0
        labels = {
            "R1": "POOL_TOO_SMALL",
            "R2": "SEMANTIC_MISMATCH",
            "R3": "KEYWORD_MISS",
            "R4": "NOT_ENRICHED",
            "R5": "WRONG_CATEGORY",
            "R6": "ROUTING_EXCLUSION",
            "E2": "EXTRACTION_MISS (no edges found)",
        }
        lines.append(f"- **{code} ({labels.get(code, code)}):** {c} ({pct:.0f}%)")
    lines.append("")

    # Fix recommendations
    r1 = counts.get("R1", 0)
    r2 = counts.get("R2", 0)
    r3 = counts.get("R3", 0)
    r4 = counts.get("R4", 0)
    r5 = counts.get("R5", 0)
    r6 = counts.get("R6", 0)

    lines.append("## Fix Recommendations\n")
    if r1 + r2 > 0:
        pct = (r1 + r2) / total * 100
        lines.append(f"### Fix 2.1: Expand search pool + lower cosine threshold (targets R1+R2)")
        lines.append(f"- Expected to address: **{r1 + r2}** questions ({pct:.0f}%)")
        lines.append(f"- Change: `search_()` with limit=100, sim_min_score=0.45\n")

    if r3 > 0:
        pct = r3 / total * 100
        lines.append(f"### Fix 2.2: Improve extract_keywords() (targets R3)")
        lines.append(f"- Expected to address: **{r3}** questions ({pct:.0f}%)")
        lines.append(f"- Change: min_len=2, include numbers, hyphenated compounds\n")

    if r5 + r6 > 0:
        pct = (r5 + r6) / total * 100
        lines.append(f"### Fix 2.3: Cross-category fallback (targets R5+R6)")
        lines.append(f"- Expected to address: **{r5 + r6}** questions ({pct:.0f}%)")
        lines.append(f"- Change: expand to all categories when category results < 5\n")

    # Per-question detail table
    lines.append("## Per-Question Details\n")
    lines.append("| # | question_id | cosine_sim | S1(50) | S1(200) | S2_kw | S3_int | S4_ent | S5_cat | class | explanation |")
    lines.append("|---|------------|-----------|--------|---------|-------|--------|--------|--------|-------|-------------|")

    for i, r in enumerate(results, 1):
        sim_str = f"{r.cosine_sim:.3f}" if r.cosine_sim is not None else "N/A"
        s = r.strategies

        def _mark(key):
            sr = s.get(key, StrategyResult(found=False))
            if sr.found:
                return f"#{sr.rank}" if sr.rank else "Y"
            return "-"

        expl = r.explanation[:80].replace("|", "\\|")
        lines.append(
            f"| {i} | {r.question_id} | {sim_str} | "
            f"{_mark('graphiti_top50')} | {_mark('graphiti_top200')} | "
            f"{_mark('cypher_kw')} | {_mark('cypher_intersect')} | "
            f"{_mark('entity_neighborhood')} | {_mark('category_routing')} | "
            f"**{r.classification}** | {expl} |"
        )

    lines.append("")

    # Dominant failure mode analysis
    lines.append("## Dominant Failure Mode Analysis\n")
    most_common = counts.most_common()
    if most_common:
        top_code, top_count = most_common[0]
        top_pct = top_count / total * 100
        lines.append(f"The dominant failure mode is **{top_code}** ({top_count} questions, {top_pct:.0f}%).\n")

    # Strategy hit rates
    lines.append("## Strategy Hit Rates\n")
    strategy_names = ["graphiti_top50", "graphiti_top200", "cypher_kw",
                      "cypher_intersect", "entity_neighborhood", "category_routing"]
    for sname in strategy_names:
        hits = sum(1 for r in results
                   if r.strategies.get(sname, StrategyResult(found=False)).found)
        pct = (hits / total * 100) if total else 0
        lines.append(f"- **{sname}:** {hits}/{total} ({pct:.0f}%)")
    lines.append("")

    # Detailed per-question analysis
    lines.append("## Detailed Per-Question Analysis\n")
    for r in results:
        lines.append(f"### {r.question_id} — **{r.classification}**\n")
        lines.append(f"- **Question:** {r.question_text}")
        lines.append(f"- **Correct answer:** {r.correct_answer}")
        lines.append(f"- **Attack vector:** {r.attack_vector}")
        sim_str = f"{r.cosine_sim:.3f}" if r.cosine_sim is not None else "N/A"
        lines.append(f"- **Cosine similarity:** {sim_str}")
        lines.append(f"- **Correct edges found:** {len(r.correct_edges)}")
        for ce in r.correct_edges[:3]:
            enriched_str = "enriched" if ce.enriched else "NOT enriched"
            lines.append(f"  - `{ce.uuid[:12]}` [{enriched_str}] cat={ce.category} — \"{ce.fact[:100]}\"")
        lines.append(f"- **Predicted categories:** {r.predicted_categories}")
        lines.append(f"- **Edge category:** {r.edge_category}")
        lines.append(f"- **Classification:** {r.classification} — {r.explanation}")
        lines.append(f"- **Strategy results:**")
        for sname, sr in r.strategies.items():
            status = "FOUND" if sr.found else "MISS"
            lines.append(f"  - {sname}: {status} — {sr.details}")
        lines.append("")

    return "\n".join(lines)


def results_to_json(results: list[DiagnosisResult]) -> list[dict]:
    out = []
    for r in results:
        out.append({
            "question_id": r.question_id,
            "persona": r.persona,
            "question_text": r.question_text,
            "correct_answer": r.correct_answer,
            "attack_vector": r.attack_vector,
            "cosine_sim": r.cosine_sim,
            "classification": r.classification,
            "explanation": r.explanation,
            "predicted_categories": r.predicted_categories,
            "edge_category": r.edge_category,
            "correct_edges": [
                {
                    "uuid": e.uuid,
                    "fact": e.fact,
                    "enriched": e.enriched,
                    "category": e.category,
                    "matched_keywords": e.matched_keywords,
                }
                for e in r.correct_edges
            ],
            "strategies": {
                name: {
                    "found": sr.found,
                    "rank": sr.rank,
                    "total_results": sr.total_results,
                    "details": sr.details,
                }
                for name, sr in r.strategies.items()
            },
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("RETRIEVAL DIAGNOSIS — E3+E4 Failure Analysis")
    print("=" * 60)

    # Validate env
    for var in ["OPENAI_API_KEY", "XAI_API_KEY"]:
        if not os.environ.get(var):
            print(f"ERROR: {var} not set.")
            sys.exit(1)

    # Load questions
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    question_map = {q["id"]: q for q in questions}

    # Load E3+E4 question set
    e3_e4 = load_e3_e4_questions()
    print(f"\nE3+E4 questions to diagnose: {len(e3_e4)}")

    # Setup clients
    graphiti = get_graphiti_client()
    driver = graphiti.driver
    embedder = graphiti.embedder
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    t_start = time_module.time()
    all_results: list[DiagnosisResult] = []

    try:
        for qi, (qid, reg) in enumerate(sorted(e3_e4.items())):
            q = question_map.get(qid)
            if not q:
                print(f"  WARNING: {qid} not in questions JSON, skipping")
                continue

            persona = qid.rsplit("_q", 1)[0]
            group_id = PERSONAS.get(persona, "")
            question_text = q["question"]
            correct_answer = q["correct_answer"]
            attack_vector = q["attack_vector"]

            print(f"\n  [{qi+1}/{len(e3_e4)}] {qid}")
            print(f"    Q: {question_text[:80]}")

            # Step 1: Find correct edges
            correct_edges = await find_correct_edges(
                driver, group_id, reg["answer_keywords"],
            )
            print(f"    Correct edges: {len(correct_edges)}")

            if not correct_edges:
                print(f"    WARNING: No correct edges found — may be E2 not E3/E4")
                all_results.append(DiagnosisResult(
                    question_id=qid, persona=persona,
                    question_text=question_text,
                    correct_answer=correct_answer,
                    attack_vector=attack_vector,
                    correct_edges=[],
                    cosine_sim=None,
                    strategies={},
                    classification="E2",
                    explanation="No correct edges found in Neo4j — true extraction miss",
                    predicted_categories=[],
                    edge_category=None,
                ))
                continue

            correct_uuids = {e.uuid for e in correct_edges}

            # Step 2: Compute cosine similarity
            best_cosine = None
            try:
                query_embedding = await embedder.create(
                    input_data=[question_text.replace("\n", " ")],
                )
                for edge in correct_edges:
                    if edge.fact_embedding:
                        sim = cosine_sim(query_embedding, edge.fact_embedding)
                        if best_cosine is None or sim > best_cosine:
                            best_cosine = sim
            except Exception as e:
                print(f"    WARNING: Embedding failed: {e}")

            sim_str = f"{best_cosine:.3f}" if best_cosine is not None else "N/A"
            print(f"    Cosine sim: {sim_str}")

            # Step 3: Test strategies
            strategies = {}

            # S1: Graphiti semantic (top-50 and top-200)
            s1_50 = await test_graphiti_search(
                graphiti, question_text, group_id, correct_uuids, num_results=50,
            )
            strategies["graphiti_top50"] = s1_50

            s1_200 = await test_graphiti_search(
                graphiti, question_text, group_id, correct_uuids, num_results=200,
            )
            strategies["graphiti_top200"] = s1_200

            # S2: Cypher keyword
            s2 = await test_cypher_keyword(driver, group_id, question_text, correct_uuids)
            strategies["cypher_kw"] = s2

            # S3: Multi-keyword intersection
            s3 = await test_cypher_intersect(driver, group_id, question_text, correct_uuids)
            strategies["cypher_intersect"] = s3

            # S4: Entity neighborhood
            s4 = await test_entity_neighborhood(driver, group_id, question_text, correct_uuids)
            strategies["entity_neighborhood"] = s4

            # S5: Category routing
            s5, predicted_cats = await test_category_routing(
                driver, group_id, question_text, correct_edges, xai_client,
            )
            strategies["category_routing"] = s5

            # Step 4: Classify failure
            classification, explanation = classify_failure(
                best_cosine, strategies, correct_edges, predicted_cats,
            )

            edge_cats = [e.category for e in correct_edges if e.category]
            edge_category = edge_cats[0] if edge_cats else None

            result = DiagnosisResult(
                question_id=qid,
                persona=persona,
                question_text=question_text,
                correct_answer=correct_answer,
                attack_vector=attack_vector,
                correct_edges=correct_edges,
                cosine_sim=best_cosine,
                strategies=strategies,
                classification=classification,
                explanation=explanation,
                predicted_categories=predicted_cats,
                edge_category=edge_category,
            )
            all_results.append(result)

            found_strats = [k for k, v in strategies.items() if v.found]
            print(f"    Classification: {classification} — {explanation[:60]}")
            print(f"    Strategies found: {found_strats or 'NONE'}")

        # Summary
        elapsed = time_module.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"DIAGNOSIS COMPLETE — {len(all_results)} questions in {elapsed:.0f}s")
        print(f"{'=' * 60}")

        counts = Counter(r.classification for r in all_results)
        for code in ["R1", "R2", "R3", "R4", "R5", "R6", "E2"]:
            c = counts.get(code, 0)
            if c > 0:
                pct = (c / len(all_results) * 100)
                print(f"  {code}: {c} ({pct:.0f}%)")

        # Write report
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        report = generate_report(all_results, timestamp)
        report_path = PROJECT_ROOT / "retrieval_diagnosis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport: {report_path}")

        # Write JSON
        json_data = results_to_json(all_results)
        json_path = PROJECT_ROOT / "retrieval_diagnosis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON:   {json_path}")

    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
