"""
calibrate_decay_rates.py — Grid search for optimal per-tier decay rates.

Uses LifeMemBench ground truth (112 questions × 8 personas) to find
decay rates that maximise: mean_activation(correct) - mean_activation(wrong).

Usage: python calibrate_decay_rates.py
"""

import asyncio
import itertools
import json
import os
import re
import sys
import time as time_module
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
QUESTIONS_PATH = PROJECT_ROOT / "LifeMemEval" / "lifemembench_questions.json"
OUTPUT_PATH = PROJECT_ROOT / "calibrated_decay_config.json"

# ===========================================================================
# Section 1: Env loading
# ===========================================================================

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

from neo4j import AsyncGraphDatabase  # noqa: E402

# ===========================================================================
# Section 2: Constants
# ===========================================================================

PERSONAS = {
    "priya":  {"group_id": "lifemembench_priya"},
    "marcus": {"group_id": "lifemembench_marcus"},
    "elena":  {"group_id": "lifemembench_elena"},
    "david":  {"group_id": "lifemembench_david"},
    "amara":  {"group_id": "lifemembench_amara"},
    "jake":   {"group_id": "lifemembench_jake"},
    "tom":    {"group_id": "lifemembench_tom"},
    "omar":   {"group_id": "lifemembench_omar"},
}

TIER_NAMES = ["tier_1_slow", "tier_2_medium", "tier_3_faster", "tier_4_fastest"]

TIER_MAP = {
    'IDENTITY_SELF_CONCEPT':  0,
    'RELATIONAL_BONDS':       0,
    'HEALTH_WELLBEING':       1,
    'INTELLECTUAL_INTERESTS':  1,
    'PROJECTS_ENDEAVORS':     1,
    'PREFERENCES_HABITS':     2,
    'HOBBIES_RECREATION':     2,
    'FINANCIAL_MATERIAL':     2,
    'OTHER':                  2,
    'OBLIGATIONS':            3,
    'LOGISTICAL_CONTEXT':     3,
}

# Current rates (from decay_engine.py DecayConfig.cluster_decay_rates)
CURRENT_RATES = {
    'IDENTITY_SELF_CONCEPT': 0.0015,
    'RELATIONAL_BONDS':      0.0015,
    'INTELLECTUAL_INTERESTS': 0.0020,
    'HEALTH_WELLBEING':      0.0025,
    'PROJECTS_ENDEAVORS':    0.0025,
    'HOBBIES_RECREATION':    0.0035,
    'PREFERENCES_HABITS':    0.0050,
    'FINANCIAL_MATERIAL':    0.0055,
    'OBLIGATIONS':           0.0060,
    'LOGISTICAL_CONTEXT':    0.0080,
    'OTHER':                 0.0050,
}

# Current tier averages (for comparison)
CURRENT_TIER_RATES = [
    np.mean([CURRENT_RATES[c] for c, t in TIER_MAP.items() if t == tier])
    for tier in range(4)
]

# Clock sensitivity (from decay_engine.py DecayConfig.clock_sensitivity)
CLOCK_SENSITIVITY = {
    'OBLIGATIONS':            (0.7, 0.1, 0.2),
    'RELATIONAL_BONDS':       (0.2, 0.6, 0.2),
    'HEALTH_WELLBEING':       (0.5, 0.3, 0.2),
    'IDENTITY_SELF_CONCEPT':  (0.4, 0.3, 0.3),
    'HOBBIES_RECREATION':     (0.3, 0.5, 0.2),
    'PREFERENCES_HABITS':     (0.6, 0.2, 0.2),
    'INTELLECTUAL_INTERESTS': (0.3, 0.4, 0.3),
    'LOGISTICAL_CONTEXT':     (0.8, 0.1, 0.1),
    'PROJECTS_ENDEAVORS':     (0.5, 0.2, 0.3),
    'FINANCIAL_MATERIAL':     (0.5, 0.3, 0.2),
    'OTHER':                  (0.4, 0.3, 0.3),
}

# ===========================================================================
# Section 3: Utilities (from evaluate_lifemembench.py)
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
    "know", "tell", "does", "doing", "been", "there", "their", "they", "them",
    "her", "him", "his", "she", "he",
}


def extract_keywords(text: str, min_len: int = 3) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOPWORDS]


def _to_unix_ts(value) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "to_native"):
        return value.to_native().timestamp()
    if hasattr(value, "timestamp"):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


# ===========================================================================
# Section 4: Data loading (async)
# ===========================================================================

async def load_edge_cache(driver, group_id: str) -> dict[str, dict]:
    result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid AND e.fr_enriched = true
        OPTIONAL MATCH (ep:Episodic)
        WHERE ep.uuid IN e.episodes
        WITH e,
             max(ep.valid_at) AS episode_time
        RETURN e.uuid AS uuid,
               e.fr_primary_category AS primary_category,
               e.fr_membership_weights AS membership_weights,
               e.fr_confidence AS confidence,
               COALESCE(episode_time, e.created_at) AS created_at,
               e.fr_is_world_knowledge AS is_world_knowledge
        """,
        gid=group_id,
    )
    cache = {}
    records = result.records if hasattr(result, "records") else result
    for rec in records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        uuid = d.pop("uuid")
        d["created_at_ts"] = _to_unix_ts(d.get("created_at"))
        cache[uuid] = d
    return cache


async def load_edge_facts(driver, group_id: str) -> dict[str, str]:
    result = await driver.execute_query(
        """
        MATCH (s)-[e:RELATES_TO]->(t)
        WHERE e.group_id = $gid
        RETURN e.uuid AS uuid, e.fact AS fact
        """,
        gid=group_id,
    )
    records = result.records if hasattr(result, "records") else result
    return {
        (r.data() if hasattr(r, "data") else dict(r))["uuid"]:
        (r.data() if hasattr(r, "data") else dict(r))["fact"]
        for r in records
    }


async def get_persona_t_now(driver, group_id: str) -> float:
    result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
        UNWIND e.episodes AS ep_uuid
        MATCH (ep:Episodic {uuid: ep_uuid})
        RETURN max(ep.valid_at) AS max_ts
        """,
        gid=group_id,
    )
    records = result.records if hasattr(result, "records") else result
    if not records:
        return time_module.time()
    d = records[0].data() if hasattr(records[0], "data") else dict(records[0])
    ts = _to_unix_ts(d.get("max_ts"))
    return ts if ts > 0 else time_module.time()


async def load_persona(driver, name: str, group_id: str):
    edge_cache, facts, t_now = await asyncio.gather(
        load_edge_cache(driver, group_id),
        load_edge_facts(driver, group_id),
        get_persona_t_now(driver, group_id),
    )
    return name, edge_cache, facts, t_now


# ===========================================================================
# Section 5: Keyword matching
# ===========================================================================

def match_edges(facts: dict[str, str], keywords: list[str]) -> list[str]:
    """Return UUIDs whose fact text contains any keyword (case-insensitive)."""
    matched = []
    for uuid, fact in facts.items():
        fact_lower = fact.lower()
        if any(kw.lower() in fact_lower for kw in keywords):
            matched.append(uuid)
    return matched


def get_correct_wrong_edges(q: dict, facts: dict[str, str],
                            edge_cache: dict[str, dict]):
    """Identify correct and wrong edges for a question."""
    correct_kws = extract_keywords(q["correct_answer"])
    correct_uuids = match_edges(facts, correct_kws)

    wrong_kws = []
    for indicator in q.get("wrong_answer_indicators", []):
        wrong_kws.append(indicator)
        wrong_kws.extend(extract_keywords(indicator))
    wrong_uuids = match_edges(facts, wrong_kws) if wrong_kws else []

    # Filter out world knowledge and edges missing from enriched cache
    def keep(uuid):
        attrs = edge_cache.get(uuid)
        if attrs is None:
            return False
        if attrs.get("is_world_knowledge"):
            return False
        return True

    correct_uuids = [u for u in correct_uuids if keep(u)]
    wrong_uuids = [u for u in wrong_uuids if keep(u)]
    return correct_uuids, wrong_uuids


# ===========================================================================
# Section 6: Precompute delta-t arrays
# ===========================================================================

def compute_delta_t(category: str, created_at_ts: float, t_now: float) -> float:
    abs_hours = max(0.0, (t_now - created_at_ts) / 3600.0)
    rel_hours = 24.0
    conv_hours = 0.0
    s_abs, s_rel, s_conv = CLOCK_SENSITIVITY.get(category, (0.4, 0.3, 0.3))
    return s_abs * abs_hours + s_rel * rel_hours + s_conv * conv_hours


def build_question_arrays(questions, persona_data):
    """Build precomputed numpy arrays for each question.

    Returns list of (correct_deltas, correct_tiers, wrong_deltas, wrong_tiers, qid)
    for questions that have both correct and wrong edges.
    """
    arrays = []
    skipped = 0

    for q in questions:
        persona = q["persona"]
        if persona not in persona_data:
            skipped += 1
            continue

        edge_cache, facts, t_now = persona_data[persona]
        correct_uuids, wrong_uuids = get_correct_wrong_edges(q, facts, edge_cache)

        if not correct_uuids or not wrong_uuids:
            skipped += 1
            continue

        c_deltas, c_tiers = [], []
        for uuid in correct_uuids:
            attrs = edge_cache[uuid]
            cat = attrs.get("primary_category", "OTHER")
            c_deltas.append(compute_delta_t(cat, attrs.get("created_at_ts", 0.0) or 0.0, t_now))
            c_tiers.append(TIER_MAP.get(cat, 2))

        w_deltas, w_tiers = [], []
        for uuid in wrong_uuids:
            attrs = edge_cache[uuid]
            cat = attrs.get("primary_category", "OTHER")
            w_deltas.append(compute_delta_t(cat, attrs.get("created_at_ts", 0.0) or 0.0, t_now))
            w_tiers.append(TIER_MAP.get(cat, 2))

        arrays.append((
            np.array(c_deltas, dtype=np.float64),
            np.array(c_tiers, dtype=np.int32),
            np.array(w_deltas, dtype=np.float64),
            np.array(w_tiers, dtype=np.int32),
            q["id"],
        ))

    return arrays, skipped


# ===========================================================================
# Section 7: Grid search
# ===========================================================================

GRID = np.logspace(np.log10(0.00001), np.log10(0.005), 15)


def compute_gap(lambdas_4: np.ndarray, question_arrays) -> float:
    total_gap = 0.0
    for c_deltas, c_tiers, w_deltas, w_tiers, _ in question_arrays:
        c_act = np.exp(-lambdas_4[c_tiers] * c_deltas)
        w_act = np.exp(-lambdas_4[w_tiers] * w_deltas)
        total_gap += c_act.mean() - w_act.mean()
    return total_gap


def run_grid_search(question_arrays):
    best_gap = -float("inf")
    best_lambdas = None
    total = len(GRID) ** 4
    count = 0
    t0 = time_module.time()

    for lam1, lam2, lam3, lam4 in itertools.product(GRID, repeat=4):
        lambdas = np.array([lam1, lam2, lam3, lam4])
        gap = compute_gap(lambdas, question_arrays)

        if gap > best_gap:
            best_gap = gap
            best_lambdas = (lam1, lam2, lam3, lam4)

        count += 1
        if count % 5000 == 0:
            elapsed = time_module.time() - t0
            print(f"  [{count:>7}/{total}]  best_gap={best_gap:.4f}  "
                  f"lam=({best_lambdas[0]:.6f}, {best_lambdas[1]:.6f}, "
                  f"{best_lambdas[2]:.6f}, {best_lambdas[3]:.6f})  "
                  f"({elapsed:.1f}s)")

    elapsed = time_module.time() - t0
    print(f"  Done in {elapsed:.1f}s ({total} combinations)")
    return best_lambdas, best_gap


# ===========================================================================
# Section 8: Output
# ===========================================================================

def build_per_category_rates(tier_lambdas):
    return {cat: tier_lambdas[tier] for cat, tier in TIER_MAP.items()}


def print_results(best_lambdas, best_gap, current_gap, questions_used, skipped):
    print()
    print("=" * 65)
    print("CALIBRATION RESULTS")
    print("=" * 65)
    print(f"  Questions used: {questions_used} (skipped {skipped})")
    print(f"  Optimal gap:    {best_gap:.4f}")
    print(f"  Current gap:    {current_gap:.4f}")
    delta = best_gap - current_gap
    print(f"  Improvement:    {delta:+.4f} ({delta / abs(current_gap) * 100:+.1f}%)" if current_gap != 0 else f"  Improvement:    {delta:+.4f}")
    print()
    print(f"  {'Tier':<22} {'Current':>10} {'Calibrated':>12} {'Change':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*10}")
    for tier_idx, tier_name in enumerate(TIER_NAMES):
        cur = CURRENT_TIER_RATES[tier_idx]
        cal = best_lambdas[tier_idx]
        pct = (cal - cur) / cur * 100 if cur != 0 else 0
        print(f"  {tier_name:<22} {cur:>10.6f} {cal:>12.6f} {pct:>+9.1f}%")

    print()
    print("  Per-category mapping:")
    per_cat = build_per_category_rates(best_lambdas)
    for cat in sorted(TIER_MAP.keys(), key=lambda c: (TIER_MAP[c], c)):
        cur = CURRENT_RATES[cat]
        cal = per_cat[cat]
        print(f"    {cat:<26} {cur:.6f} -> {cal:.6f}")


def save_config(best_lambdas, best_gap, current_gap, questions_used):
    per_cat = build_per_category_rates(best_lambdas)
    config = {
        "tier_lambdas": {name: float(best_lambdas[i]) for i, name in enumerate(TIER_NAMES)},
        "per_category_rates": {cat: float(rate) for cat, rate in per_cat.items()},
        "current_rates": CURRENT_RATES,
        "optimal_gap": float(best_gap),
        "current_gap": float(current_gap),
        "questions_used": questions_used,
        "grid_size": len(GRID),
        "grid_range": [float(GRID[0]), float(GRID[-1])],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {OUTPUT_PATH}")


# ===========================================================================
# Main
# ===========================================================================

async def main():
    print("=== DECAY RATE CALIBRATION ===")

    # Load questions
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    print(f"Loaded {len(questions)} questions")

    # Connect to Neo4j
    driver = AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "testpassword123"),
        ),
    )

    # Load all persona data in parallel
    print("Loading data for 8 personas...")
    tasks = [
        load_persona(driver, name, p["group_id"])
        for name, p in PERSONAS.items()
    ]
    results = await asyncio.gather(*tasks)

    persona_data = {}
    for name, edge_cache, facts, t_now in results:
        enriched = len(edge_cache)
        total = len(facts)
        print(f"  {name}: {total} edges ({enriched} enriched), "
              f"t_now={time_module.strftime('%Y-%m-%d', time_module.gmtime(t_now))}")
        persona_data[name] = (edge_cache, facts, t_now)

    await driver.close()

    # Build question arrays
    print("\nMatching keywords...")
    question_arrays, skipped = build_question_arrays(questions, persona_data)
    questions_used = len(question_arrays)
    print(f"  {questions_used} questions with both correct and wrong edges")
    print(f"  {skipped} questions skipped (missing correct or wrong edges)")

    # Compute current gap
    current_lambdas = np.array([CURRENT_TIER_RATES[i] for i in range(4)])
    current_gap = compute_gap(current_lambdas, question_arrays)
    print(f"  Current gap: {current_gap:.4f}")

    # Grid search
    print(f"\nRunning grid search: {len(GRID)**4} combinations...")
    best_lambdas, best_gap = run_grid_search(question_arrays)

    # Output
    print_results(best_lambdas, best_gap, current_gap, questions_used, skipped)
    save_config(best_lambdas, best_gap, current_gap, questions_used)


if __name__ == "__main__":
    asyncio.run(main())
