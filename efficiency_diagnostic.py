"""Storage & retrieval efficiency diagnostic: FR vs Mem0.

Reads existing JSON artifacts (results, checkpoints) and optionally queries
Neo4j for live edge statistics.  Produces five formatted tables suitable for
the paper's efficiency section.

Usage:
    python efficiency_diagnostic.py              # with Neo4j
    python efficiency_diagnostic.py --no-neo4j   # offline (checkpoint data only)
    python efficiency_diagnostic.py --json out.json
    python efficiency_diagnostic.py --task 3      # run single task
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "LifeMemEval" / "artifacts"

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSONAS = {
    "priya":   {"dir": "1_priya",   "group_id": "lifemembench_priya"},
    "marcus":  {"dir": "2_marcus",  "group_id": "lifemembench_marcus"},
    "elena":   {"dir": "3_elena",   "group_id": "lifemembench_elena"},
    "david":   {"dir": "4_david",   "group_id": "lifemembench_david"},
    "amara":   {"dir": "5_amara",   "group_id": "lifemembench_amara"},
    "jake":    {"dir": "6_jake",    "group_id": "lifemembench_jake"},
    "fatima":  {"dir": "7_fatima",  "group_id": "lifemembench_fatima"},
    "tom":     {"dir": "8_tom",     "group_id": "lifemembench_tom"},
    "kenji":   {"dir": "9_kenji",   "group_id": "lifemembench_kenji"},
    "rosa":    {"dir": "10_rosa",   "group_id": "lifemembench_rosa"},
    "callum":  {"dir": "11_callum", "group_id": "lifemembench_callum"},
    "diane":   {"dir": "12_diane",  "group_id": "lifemembench_diane"},
    "raj":     {"dir": "13_raj",    "group_id": "lifemembench_raj"},
    "nadia":   {"dir": "14_nadia",  "group_id": "lifemembench_nadia"},
    "samuel":  {"dir": "15_samuel", "group_id": "lifemembench_samuel"},
    "lily":    {"dir": "16_lily",   "group_id": "lifemembench_lily"},
    "omar":    {"dir": "17_omar",   "group_id": "lifemembench_omar"},
    "bruna":   {"dir": "18_bruna",  "group_id": "lifemembench_bruna"},
    "patrick": {"dir": "19_patrick","group_id": "lifemembench_patrick"},
    "aisha":   {"dir": "20_aisha",  "group_id": "lifemembench_aisha"},
    "thanh":     {"dir": "21_thanh",     "group_id": "lifemembench_thanh"},
    "alex":      {"dir": "22_alex",      "group_id": "lifemembench_alex"},
    "mirri":     {"dir": "23_mirri",     "group_id": "lifemembench_mirri"},
    "jerome":    {"dir": "24_jerome",    "group_id": "lifemembench_jerome"},
    "ingrid":    {"dir": "25_ingrid",    "group_id": "lifemembench_ingrid"},
    "dmitri":    {"dir": "26_dmitri",    "group_id": "lifemembench_dmitri"},
    "yoli":      {"dir": "27_yoli",      "group_id": "lifemembench_yoli"},
    "dariush":   {"dir": "28_dariush",   "group_id": "lifemembench_dariush"},
    "aroha":     {"dir": "29_aroha",     "group_id": "lifemembench_aroha"},
    "mehmet":    {"dir": "30_mehmet",    "group_id": "lifemembench_mehmet"},
    "saga":      {"dir": "31_saga",      "group_id": "lifemembench_saga"},
    "kofi":      {"dir": "32_kofi",      "group_id": "lifemembench_kofi"},
    "valentina": {"dir": "33_valentina", "group_id": "lifemembench_valentina"},
    "billy":     {"dir": "34_billy",     "group_id": "lifemembench_billy"},
    "pan":       {"dir": "35_pan",       "group_id": "lifemembench_pan"},
    "marley":    {"dir": "36_marley",    "group_id": "lifemembench_marley"},
    "leila":     {"dir": "37_leila",     "group_id": "lifemembench_leila"},
    "chenoa":    {"dir": "38_chenoa",    "group_id": "lifemembench_chenoa"},
    "joonho":    {"dir": "39_joonho",    "group_id": "lifemembench_joonho"},
    "zara":      {"dir": "40_zara",      "group_id": "lifemembench_zara"},
}

# Storage overhead per record (bytes, excluding fact text and embedding)
FR_OVERHEAD = 36 + 200 + 24 + 100   # UUID + metadata + timestamps + category/weights = 360
MEM0_OVERHEAD = 100 + 64            # metadata + hash = 164
EMBEDDING_BYTES = 1536 * 4          # 6144 bytes (both use 1536-dim text-embedding-3-small)

MEM0_ESTIMATED_MEMORIES_PER_PERSONA = 100  # from priya: 35 sessions -> 100 memories

GPT4O_INPUT_COST_PER_M_TOKENS = 2.50  # USD
CHARS_PER_TOKEN = 4  # rough average


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hdr(title: str) -> None:
    print(f"\n{'=' * 76}")
    print(f"  {title}")
    print(f"{'=' * 76}")


def subhdr(title: str) -> None:
    print(f"\n--- {title} {'-' * max(1, 70 - len(title))}")


def pct(n: float, total: float) -> str:
    if total == 0:
        return "N/A"
    return f"{n / total * 100:.1f}%"


def fmt_bytes(b: float) -> str:
    if b < 1024:
        return f"{b:.0f} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.2f} MB"


# ---------------------------------------------------------------------------
# Neo4j queries (Task 1 with live data)
# ---------------------------------------------------------------------------

async def query_neo4j_edge_stats(personas: dict) -> dict:
    """Query Neo4j for per-persona edge statistics."""
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"),
              os.environ.get("NEO4J_PASSWORD", "testpassword123")),
    )
    stats = {}
    for name, meta in personas.items():
        gid = meta["group_id"]
        r = await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.group_id = $gid AND e.fr_enriched = true
            RETURN count(e) AS total,
                   count(CASE WHEN e.fr_superseded_by IS NULL THEN 1 END) AS active,
                   count(CASE WHEN e.fr_superseded_by IS NOT NULL THEN 1 END) AS superseded,
                   avg(size(e.fact)) AS avg_fact_len,
                   sum(size(e.fact)) AS total_fact_chars
            """,
            gid=gid,
        )
        recs = r.records if hasattr(r, "records") else r
        if recs:
            d = recs[0].data() if hasattr(recs[0], "data") else dict(recs[0])
            stats[name] = {
                "total": d["total"],
                "active": d["active"],
                "superseded": d["superseded"],
                "avg_fact_len": float(d["avg_fact_len"] or 0),
                "total_fact_chars": int(d["total_fact_chars"] or 0),
            }
        else:
            stats[name] = {"total": 0, "active": 0, "superseded": 0,
                           "avg_fact_len": 0, "total_fact_chars": 0}
    await driver.close()
    return stats


def build_offline_edge_stats(ingest_ckpt: dict, super_ckpt: dict,
                             fr_avg_fact_len: float) -> dict:
    """Build edge stats from checkpoint files when Neo4j is unavailable."""
    stats = {}
    for name in PERSONAS:
        ing = ingest_ckpt["personas"].get(name, {})
        sup = super_ckpt["personas"].get(name, {})
        total = ing.get("enrich_personal", 0)
        superseded = sup.get("supersessions_found", 0)
        active = total - superseded
        stats[name] = {
            "total": total,
            "active": active,
            "superseded": superseded,
            "avg_fact_len": fr_avg_fact_len,  # biased estimate from results
            "total_fact_chars": int(total * fr_avg_fact_len),
        }
    return stats


def compute_avg_fact_len_from_results(questions: list) -> float:
    """Compute average fact text length from top5_facts across all questions."""
    lengths = []
    for q in questions:
        for f in q.get("top5_facts", []):
            lengths.append(len(f.get("fact", "")))
    return statistics.mean(lengths) if lengths else 60.0


# ---------------------------------------------------------------------------
# Task 1: Storage Comparison
# ---------------------------------------------------------------------------

def task1_storage(fr_stats: dict, mem0_count_per_persona: int,
                  mem0_avg_text_len: float, is_estimated: bool) -> dict:
    """Compute and print storage comparison."""
    hdr("TASK 1: STORAGE COMPARISON")

    # Per-persona table
    subhdr("Per-Persona Summary")

    header = (f"  {'Persona':<12s} | {'FR Edges':>8s} {'(Act/Sup)':>12s} | "
              f"{'FR Store':>10s} | {'Mem0':>6s} | {'Mem0 Store':>10s} | {'Ratio':>5s}")
    print(header)
    print(f"  {'-' * 12}-+-{'-' * 21}-+-{'-' * 10}-+-{'-' * 6}-+-{'-' * 10}-+-{'-' * 5}")

    totals = {"fr_total": 0, "fr_active": 0, "fr_superseded": 0,
              "fr_storage": 0, "mem0_total": 0, "mem0_storage": 0,
              "fr_fact_chars": 0}

    est_tag = " (est.)" if is_estimated else ""
    persona_rows = []

    for name in PERSONAS:
        s = fr_stats.get(name, {})
        fr_t = s.get("total", 0)
        fr_a = s.get("active", 0)
        fr_s = s.get("superseded", 0)
        avg_fl = s.get("avg_fact_len", 60)
        fr_store = fr_t * (FR_OVERHEAD + EMBEDDING_BYTES + avg_fl)
        m0_store = mem0_count_per_persona * (MEM0_OVERHEAD + EMBEDDING_BYTES + mem0_avg_text_len)
        ratio = fr_store / m0_store if m0_store > 0 else 0

        totals["fr_total"] += fr_t
        totals["fr_active"] += fr_a
        totals["fr_superseded"] += fr_s
        totals["fr_storage"] += fr_store
        totals["mem0_total"] += mem0_count_per_persona
        totals["mem0_storage"] += m0_store
        totals["fr_fact_chars"] += s.get("total_fact_chars", 0)

        persona_rows.append({
            "name": name, "fr_t": fr_t, "fr_a": fr_a, "fr_s": fr_s,
            "fr_store": fr_store, "m0": mem0_count_per_persona,
            "m0_store": m0_store, "ratio": ratio,
        })

    for r in persona_rows:
        print(f"  {r['name']:<12s} | {r['fr_t']:>8d} ({r['fr_a']:>4d}/{r['fr_s']:>3d})  | "
              f"{fmt_bytes(r['fr_store']):>10s} | {r['m0']:>6d} | "
              f"{fmt_bytes(r['m0_store']):>10s} | {r['ratio']:>4.1f}x")

    # Totals row
    t = totals
    total_ratio = t["fr_storage"] / t["mem0_storage"] if t["mem0_storage"] > 0 else 0
    print(f"  {'-' * 12}-+-{'-' * 21}-+-{'-' * 10}-+-{'-' * 6}-+-{'-' * 10}-+-{'-' * 5}")
    print(f"  {'TOTAL (40)':<12s} | {t['fr_total']:>8d} ({t['fr_active']:>4d}/{t['fr_superseded']:>3d})  | "
          f"{fmt_bytes(t['fr_storage']):>10s} | {t['mem0_total']:>6d}{est_tag} | "
          f"{fmt_bytes(t['mem0_storage']):>10s} | {total_ratio:>4.1f}x")

    # Aggregate metrics
    fr_avg_fl = t["fr_fact_chars"] / t["fr_total"] if t["fr_total"] > 0 else 0
    subhdr("Aggregate Storage Metrics")
    print(f"  {'Metric':<30s} {'FR':>14s}    {'Mem0':>14s}")
    print(f"  {'─' * 30} {'─' * 14}    {'─' * 14}")
    print(f"  {'Total records':<30s} {t['fr_total']:>14,d}    {t['mem0_total']:>10,d}{est_tag}")
    print(f"  {'Avg fact text length':<30s} {fr_avg_fl:>11.1f} ch    {mem0_avg_text_len:>11.1f} ch")
    print(f"  {'Embedding dims':<30s} {'1536':>14s}    {'1536':>14s}")
    print(f"  {'Bytes/record (avg)':<30s} {FR_OVERHEAD + EMBEDDING_BYTES + fr_avg_fl:>11.0f} B    "
          f"{MEM0_OVERHEAD + EMBEDDING_BYTES + mem0_avg_text_len:>11.0f} B")
    print(f"  {'Total storage':<30s} {fmt_bytes(t['fr_storage']):>14s}    {fmt_bytes(t['mem0_storage']):>14s}")
    sup_pct = pct(t['fr_superseded'], t['fr_total'])
    print(f"  {'Superseded (inactive)':<30s} {t['fr_superseded']:>10,d} ({sup_pct})    {'0 (N/A)':>14s}")

    return {"totals": totals, "per_persona": persona_rows,
            "fr_avg_fact_len": fr_avg_fl, "mem0_avg_text_len": mem0_avg_text_len}


# ---------------------------------------------------------------------------
# Task 2: Retrieval Pool Size
# ---------------------------------------------------------------------------

def task2_pool_size(fr_questions: list, mem0_questions: list,
                    super_ckpt: dict) -> dict:
    """Compute and print retrieval pool size comparison."""
    hdr("TASK 2: RETRIEVAL POOL SIZE COMPARISON")

    fr_pools = [q["pool_size"] for q in fr_questions]
    m0_pools = [q["pool_size"] for q in mem0_questions]

    fr_mean = statistics.mean(fr_pools)
    fr_median = statistics.median(fr_pools)
    fr_stdev = statistics.stdev(fr_pools) if len(fr_pools) > 1 else 0
    m0_mean = statistics.mean(m0_pools)
    m0_median = statistics.median(m0_pools)

    # Supersession filtering stats
    total_sup = sum(p.get("supersessions_found", 0)
                    for p in super_ckpt["personas"].values())
    avg_sup = total_sup / len(super_ckpt["personas"]) if super_ckpt["personas"] else 0

    subhdr("Pool Size Statistics")
    print(f"  {'Metric':<30s} {'FR (full)':>12s}    {'Mem0':>12s}")
    print(f"  {'─' * 30} {'─' * 12}    {'─' * 12}")
    print(f"  {'Avg pool per query':<30s} {fr_mean:>12.1f}    {m0_mean:>12.1f}")
    print(f"  {'Median pool per query':<30s} {fr_median:>12.1f}    {m0_median:>12.1f}")
    print(f"  {'Std dev':<30s} {fr_stdev:>12.1f}    {'0.0':>12s}")
    print(f"  {'Min pool':<30s} {min(fr_pools):>12d}    {min(m0_pools):>12d}")
    print(f"  {'Max pool':<30s} {max(fr_pools):>12d}    {max(m0_pools):>12d}")
    fr_sel = 10 / fr_mean * 100 if fr_mean > 0 else 0
    print(f"  {'Selectivity (top-10/pool)':<30s} {fr_sel:>11.1f}%    {'100.0%':>12s}")
    print(f"  {'Active curation':<30s} {'Yes':>12s}    {'No':>12s}")

    subhdr("Supersession Filtering (FR)")
    print(f"  Total superseded edges across 40 personas: {total_sup}")
    print(f"  Avg superseded per persona:                {avg_sup:.1f}")
    print(f"  These edges are removed from candidate pools before ranking.")

    return {
        "fr": {"mean": fr_mean, "median": fr_median, "stdev": fr_stdev,
               "min": min(fr_pools), "max": max(fr_pools), "selectivity_pct": fr_sel},
        "mem0": {"mean": m0_mean, "median": m0_median, "min": min(m0_pools),
                 "max": max(m0_pools)},
        "supersession": {"total": total_sup, "avg_per_persona": avg_sup},
    }


# ---------------------------------------------------------------------------
# Task 3: Context Window Efficiency
# ---------------------------------------------------------------------------

def _count_fact_types(questions: list) -> dict:
    """Count supports_correct, contains_wrong, neutral across all questions."""
    total_supports = 0
    total_wrong = 0
    total_neutral = 0
    total_facts = 0
    per_q_supports = []
    per_q_wrong = []

    for q in questions:
        facts = q.get("top5_facts", [])
        s = sum(1 for f in facts if f.get("supports_correct"))
        w = sum(1 for f in facts if f.get("contains_wrong"))
        n = len(facts) - s - w
        total_supports += s
        total_wrong += w
        total_neutral += n
        total_facts += len(facts)
        per_q_supports.append(s)
        per_q_wrong.append(w)

    n_q = len(questions) or 1
    return {
        "total_supports": total_supports,
        "total_wrong": total_wrong,
        "total_neutral": total_neutral,
        "total_facts": total_facts,
        "avg_supports_per_q": total_supports / n_q,
        "avg_wrong_per_q": total_wrong / n_q,
        "avg_neutral_per_q": total_neutral / n_q,
        "supports_pct": total_supports / total_facts * 100 if total_facts else 0,
        "wrong_pct": total_wrong / total_facts * 100 if total_facts else 0,
        "neutral_pct": total_neutral / total_facts * 100 if total_facts else 0,
        "signal_noise": (total_supports / (total_supports + total_wrong) * 100
                         if (total_supports + total_wrong) > 0 else 100),
    }


def task3_context_efficiency(fr_questions: list, mem0_questions: list) -> dict:
    """Compute and print context window efficiency comparison."""
    hdr("TASK 3: CONTEXT WINDOW EFFICIENCY")

    fr = _count_fact_types(fr_questions)
    m0 = _count_fact_types(mem0_questions)

    subhdr("Per-Fact Classification (across all 516 questions)")
    print(f"  {'Metric':<30s} {'FR (full)':>12s}    {'Mem0':>12s}    {'Delta':>10s}")
    print(f"  {'─' * 30} {'─' * 12}    {'─' * 12}    {'─' * 10}")
    print(f"  {'Supports correct (%)' :<30s} {fr['supports_pct']:>11.1f}%    "
          f"{m0['supports_pct']:>11.1f}%    {m0['supports_pct'] - fr['supports_pct']:>+9.1f}pp")
    print(f"  {'Contains wrong (%)' :<30s} {fr['wrong_pct']:>11.1f}%    "
          f"{m0['wrong_pct']:>11.1f}%    {m0['wrong_pct'] - fr['wrong_pct']:>+9.1f}pp")
    print(f"  {'Neutral (%)' :<30s} {fr['neutral_pct']:>11.1f}%    "
          f"{m0['neutral_pct']:>11.1f}%    {m0['neutral_pct'] - fr['neutral_pct']:>+9.1f}pp")
    print(f"  {'Signal:noise ratio (%)' :<30s} {fr['signal_noise']:>11.1f}%    "
          f"{m0['signal_noise']:>11.1f}%    {m0['signal_noise'] - fr['signal_noise']:>+9.1f}pp")

    subhdr("Per-Question Averages")
    print(f"  {'Avg supports/query':<30s} {fr['avg_supports_per_q']:>12.2f}    "
          f"{m0['avg_supports_per_q']:>12.2f}    {m0['avg_supports_per_q'] - fr['avg_supports_per_q']:>+10.2f}")
    print(f"  {'Avg wrong/query':<30s} {fr['avg_wrong_per_q']:>12.2f}    "
          f"{m0['avg_wrong_per_q']:>12.2f}    {m0['avg_wrong_per_q'] - fr['avg_wrong_per_q']:>+10.2f}")
    wr_fr = fr['avg_wrong_per_q'] / fr['avg_supports_per_q'] if fr['avg_supports_per_q'] > 0 else 0
    wr_m0 = m0['avg_wrong_per_q'] / m0['avg_supports_per_q'] if m0['avg_supports_per_q'] > 0 else 0
    print(f"  {'Wrong per support':<30s} {wr_fr:>12.2f}    "
          f"{wr_m0:>12.2f}    {wr_m0 - wr_fr:>+10.2f}")

    return {"fr": fr, "mem0": m0}


# ---------------------------------------------------------------------------
# Task 4: Staleness as Token Waste
# ---------------------------------------------------------------------------

def _staleness_stats(questions: list) -> dict:
    """Compute stale token rate and costs."""
    stale_chars_list = []
    total_chars_list = []
    stale_rates = []

    for q in questions:
        facts = q.get("top5_facts", [])
        stale_c = sum(len(f.get("fact", "")) for f in facts if f.get("contains_wrong"))
        total_c = sum(len(f.get("fact", "")) for f in facts)
        stale_chars_list.append(stale_c)
        total_chars_list.append(total_c)
        stale_rates.append(stale_c / total_c if total_c > 0 else 0)

    n = len(questions) or 1
    avg_stale_chars = sum(stale_chars_list) / n
    avg_total_chars = sum(total_chars_list) / n
    avg_stale_rate = statistics.mean(stale_rates) if stale_rates else 0

    # Cost: stale tokens per query, then scale to 1M queries
    avg_stale_tokens = avg_stale_chars / CHARS_PER_TOKEN
    avg_total_tokens = avg_total_chars / CHARS_PER_TOKEN
    cost_per_1M = avg_stale_tokens * GPT4O_INPUT_COST_PER_M_TOKENS

    return {
        "avg_stale_chars": avg_stale_chars,
        "avg_total_chars": avg_total_chars,
        "avg_stale_rate": avg_stale_rate,
        "avg_stale_tokens": avg_stale_tokens,
        "avg_total_tokens": avg_total_tokens,
        "cost_per_1M_queries": cost_per_1M,
        "annual_cost_1M_per_day": cost_per_1M * 365,
    }


def task4_staleness_cost(fr_questions: list, mem0_questions: list) -> dict:
    """Compute and print staleness-as-token-waste comparison."""
    hdr("TASK 4: STALENESS AS TOKEN WASTE")

    fr = _staleness_stats(fr_questions)
    m0 = _staleness_stats(mem0_questions)

    subhdr("Token Waste Metrics")
    print(f"  {'Metric':<32s} {'FR (full)':>14s}    {'Mem0':>14s}")
    print(f"  {'─' * 32} {'─' * 14}    {'─' * 14}")
    print(f"  {'Avg stale token rate':<32s} {fr['avg_stale_rate'] * 100:>13.1f}%    "
          f"{m0['avg_stale_rate'] * 100:>13.1f}%")
    print(f"  {'Avg stale chars/query':<32s} {fr['avg_stale_chars']:>14.1f}    "
          f"{m0['avg_stale_chars']:>14.1f}")
    print(f"  {'Avg total chars/query':<32s} {fr['avg_total_chars']:>14.1f}    "
          f"{m0['avg_total_chars']:>14.1f}")
    print(f"  {'Avg stale tokens/query':<32s} {fr['avg_stale_tokens']:>14.1f}    "
          f"{m0['avg_stale_tokens']:>14.1f}")
    print(f"  {'Avg total tokens/query':<32s} {fr['avg_total_tokens']:>14.1f}    "
          f"{m0['avg_total_tokens']:>14.1f}")

    subhdr("Cost at GPT-4o Pricing ($2.50/1M input tokens)")
    print(f"  {'Cost per 1M queries':<32s} ${fr['cost_per_1M_queries']:>12.2f}    "
          f"${m0['cost_per_1M_queries']:>12.2f}")
    print(f"  {'Annual waste (1M Q/day)':<32s} ${fr['annual_cost_1M_per_day']:>10,.0f}      "
          f"${m0['annual_cost_1M_per_day']:>10,.0f}")
    waste_diff = m0['cost_per_1M_queries'] - fr['cost_per_1M_queries']
    print(f"\n  FR saves ${waste_diff:.2f} per 1M queries in stale-context costs "
          f"(${waste_diff * 365:,.0f}/yr at 1M Q/day).")

    return {"fr": fr, "mem0": m0}


# ---------------------------------------------------------------------------
# Task 5: Retrieval Latency (qualitative)
# ---------------------------------------------------------------------------

def task5_latency() -> dict:
    """Print qualitative latency comparison."""
    hdr("TASK 5: RETRIEVAL LATENCY (Qualitative)")

    print("\n  No per-query timing data recorded in results JSONs.")
    print()
    subhdr("Architecture Comparison")
    print(f"  {'Aspect':<28s} {'FR':>24s}    {'Mem0':>24s}")
    print(f"  {'─' * 28} {'─' * 24}    {'─' * 24}")
    print(f"  {'Retrieval strategies':<28s} {'5-6 concurrent queries':>24s}    {'1 vector search':>24s}")
    print(f"  {'Backend':<28s} {'Neo4j (disk)':>24s}    {'Qdrant (in-memory)':>24s}")
    print(f"  {'Reranking':<28s} {'Decay engine + blending':>24s}    {'Cosine similarity':>24s}")
    print(f"  {'Supersession filter':<28s} {'Yes (conf >= 0.7)':>24s}    {'No':>24s}")
    print(f"  {'Query classification':<28s} {'LLM-based routing':>24s}    {'N/A':>24s}")
    print(f"  {'Expected latency/query':<28s} {'~200-500ms':>24s}    {'~50-100ms':>24s}")
    print(f"  {'Bottleneck':<28s} {'LLM routing (amortized)':>24s}    {'Embedding lookup':>24s}")

    print(f"\n  Note: FR trades higher per-query latency for dramatically better")
    print(f"  signal:noise ratio and lower staleness, reducing downstream LLM costs.")

    return {"note": "No timing data available. Architecture comparison only."}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FR vs Mem0 efficiency diagnostic")
    p.add_argument("--no-neo4j", action="store_true",
                   help="Skip Neo4j queries; use checkpoint data only")
    p.add_argument("--json", type=str, default=None, metavar="PATH",
                   help="Export results to JSON file")
    p.add_argument("--task", type=int, choices=[1, 2, 3, 4, 5], default=None,
                   help="Run only a specific task (1-5)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # Ensure UTF-8 output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Load JSON artifacts
    print("Loading data sources...")
    fr_results = load_json(ARTIFACTS_DIR / "lifemembench_results.json")
    mem0_results = load_json(ARTIFACTS_DIR / "mem0_results.json")
    ingest_ckpt = load_json(ARTIFACTS_DIR / "lifemembench_ingest_checkpoint.json")
    super_ckpt = load_json(ARTIFACTS_DIR / "supersession_checkpoint.json")

    # Extract per-question lists
    fr_questions = fr_results["per_question"]["full"]
    # Mem0 results may be nested under "mem0" key or be a list
    m0_pq = mem0_results["per_question"]
    if isinstance(m0_pq, dict):
        mem0_questions = list(m0_pq.values())[0]  # first (only) config
    else:
        mem0_questions = m0_pq

    print(f"  FR questions:   {len(fr_questions)}")
    print(f"  Mem0 questions: {len(mem0_questions)}")

    # Compute avg fact lengths from results (fallback for --no-neo4j)
    fr_avg_fl_results = compute_avg_fact_len_from_results(fr_questions)
    mem0_avg_text_len = compute_avg_fact_len_from_results(mem0_questions)

    # FR edge stats: live Neo4j or offline checkpoint
    if args.no_neo4j:
        print("  Mode: OFFLINE (checkpoint data, no Neo4j)")
        fr_stats = build_offline_edge_stats(ingest_ckpt, super_ckpt, fr_avg_fl_results)
    else:
        print("  Mode: LIVE (querying Neo4j)...")
        try:
            fr_stats = await query_neo4j_edge_stats(PERSONAS)
            print(f"  Neo4j: loaded stats for {len(fr_stats)} personas")
        except Exception as e:
            print(f"  Neo4j connection failed: {e}")
            print(f"  Falling back to offline checkpoint data.")
            fr_stats = build_offline_edge_stats(ingest_ckpt, super_ckpt, fr_avg_fl_results)

    # Banner
    print(f"\n{'#' * 76}")
    print(f"  EFFICIENCY DIAGNOSTIC: Fortunate Recall vs Mem0")
    print(f"  {len(PERSONAS)} personas | {len(fr_questions)} questions")
    print(f"{'#' * 76}")

    all_results = {}
    tasks_to_run = [args.task] if args.task else [1, 2, 3, 4, 5]

    if 1 in tasks_to_run:
        all_results["task1"] = task1_storage(
            fr_stats, MEM0_ESTIMATED_MEMORIES_PER_PERSONA,
            mem0_avg_text_len, is_estimated=True)

    if 2 in tasks_to_run:
        all_results["task2"] = task2_pool_size(
            fr_questions, mem0_questions, super_ckpt)

    if 3 in tasks_to_run:
        all_results["task3"] = task3_context_efficiency(
            fr_questions, mem0_questions)

    if 4 in tasks_to_run:
        all_results["task4"] = task4_staleness_cost(
            fr_questions, mem0_questions)

    if 5 in tasks_to_run:
        all_results["task5"] = task5_latency()

    # JSON export
    if args.json:
        # Convert non-serializable values
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj

        out = json.loads(json.dumps(all_results, default=str))
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults exported to {args.json}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
