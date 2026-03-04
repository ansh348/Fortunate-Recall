"""
failure_audit.py — Complete failure audit for LifeMemBench evaluation.

Reads local JSON files only (no Neo4j). Produces failure_audit_report.md with:
  1. Per-question failure table (41 rows) with 8-category taxonomy
  2. Failures grouped by attack vector
  3. Fixability classification

Usage: python failure_audit.py
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
ARTIFACTS_DIR = LIFEMEMEVAL_DIR / "artifacts"
RESULTS_PATH = ARTIFACTS_DIR / "lifemembench_results.json"
QUESTIONS_PATH = LIFEMEMEVAL_DIR / "lifemembench_questions.json"
REPORT_PATH = PROJECT_ROOT / "failure_audit_report.md"

sys.path.insert(0, str(PROJECT_ROOT))
from extraction_audit import KEYWORD_REGISTRY

# ---------------------------------------------------------------------------
# Classification map: question_id → (category, reason, fix_type)
#
# Categories:
#   E1           — Benchmark design bug (wrong ground truth)
#   E2           — Fact never extracted (no edge exists)
#   E3-TRUE      — Correct edge exists but retrieval didn't surface it
#   E3-WEAK      — Edge exists but too vague/partial to answer
#   E4           — Edge in wrong category
#   AV7-CROSS    — Cross-category supersession failure (AV7 + wrong category)
#   RETRACTION   — Retraction event not extracted
#   RANKING      — Correct edge in pool but outranked by stale edge
#
# Fix types:
#   (a) code   — fixable by code change only (retrieval/ranking/routing)
#   (b) ingest — fixable by re-ingestion only (extraction model)
#   (c) both   — needs both code change and re-ingestion
#   (d) bench  — benchmark design issue (fix the benchmark, not the system)
# ---------------------------------------------------------------------------

CLASSIFICATION_MAP = {
    # ── E1: Benchmark design bug ──────────────────────────────────────────
    "elena_q13": (
        "E1",
        "Edge has 'about 42k' but question expects exact '$40,000' — benchmark/conversation mismatch",
        "d",
    ),
    "omar_q11": (
        "E1",
        "Correct answer is 'None explicitly mentioned' — no extractable fact possible",
        "d",
    ),

    # ── E2: Fact never extracted ──────────────────────────────────────────
    "priya_q07": (
        "E2",
        "Rock climbing fact in session 25 but no correct-answer edge extracted (only topic-level edges)",
        "b",
    ),
    "amara_q14": (
        "E2",
        "UCL LLM cost £18,000 in session 11 — entity node references it but no answer edge extracted",
        "b",
    ),

    # ── RETRACTION: AV7 retraction event not extracted ────────────────────
    "david_q04": (
        "RETRACTION",
        "Book abandonment ('nobody would read it') in sessions 9/18 — retraction never extracted as edge",
        "b",
    ),
    "amara_q06": (
        "RETRACTION",
        "Dropped UCL human rights LLM in sessions 11/17 — retraction event never extracted",
        "b",
    ),

    # ── AV7-CROSS: Cross-category supersession failure ────────────────────
    "priya_q05": (
        "AV7-CROSS",
        "Retraction edges in {RELATIONAL_BONDS, FINANCIAL_MATERIAL} but query routes to {PREFERENCES_HABITS, LOGISTICAL_CONTEXT}",
        "a",
    ),

    # ── E4: Edge in wrong category ────────────────────────────────────────
    "marcus_q13": (
        "E4",
        "Carlos hiring edge in {IDENTITY_SELF_CONCEPT, RELATIONAL_BONDS} but question expects {PROJECTS_ENDEAVORS}",
        "a",
    ),
    "elena_q12": (
        "E4",
        "Knitting edge in {PREFERENCES_HABITS} but broad query routes to {HOBBIES_RECREATION, HEALTH_WELLBEING}",
        "a",
    ),
    "amara_q02": (
        "E4",
        "Samsung Galaxy S25 edge in {FINANCIAL_MATERIAL} but question routes to {PREFERENCES_HABITS}",
        "a",
    ),
    "amara_q07": (
        "E4",
        "Chambers dinner wine edge in {LOGISTICAL_CONTEXT} but contradiction question needs {PREFERENCES_HABITS, HEALTH_WELLBEING}",
        "a",
    ),

    # ── RANKING: Correct edge in pool but outranked ───────────────────────
    "omar_q01": (
        "RANKING",
        "Correct 'Uber pays the bills' edge at rank 4; stale 'driving for Lyft' edge at rank 2 outranks it",
        "a",
    ),

    # ── E3-WEAK: Edge exists but too vague/partial ────────────────────────
    "elena_q08": (
        "E3-WEAK",
        "Composite fact across 3 sessions (Guadalajara, first-gen, Sofia/Down syndrome) — partial edges exist but incomplete",
        "c",
    ),
    "jake_q10": (
        "E3-WEAK",
        "Composite fact across 3 sessions (rec hockey, gaming PC, craft beer) — partial edges exist but incomplete",
        "c",
    ),

    # ── E3-TRUE: Correct edge exists but retrieval didn't surface it ──────
    "priya_q03": (
        "E3-TRUE",
        "Edge 'user received ADHD diagnosis since college' exists (2 enriched) — semantic search misses it",
        "a",
    ),
    "priya_q09": (
        "E3-TRUE",
        "Edge 'user has chronic migraines' exists (9 enriched) — retrieval misses HEALTH_WELLBEING edges",
        "a",
    ),
    "marcus_q03": (
        "E3-TRUE",
        "Edge about scrapping Germantown location exists (7 enriched) — not surfaced by any retrieval strategy",
        "a",
    ),
    "marcus_q06": (
        "E3-TRUE",
        "Edge 'user has been going to Sardis Lake for fishing' exists (1 enriched) — category routing misses it",
        "a",
    ),
    "elena_q01": (
        "E3-TRUE",
        "Edge 'user is employed at Rush University Medical Center' exists (1 enriched) — not retrieved for workplace query",
        "a",
    ),
    "elena_q02": (
        "E3-TRUE",
        "Edge 'doctor told user to switch to Mediterranean diet' exists (2 enriched) — retrieval gap",
        "a",
    ),
    "elena_q03": (
        "E3-TRUE",
        "Edge about dropping NP plan exists (9 enriched) — retraction edge not surfaced despite keyword match",
        "a",
    ),
    "elena_q04": (
        "E3-TRUE",
        "Edge 'user is on sertraline for anxiety' exists (3 enriched) — stable identity fact buried",
        "a",
    ),
    "david_q01": (
        "E3-TRUE",
        "Edge about switching to AP US History exists (5 enriched) — multi-version fact not surfaced",
        "a",
    ),
    "david_q03": (
        "E3-TRUE",
        "Edge 'drove home in a Subaru Outback' exists (10 enriched) — vehicle supersession not surfaced",
        "a",
    ),
    "david_q05": (
        "E3-TRUE",
        "Edge 'user wears a hearing aid, mild hearing loss in left ear' exists (1 enriched) — stable fact buried",
        "a",
    ),
    "david_q11": (
        "E3-TRUE",
        "Edge about parent-teacher conferences exists (4 enriched) — expired logistics edge not retrieved",
        "a",
    ),
    "amara_q03": (
        "E3-TRUE",
        "Edge 'user has started boxing at a gym in Bethnal Green' exists (7 enriched) — superseded preference not surfaced",
        "a",
    ),
    "amara_q08": (
        "E3-TRUE",
        "Edge referencing Hackney exists (10 enriched) — soft supersession fact about considering Islington buried",
        "a",
    ),
    "amara_q11": (
        "E3-TRUE",
        "Edge about boxing in Bethnal Green exists (4 enriched) — broad query fails to aggregate hobbies",
        "a",
    ),
    "amara_q12": (
        "E3-TRUE",
        "Edge 'user did LLB at King's' exists (1 enriched) — stable identity fact not retrieved",
        "a",
    ),
    "jake_q03": (
        "E3-TRUE",
        "Edge about mom at Eire Pub in Dorchester exists (8 enriched) — family background fact buried",
        "a",
    ),
    "jake_q05": (
        "E3-TRUE",
        "Edge about EV charger side gig exists (7 enriched) — retraction not surfaced by retrieval",
        "a",
    ),
    "jake_q07": (
        "E3-TRUE",
        "Edge 'user was dragged to Trillium' exists (2 enriched) — craft beer preference not surfaced",
        "a",
    ),
    "tom_q02": (
        "E3-TRUE",
        "Edge 'user has just bought a Hyundai Ioniq 5' exists (3 enriched) — vehicle supersession not surfaced",
        "a",
    ),
    "tom_q03": (
        "E3-TRUE",
        "Edge 'user has recently joined a walking group' exists (11 enriched) — hobby supersession not surfaced",
        "a",
    ),
    "tom_q04": (
        "E3-TRUE",
        "Edge 'user has atrial fibrillation' exists (3 enriched) — stable health fact buried in noise",
        "a",
    ),
    "tom_q06": (
        "E3-TRUE",
        "Edge about barn conversion planning permission exists (4 enriched) — retraction not surfaced",
        "a",
    ),
    "tom_q07": (
        "E3-TRUE",
        "Edges about 'no social media' AND 'Instagram bee photo' exist (9 enriched) — contradiction not surfaced",
        "a",
    ),
    "omar_q02": (
        "E3-TRUE",
        "Edge 'user just moved to Gulfton' exists (2 enriched) — address supersession not surfaced",
        "a",
    ),
    "omar_q05": (
        "E3-TRUE",
        "Edge 'the whole Camry plan is dead' exists (1 enriched) — retraction edge not surfaced",
        "a",
    ),
    "omar_q06": (
        "E3-TRUE",
        "Edge 'user got PS5 off Facebook Marketplace' exists (1 enriched) — cross-session contradiction buried",
        "a",
    ),
}

FIX_LABELS = {
    "a": "(a) Code change only",
    "b": "(b) Re-ingestion only",
    "c": "(c) Both code + re-ingestion",
    "d": "(d) Benchmark fix",
}

FIX_ACTIONS = {
    "a": "Fix retrieval routing, ranking, or category logic",
    "b": "Re-run extraction/ingestion pipeline to capture missing facts",
    "c": "Improve extraction for composite facts AND fix retrieval for multi-session answers",
    "d": "Update benchmark ground truth to match actual conversation data",
}

CATEGORY_ORDER = ["E1", "E2", "E3-TRUE", "E3-WEAK", "E4", "AV7-CROSS", "RETRACTION", "RANKING"]

AV_ORDER = [
    "AV1_superseded_preference",
    "AV2_expired_logistics",
    "AV3_stable_identity",
    "AV4_multi_version_fact",
    "AV5_broad_query",
    "AV6_cross_session_contradiction",
    "AV7_selective_forgetting",
    "AV8_numeric_preservation",
    "AV9_soft_supersession",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results_data = json.load(f)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions_list = json.load(f)
    questions = {q["id"]: q for q in questions_list}
    per_question = results_data["per_question"]["full"]
    per_av = results_data.get("per_attack_vector", {}).get("full", {})
    return results_data, questions, per_question, per_av


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_top_edges(top5_facts: list[dict], max_edges: int = 5) -> str:
    """Format top retrieved edges as a compact multi-line string for the table."""
    if not top5_facts:
        return "(no edges retrieved)"
    lines = []
    for f in top5_facts[:max_edges]:
        rank = f["rank"]
        fact = f["fact"][:80].replace("|", "\\|").replace("\n", " ")
        bl = f.get("blended", 0)
        marker = ""
        if f.get("supports_correct"):
            marker = " **[CORRECT]**"
        elif f.get("contains_wrong"):
            marker = " *[STALE]*"
        lines.append(f"#{rank} (bl={bl:.3f}) {fact}{marker}")
    return "<br>".join(lines)


def generate_report(questions, per_question, per_av, results_data):
    failing = [r for r in per_question if not r["av_pass"]]
    failing.sort(key=lambda r: r["question_id"])

    total_qs = results_data["meta"]["total_questions"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append("# LifeMemBench Failure Audit Report\n")
    lines.append(f"*Generated: {timestamp}*\n")
    lines.append(f"**Results file:** `lifemembench_results.json` (alpha=0.3, {total_qs} questions)\n")

    # ── Executive Summary ─────────────────────────────────────────────────
    cls_counts = Counter()
    fix_counts = Counter()
    for r in failing:
        qid = r["question_id"]
        cat, _, fix = CLASSIFICATION_MAP.get(qid, ("UNKNOWN", "Not classified", "?"))
        cls_counts[cat] += 1
        fix_counts[fix] += 1

    lines.append("## Executive Summary\n")
    lines.append(f"- **Total questions:** {total_qs}")
    lines.append(f"- **Passing (av_pass=True):** {total_qs - len(failing)} ({(total_qs - len(failing)) / total_qs * 100:.1f}%)")
    lines.append(f"- **Failing (av_pass=False):** {len(failing)} ({len(failing) / total_qs * 100:.1f}%)")
    lines.append(f"- **Answerable but failing:** {sum(1 for r in failing if r['answerable'])}")
    lines.append("")
    lines.append("| Classification | Count | % of Failures | Description |")
    lines.append("|---|---|---|---|")
    for cat in CATEGORY_ORDER:
        c = cls_counts.get(cat, 0)
        pct = (c / len(failing) * 100) if failing else 0
        desc = {
            "E1": "Benchmark design bug (wrong ground truth)",
            "E2": "Fact never extracted (no edge exists)",
            "E3-TRUE": "Correct edge exists but retrieval didn't surface it",
            "E3-WEAK": "Edge exists but too vague/partial to answer",
            "E4": "Edge in wrong behavioral category",
            "AV7-CROSS": "Cross-category supersession failure",
            "RETRACTION": "Retraction event not extracted",
            "RANKING": "Correct edge in pool but outranked",
        }.get(cat, "")
        lines.append(f"| **{cat}** | {c} | {pct:.1f}% | {desc} |")
    if cls_counts.get("UNKNOWN"):
        lines.append(f"| UNKNOWN | {cls_counts['UNKNOWN']} | — | Not classified |")
    lines.append(f"| **Total** | **{len(failing)}** | **100%** | |")
    lines.append("")

    # ── Key Insight ───────────────────────────────────────────────────────
    code_fixable = fix_counts.get("a", 0)
    lines.append(f"**Key insight:** {code_fixable}/{len(failing)} failures ({code_fixable / len(failing) * 100:.0f}%) "
                 f"are fixable by code changes alone (retrieval/routing/ranking). "
                 f"Only {fix_counts.get('b', 0)} require re-ingestion.\n")

    # ── Table 1: Per-Question Failures ────────────────────────────────────
    lines.append("## Table 1: Per-Question Failure Detail\n")
    lines.append("| # | Question ID | AV | Persona | Question | Expected Answer | Top Retrieved Edges | Classification | Fix |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for i, r in enumerate(failing, 1):
        qid = r["question_id"]
        q = questions.get(qid, {})
        av_raw = r.get("attack_vector", q.get("attack_vector", ""))
        av_short = av_raw.split("_")[0] if av_raw else "?"
        persona = q.get("persona", qid.split("_")[0])
        question_text = q.get("question", "").replace("|", "\\|")[:80]
        correct = q.get("correct_answer", "").replace("|", "\\|")[:80]
        top_edges = format_top_edges(r.get("top5_facts", []))

        cat, reason, fix = CLASSIFICATION_MAP.get(qid, ("UNKNOWN", "Not classified", "?"))
        fix_label = FIX_LABELS.get(fix, fix)

        lines.append(
            f"| {i} | `{qid}` | {av_short} | {persona} | {question_text} | {correct} | {top_edges} | **{cat}** | {fix_label} |"
        )

    lines.append("")

    # ── Table 2: Failures by Attack Vector ────────────────────────────────
    lines.append("## Table 2: Failures by Attack Vector\n")

    # Build AV → total questions count
    av_total = Counter(q.get("attack_vector", "") for q in questions.values())

    # Build AV → classification counts
    av_cls = defaultdict(lambda: Counter())
    for r in failing:
        qid = r["question_id"]
        av = r.get("attack_vector", questions.get(qid, {}).get("attack_vector", ""))
        cat, _, _ = CLASSIFICATION_MAP.get(qid, ("UNKNOWN", "", ""))
        av_cls[av][cat] += 1

    header_cats = " | ".join(CATEGORY_ORDER)
    lines.append(f"| Attack Vector | Total Qs | Failing | {header_cats} |")
    lines.append(f"|---|---|---|{'---|' * len(CATEGORY_ORDER)}")

    for av in AV_ORDER:
        total = av_total.get(av, 0)
        cls = av_cls.get(av, Counter())
        fail_count = sum(cls.values())
        cols = " | ".join(str(cls.get(cat, 0)) if cls.get(cat, 0) > 0 else "·" for cat in CATEGORY_ORDER)
        av_label = av.replace("_", " ", 1).replace("_", "-")
        lines.append(f"| {av_label} | {total} | {fail_count} | {cols} |")

    lines.append(f"| **Total** | **{total_qs}** | **{len(failing)}** | "
                 + " | ".join(f"**{cls_counts.get(cat, 0)}**" for cat in CATEGORY_ORDER) + " |")
    lines.append("")

    # ── Table 3: Fixability Summary ───────────────────────────────────────
    lines.append("## Table 3: Fixability Summary\n")
    lines.append("| Fix Type | Count | % | Categories | Action |")
    lines.append("|---|---|---|---|---|")

    fix_to_cats = defaultdict(list)
    for r in failing:
        qid = r["question_id"]
        cat, _, fix = CLASSIFICATION_MAP.get(qid, ("UNKNOWN", "", "?"))
        if cat not in fix_to_cats[fix]:
            fix_to_cats[fix].append(cat)

    for fix_code in ["a", "b", "c", "d"]:
        c = fix_counts.get(fix_code, 0)
        pct = (c / len(failing) * 100) if failing else 0
        cats = ", ".join(fix_to_cats.get(fix_code, []))
        action = FIX_ACTIONS.get(fix_code, "")
        label = FIX_LABELS.get(fix_code, fix_code)
        lines.append(f"| {label} | {c} | {pct:.0f}% | {cats} | {action} |")

    lines.append(f"| **Total** | **{len(failing)}** | **100%** | | |")
    lines.append("")

    # ── Table 4: Detailed Failure Reasons ─────────────────────────────────
    lines.append("## Table 4: Detailed Failure Reasons\n")

    for cat in CATEGORY_ORDER:
        cat_failures = [(r, CLASSIFICATION_MAP.get(r["question_id"], ("", "", "")))
                        for r in failing
                        if CLASSIFICATION_MAP.get(r["question_id"], ("", "", ""))[0] == cat]
        if not cat_failures:
            continue

        lines.append(f"### {cat}\n")
        for r, (_, reason, fix) in cat_failures:
            qid = r["question_id"]
            q = questions.get(qid, {})
            reg = KEYWORD_REGISTRY.get(qid, {})
            lines.append(f"**{qid}** — {q.get('persona', '')} | {q.get('attack_vector', '')}")
            lines.append(f"- **Q:** {q.get('question', '')}")
            lines.append(f"- **Expected:** {q.get('correct_answer', '')}")
            lines.append(f"- **Reason:** {reason}")
            if reg.get("fact_text"):
                lines.append(f"- **Expected fact:** {reg['fact_text']}")
            rank = r.get("answer_rank")
            if rank:
                lines.append(f"- **Answer rank:** {rank} (pool size: {r.get('pool_size', '?')})")
            else:
                lines.append(f"- **Answer rank:** not found (pool size: {r.get('pool_size', '?')})")

            # Show top 3 retrieved edges
            top = r.get("top5_facts", [])[:3]
            if top:
                lines.append("- **Top retrieved:**")
                for f in top:
                    marker = ""
                    if f.get("supports_correct"):
                        marker = " **[CORRECT]**"
                    elif f.get("contains_wrong"):
                        marker = " *[STALE]*"
                    lines.append(f"  - #{f['rank']} (bl={f.get('blended', 0):.3f}) "
                                 f"{f['fact'][:100]}{marker}")
            lines.append("")

    # ── Per-Persona Summary ───────────────────────────────────────────────
    lines.append("## Per-Persona Summary\n")
    persona_cls = defaultdict(lambda: Counter())
    for r in failing:
        qid = r["question_id"]
        persona = questions.get(qid, {}).get("persona", qid.split("_")[0])
        cat, _, _ = CLASSIFICATION_MAP.get(qid, ("UNKNOWN", "", ""))
        persona_cls[persona][cat] += 1

    header = " | ".join(CATEGORY_ORDER)
    lines.append(f"| Persona | Failing | {header} |")
    lines.append(f"|---|---|{'---|' * len(CATEGORY_ORDER)}")

    for p in sorted(persona_cls):
        cls = persona_cls[p]
        total = sum(cls.values())
        cols = " | ".join(str(cls.get(cat, 0)) if cls.get(cat, 0) > 0 else "·" for cat in CATEGORY_ORDER)
        lines.append(f"| {p} | {total} | {cols} |")

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_data, questions, per_question, per_av = load_data()

    # Validate all failing questions are classified
    failing = [r for r in per_question if not r["av_pass"]]
    unclassified = [r["question_id"] for r in failing if r["question_id"] not in CLASSIFICATION_MAP]
    if unclassified:
        print(f"WARNING: {len(unclassified)} unclassified failures: {unclassified}", file=sys.stderr)

    report = generate_report(questions, per_question, per_av, results_data)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {REPORT_PATH}")
    print(f"  Failing: {len(failing)} / {results_data['meta']['total_questions']}")
    print(f"  Classified: {len(failing) - len(unclassified)} / {len(failing)}")

    # Quick sanity check
    cls_counts = Counter()
    for r in failing:
        cat, _, _ = CLASSIFICATION_MAP.get(r["question_id"], ("UNKNOWN", "", ""))
        cls_counts[cat] += 1
    for cat in CATEGORY_ORDER:
        print(f"  {cat:12s}: {cls_counts.get(cat, 0)}")


if __name__ == "__main__":
    main()
