#!/usr/bin/env python3
"""LifeMemBench post-eval diagnostic analyzer.

Usage:
    python analyze_results.py [--results path/to/results.json] [--config full] [--compare path/to/other.json]

Reads lifemembench_results.json and lifemembench_questions.json, prints a
comprehensive diagnostic dashboard covering pass rates, failure classification,
staleness analysis, source distributions, and near-miss identification.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "artifacts" / "lifemembench_results.json"
QUESTIONS_PATH = SCRIPT_DIR / "lifemembench_questions.json"

OLD_PERSONAS = {"priya", "marcus", "elena", "david", "amara",
                "jake", "fatima", "tom", "kenji", "rosa"}
NEW_PERSONAS = {"callum", "diane", "raj", "nadia", "samuel",
                "lily", "omar", "bruna", "patrick", "aisha"}

FAILURE_CODES = {
    "G": "RETRACTION_KILLED",
    "H": "STALE_DOMINANCE",
    "E": "JUDGE_ERROR",
    "C": "RETRIEVAL_MISS",
    "A": "EXTRACTION_MISSING",
    "B": "EXTRACTION_WEAK",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trunc(s: str, n: int) -> str:
    """Truncate string to n chars, adding ... if needed."""
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def pct(num: int, den: int) -> str:
    """Format as percentage string."""
    if den == 0:
        return "  -  "
    return f"{100.0 * num / den:5.1f}%"


def fpct(val: float) -> str:
    """Format a float 0-1 as percentage."""
    return f"{100.0 * val:5.1f}%"


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

def classify_failure(q_result: dict) -> tuple[str, str]:
    """Classify a failing question into category (letter, name).

    Returns (code, evidence_text).
    """
    rank = q_result.get("answer_rank")
    staleness = q_result.get("staleness_penalty", 0)
    answerable = q_result.get("answerable", False)
    retraction_pass = q_result.get("retraction_pass", True)
    top_facts = q_result.get("top5_facts", [])

    # Find first wrong fact for evidence
    first_wrong = ""
    for f in top_facts:
        if f.get("contains_wrong"):
            first_wrong = f.get("fact", "")
            break

    # G: retraction killed
    if not retraction_pass:
        ev = first_wrong or "retracted fact in top-5"
        return "G", ev

    # H: stale dominance (correct found but stale outranks)
    if rank is not None and staleness > 0:
        return "H", first_wrong or "wrong outranks correct"

    # E: judge / scoring error (correct found, no staleness, still fails)
    if rank is not None and staleness == 0:
        return "E", f"rank={rank}, no staleness but av_pass=False"

    # answer_rank is None from here
    # C: retrieval miss (correct never extracted)
    if not answerable:
        return "C", "correct answer not in pool"

    # A vs B: check top-10 for topical adjacency
    has_adjacent = any(
        f.get("supports_correct") or f.get("contains_wrong")
        for f in top_facts
    )
    if not has_adjacent:
        return "A", "no related facts in top-10"

    # B: extraction weak (topically adjacent but no correct)
    ev = first_wrong or "adjacent fact found"
    return "B", ev


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

def print_header(title: str):
    w = 72
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def print_subheader(title: str):
    print(f"\n--- {title} {'-' * max(0, 66 - len(title))}")


def print_summary_dashboard(results: dict, config: str, per_q: list, q_lookup: dict):
    print_header("SUMMARY DASHBOARD")
    summ = results.get("summary", {}).get(config, {})

    total = summ.get("total_questions", len(per_q))
    answerable_n = summ.get("answerable", sum(1 for q in per_q if q.get("answerable")))
    pass_n = sum(1 for q in per_q if q.get("av_pass"))
    fail_n = total - pass_n
    mrr = summ.get("mrr", 0)
    h1 = summ.get("hit_at_1", 0)
    h5 = summ.get("hit_at_5", 0)
    avg_stale = summ.get("avg_staleness_penalty",
                         sum(q.get("staleness_penalty", 0) for q in per_q) / max(total, 1))

    print(f"  Config:     {config}")
    print(f"  Questions:  {total}  (answerable: {answerable_n})")
    print(f"  Pass:       {pass_n}/{total} ({pct(pass_n, total).strip()})")
    print(f"  Fail:       {fail_n}")
    print(f"  Staleness:  {fpct(avg_stale).strip()}")
    print(f"  MRR:        {mrr:.4f}")
    print(f"  H@1:        {fpct(h1).strip()}")
    print(f"  H@5:        {fpct(h5).strip()}")

    # --- Per-AV table ---
    print_subheader("Per Attack-Vector")

    av_groups = defaultdict(list)
    for q in per_q:
        av_groups[q["attack_vector"]].append(q)

    # Pre-compute failure categories per AV
    av_failure_cats = defaultdict(list)
    for q in per_q:
        if not q.get("av_pass"):
            code, _ = classify_failure(q)
            av_failure_cats[q["attack_vector"]].append(code)

    hdr = f"  {'AV':<32s}{'total':>6s}{'pass':>6s}{'fail':>6s}{'pass%':>7s}{'stale%':>8s}  {'dominant_failure'}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for av in sorted(av_groups):
        qs = av_groups[av]
        t = len(qs)
        p = sum(1 for q in qs if q.get("av_pass"))
        f_ = t - p
        st = sum(1 for q in qs if q.get("staleness_penalty", 0) > 0)
        cats = av_failure_cats.get(av, [])
        if cats:
            dominant = Counter(cats).most_common(1)[0]
            dom_str = f"{dominant[0]}:{FAILURE_CODES[dominant[0]]}({dominant[1]})"
        else:
            dom_str = "-"
        print(f"  {av:<32s}{t:>6d}{p:>6d}{f_:>6d}{pct(p, t):>7s}{pct(st, t):>8s}  {dom_str}")

    # --- Per-persona table ---
    print_subheader("Per Persona")

    persona_groups = defaultdict(list)
    for q in per_q:
        qid = q["question_id"]
        persona = qid.split("_")[0]
        persona_groups[persona].append(q)

    hdr = f"  {'persona':<12s}{'pass':>6s}{'total':>7s}{'rate%':>7s}  {'group'}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for persona in sorted(persona_groups):
        qs = persona_groups[persona]
        t = len(qs)
        p = sum(1 for q in qs if q.get("av_pass"))
        grp = "old" if persona in OLD_PERSONAS else "new" if persona in NEW_PERSONAS else "?"
        print(f"  {persona:<12s}{p:>6d}{t:>7d}{pct(p, t):>7s}  {grp}")


def print_failure_classification(per_q: list, q_lookup: dict):
    print_header("FAILURE CLASSIFICATION")

    failures = []
    for q in per_q:
        if q.get("av_pass"):
            continue
        code, evidence = classify_failure(q)
        failures.append((q, code, evidence))

    if not failures:
        print("  No failures!")
        return

    # Category summary
    print_subheader("Category Counts")
    cat_counts = Counter(code for _, code, _ in failures)
    total_f = len(failures)

    for code in ["G", "H", "E", "C", "A", "B"]:
        n = cat_counts.get(code, 0)
        label = FAILURE_CODES[code]
        print(f"  {code} ({label:<22s}): {n:>4d}  ({pct(n, total_f).strip()})")
    print(f"  {'TOTAL':<27s}: {total_f:>4d}")

    # Full table
    print_subheader("Failure Detail")
    hdr = f"  {'question_id':<16s}{'question':<52s}{'AV':<10s}{'cat':>4s}{'rank':>6s}{'stale':>7s}  {'evidence'}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for q, code, evidence in sorted(failures, key=lambda x: x[1]):
        qid = q["question_id"]
        qtext = q_lookup.get(qid, {}).get("question", "")
        av_short = q["attack_vector"].split("_")[0]
        rank = q.get("answer_rank")
        rank_s = str(rank) if rank is not None else "-"
        stale = q.get("staleness_penalty", 0)
        print(f"  {qid:<16s}{trunc(qtext, 50):<52s}{av_short:<10s}{code:>4s}{rank_s:>6s}{stale:>7.1f}  {trunc(evidence, 50)}")


def print_answerable_analysis(per_q: list, q_lookup: dict):
    print_header("ANSWERABLE ANALYSIS")

    has_rank = [q for q in per_q if q.get("answer_rank") is not None]
    no_rank = [q for q in per_q if q.get("answer_rank") is None]

    pass_has = sum(1 for q in has_rank if q.get("av_pass"))
    pass_no = sum(1 for q in no_rank if q.get("av_pass"))

    print(f"  answer_rank != None:  {len(has_rank):>4d} questions, pass={pass_has}, rate={pct(pass_has, len(has_rank)).strip()}")
    print(f"  answer_rank == None:  {len(no_rank):>4d} questions, pass={pass_no}, rate={pct(pass_no, len(no_rank)).strip()}")

    # Answerable failures: rank exists but still fails
    ans_failures = [q for q in has_rank if not q.get("av_pass")]
    if ans_failures:
        print_subheader("Answerable Failures (rank != None, av_pass = False)")
        hdr = f"  {'question_id':<16s}{'rank':>6s}{'stale':>7s}  {'top wrong fact'}"
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")
        for q in sorted(ans_failures, key=lambda x: x.get("answer_rank", 999)):
            qid = q["question_id"]
            rank = q["answer_rank"]
            stale = q.get("staleness_penalty", 0)
            wrong = ""
            for f in q.get("top5_facts", []):
                if f.get("contains_wrong"):
                    wrong = f.get("fact", "")
                    break
            print(f"  {qid:<16s}{rank:>6d}{stale:>7.1f}  {trunc(wrong, 60)}")
    else:
        print("  No answerable failures.")


def print_staleness_analysis(per_q: list, q_lookup: dict):
    print_header("STALENESS ANALYSIS")

    stale = [q for q in per_q if q.get("staleness_penalty", 0) > 0]
    clean = [q for q in per_q if q.get("staleness_penalty", 0) == 0]

    pass_stale = sum(1 for q in stale if q.get("av_pass"))
    pass_clean = sum(1 for q in clean if q.get("av_pass"))

    print(f"  staleness == 0:  {len(clean):>4d} questions, pass={pass_clean}, rate={pct(pass_clean, len(clean)).strip()}")
    print(f"  staleness >  0:  {len(stale):>4d} questions, pass={pass_stale}, rate={pct(pass_stale, len(stale)).strip()}")

    if stale:
        print_subheader("All Stale Questions")
        hdr = f"  {'question_id':<16s}{'AV':<10s}{'rank':>6s}{'pass':>6s}  {'top wrong fact'}"
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")
        for q in sorted(stale, key=lambda x: x["question_id"]):
            qid = q["question_id"]
            av_short = q["attack_vector"].split("_")[0]
            rank = q.get("answer_rank")
            rank_s = str(rank) if rank is not None else "-"
            passed = "PASS" if q.get("av_pass") else "FAIL"
            wrong = ""
            for f in q.get("top5_facts", []):
                if f.get("contains_wrong"):
                    wrong = f.get("fact", "")
                    break
            print(f"  {qid:<16s}{av_short:<10s}{rank_s:>6s}{passed:>6s}  {trunc(wrong, 55)}")


def print_source_distribution(per_q: list):
    print_header("SOURCE DISTRIBUTION (top-10 facts)")

    source_pass = Counter()
    source_fail = Counter()
    count_pass = 0
    count_fail = 0

    for q in per_q:
        facts = q.get("top5_facts", [])
        if not facts:
            continue
        src_counts = Counter(f.get("source", "unknown") for f in facts)
        if q.get("av_pass"):
            count_pass += 1
            for src, n in src_counts.items():
                source_pass[src] += n
        else:
            count_fail += 1
            for src, n in src_counts.items():
                source_fail[src] += n

    all_sources = sorted(set(source_pass) | set(source_fail))
    total_pass = sum(source_pass.values()) or 1
    total_fail = sum(source_fail.values()) or 1

    hdr = f"  {'source':<22s}{'pass_avg':>10s}{'pass_%':>8s}{'fail_avg':>10s}{'fail_%':>8s}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for src in all_sources:
        p = source_pass[src]
        f_ = source_fail[src]
        p_avg = p / max(count_pass, 1)
        f_avg = f_ / max(count_fail, 1)
        print(f"  {src:<22s}{p_avg:>10.2f}{pct(p, total_pass):>8s}{f_avg:>10.2f}{pct(f_, total_fail):>8s}")


def print_almost_passing(per_q: list, q_lookup: dict):
    print_header("ALMOST PASSING")

    supersession_flips = []
    judge_issues = []

    for q in per_q:
        if q.get("av_pass"):
            continue
        rank = q.get("answer_rank")
        if rank is None or rank > 10:
            continue
        stale = q.get("staleness_penalty", 0)
        if stale > 0:
            supersession_flips.append(q)
        else:
            judge_issues.append(q)

    print_subheader("Flip with better supersession (rank 1-10, stale > 0)")
    if supersession_flips:
        for q in sorted(supersession_flips, key=lambda x: x["answer_rank"]):
            qid = q["question_id"]
            rank = q["answer_rank"]
            qtext = q_lookup.get(qid, {}).get("question", "")
            print(f"  {qid:<16s} rank={rank:<3d} {trunc(qtext, 55)}")
    else:
        print("  None")

    print_subheader("Potential judge issues (rank 1-10, stale == 0)")
    if judge_issues:
        for q in sorted(judge_issues, key=lambda x: x["answer_rank"]):
            qid = q["question_id"]
            rank = q["answer_rank"]
            qtext = q_lookup.get(qid, {}).get("question", "")
            print(f"  {qid:<16s} rank={rank:<3d} {trunc(qtext, 55)}")
    else:
        print("  None")


def print_comparison(per_q_a: list, compare_path: str, config: str, q_lookup: dict):
    print_header(f"COMPARISON: current vs {Path(compare_path).name}")

    compare_data = load_json(Path(compare_path))
    if config not in compare_data.get("per_question", {}):
        print(f"  ERROR: Config '{config}' not found in comparison file.")
        avail = list(compare_data.get("per_question", {}).keys())
        print(f"  Available: {', '.join(avail) if avail else 'none'}")
        return

    per_q_b = compare_data["per_question"][config]

    # Build lookup by question_id
    a_map = {q["question_id"]: q.get("av_pass", False) for q in per_q_a}
    b_map = {q["question_id"]: q.get("av_pass", False) for q in per_q_b}

    all_ids = sorted(set(a_map) | set(b_map))

    gained = []   # FAIL -> PASS
    lost = []     # PASS -> FAIL

    for qid in all_ids:
        old = b_map.get(qid)
        new = a_map.get(qid)
        if old is None or new is None:
            continue
        if not old and new:
            gained.append(qid)
        elif old and not new:
            lost.append(qid)

    print(f"  Gained (FAIL->PASS): {len(gained)}")
    print(f"  Lost   (PASS->FAIL): {len(lost)}")
    print(f"  Net:                 {len(gained) - len(lost):+d}")

    if gained:
        print_subheader("Gained (FAIL -> PASS)")
        for qid in gained:
            qtext = q_lookup.get(qid, {}).get("question", "")
            av = q_lookup.get(qid, {}).get("attack_vector", "?").split("_")[0]
            print(f"  {qid:<16s}{av:<8s}{trunc(qtext, 55)}")

    if lost:
        print_subheader("Lost (PASS -> FAIL)")
        for qid in lost:
            qtext = q_lookup.get(qid, {}).get("question", "")
            av = q_lookup.get(qid, {}).get("attack_vector", "?").split("_")[0]
            print(f"  {qid:<16s}{av:<8s}{trunc(qtext, 55)}")

    # Per-AV delta
    print_subheader("Per-AV Delta")
    av_delta = defaultdict(lambda: {"gained": 0, "lost": 0})

    for qid in gained:
        av = q_lookup.get(qid, {}).get("attack_vector", "unknown")
        av_delta[av]["gained"] += 1
    for qid in lost:
        av = q_lookup.get(qid, {}).get("attack_vector", "unknown")
        av_delta[av]["lost"] += 1

    hdr = f"  {'AV':<32s}{'gained':>8s}{'lost':>6s}{'net':>6s}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for av in sorted(av_delta):
        g = av_delta[av]["gained"]
        lo = av_delta[av]["lost"]
        print(f"  {av:<32s}{g:>8d}{lo:>6d}{g - lo:>+6d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LifeMemBench post-eval diagnostic analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS),
                        help=f"Path to results JSON (default: {DEFAULT_RESULTS.name})")
    parser.add_argument("--config", type=str, default="full",
                        help="Config to analyze (default: full)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to second results JSON for delta analysis")
    args = parser.parse_args()

    # Load data
    results = load_json(Path(args.results))
    questions_raw = load_json(QUESTIONS_PATH)
    q_lookup = {q["id"]: q for q in questions_raw}

    # Show available configs
    available = list(results.get("per_question", {}).keys())
    print(f"Available configs: {', '.join(available) if available else 'none'}")

    if args.config not in results.get("per_question", {}):
        print(f"\nERROR: Config '{args.config}' not found in results.")
        print(f"Available: {', '.join(available)}")
        sys.exit(1)

    per_q = results["per_question"][args.config]
    print(f"Analyzing config: {args.config}  ({len(per_q)} questions)")

    # Run all sections
    print_summary_dashboard(results, args.config, per_q, q_lookup)
    print_failure_classification(per_q, q_lookup)
    print_answerable_analysis(per_q, q_lookup)
    print_staleness_analysis(per_q, q_lookup)
    print_source_distribution(per_q)
    print_almost_passing(per_q, q_lookup)

    if args.compare:
        print_comparison(per_q, args.compare, args.config, q_lookup)

    print(f"\n{'=' * 72}")
    print("  Analysis complete.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
