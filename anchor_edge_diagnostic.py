r"""
anchor_edge_diagnostic.py — Identify generic anchor edges that dominate retrieval
and simulate their suppression to count recoverable failing questions.

An "anchor edge" is a fact that:
  1. Appears in top-10 of 5+ different questions for the same persona, AND
  2. Has avg semantic score > 0.90, AND
  3. Almost never supports_correct (< 20% of appearances)

These edges are identity tautologies ("user is Omar", "Amara works at chambers")
that match every question about a persona equally well, crowding out specific
factual edges.

Usage: python anchor_edge_diagnostic.py
"""

import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
ARTIFACTS_DIR = LIFEMEMEVAL_DIR / "artifacts"
RESULTS_PATH = ARTIFACTS_DIR / "lifemembench_results.json"
QUESTIONS_PATH = LIFEMEMEVAL_DIR / "lifemembench_questions.json"


# ---------------------------------------------------------------------------
# Pattern-based anchor detection
# ---------------------------------------------------------------------------

# Regex patterns that match ultra-generic identity edges
ANCHOR_PATTERNS = [
    # "[name] is the user" / "user is [name]"
    r"^(?:user is |assistant (?:refers to|addresses|calls) user as )\w+$",
    r"^\w+ is (?:the )?user$",
    r"^user is (?:identified|referred to|addressed|known) as \w+",
    # "[name] is a member of [X]"
    r"is a member of",
    # Name-only tautologies
    r"^\w+ \w+ is the name of the user$",
]

ANCHOR_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in ANCHOR_PATTERNS]


def is_pattern_anchor(fact: str) -> bool:
    """Check if a fact matches known ultra-generic patterns."""
    for pat in ANCHOR_PATTERNS_COMPILED:
        if pat.search(fact):
            return True
    return False


# ---------------------------------------------------------------------------
# Frequency-based anchor detection
# ---------------------------------------------------------------------------

def find_frequency_anchors(
    per_question: list[dict],
    min_questions: int = 5,
    min_avg_semantic: float = 0.90,
    max_correct_rate: float = 0.20,
) -> set[str]:
    """Find edges that appear in top-10 of many questions but rarely help."""
    # Group by persona to count per-persona frequency
    persona_edge_data = defaultdict(lambda: defaultdict(list))

    for r in per_question:
        qid = r["question_id"]
        persona = qid.rsplit("_q", 1)[0]
        for f in r.get("top5_facts", []):
            fact = f["fact"]
            persona_edge_data[persona][fact].append({
                "qid": qid,
                "semantic": f.get("semantic", 0),
                "supports_correct": f.get("supports_correct", False),
                "contains_wrong": f.get("contains_wrong", False),
                "rank": f.get("rank", 99),
            })

    anchor_facts = set()
    anchor_details = {}

    for persona, edges in persona_edge_data.items():
        for fact, appearances in edges.items():
            n_questions = len(set(a["qid"] for a in appearances))
            if n_questions < min_questions:
                continue

            avg_sem = sum(a["semantic"] for a in appearances) / len(appearances)
            if avg_sem < min_avg_semantic:
                continue

            correct_rate = sum(1 for a in appearances if a["supports_correct"]) / len(appearances)
            if correct_rate > max_correct_rate:
                continue

            anchor_facts.add(fact)
            anchor_details[fact] = {
                "persona": persona,
                "n_questions": n_questions,
                "avg_semantic": round(avg_sem, 3),
                "correct_rate": round(correct_rate, 3),
                "appearances": len(appearances),
            }

    return anchor_facts, anchor_details


# ---------------------------------------------------------------------------
# AV-specific pass simulation
# ---------------------------------------------------------------------------

def simulate_av_pass(attack_vector: str, top10_verdicts: list[dict]) -> bool:
    """Simulate av_pass from verdict dicts (same logic as evaluate_lifemembench)."""
    topK = top10_verdicts[:10]
    top5 = top10_verdicts[:5]

    has_correct = any(v["supports_correct"] for v in topK)
    has_wrong_topK = any(v["contains_wrong"] for v in topK)
    has_wrong_top5 = any(v["contains_wrong"] for v in top5)

    first_correct_rank = None
    first_wrong_rank = None
    for i, v in enumerate(topK):
        if v["supports_correct"] and first_correct_rank is None:
            first_correct_rank = i + 1
        if v["contains_wrong"] and first_wrong_rank is None:
            first_wrong_rank = i + 1

    av_prefix = attack_vector.split("_")[0]

    if av_prefix in ("AV1", "AV4", "AV6", "AV9"):
        if not has_correct:
            return False
        if has_wrong_topK and first_wrong_rank < first_correct_rank:
            return False
        return True
    elif av_prefix == "AV2":
        if not has_wrong_top5:
            return True
        if has_correct and first_correct_rank < first_wrong_rank:
            return True
        return False
    elif av_prefix == "AV7":
        return has_correct and not has_wrong_top5
    else:
        return has_correct


# ---------------------------------------------------------------------------
# Dry-run simulation
# ---------------------------------------------------------------------------

def simulate_suppression(
    per_question: list[dict],
    anchor_facts: set[str],
    questions: dict,
) -> dict:
    """Simulate removing anchor edges from all questions and re-evaluating.

    For each question, remove anchor edges from top-10, let remaining edges
    slide up, and re-check av_pass.
    """
    results = {
        "recovered": [],
        "regressed": [],
        "unchanged_pass": [],
        "unchanged_fail": [],
        "details": {},
    }

    for r in per_question:
        qid = r["question_id"]
        av = r["attack_vector"]
        old_pass = r["av_pass"]
        top_facts = r.get("top5_facts", [])

        # Remove anchor edges
        filtered = [f for f in top_facts if f["fact"] not in anchor_facts]
        removed_count = len(top_facts) - len(filtered)

        # Build verdict list from remaining edges (up to 10)
        new_top10 = filtered[:10]
        verdicts = [
            {
                "supports_correct": f.get("supports_correct", False),
                "contains_wrong": f.get("contains_wrong", False),
            }
            for f in new_top10
        ]

        new_pass = simulate_av_pass(av, verdicts) if verdicts else False

        detail = {
            "old_pass": old_pass,
            "new_pass": new_pass,
            "removed_anchors": removed_count,
            "old_top10_size": len(top_facts),
            "new_top10_size": len(new_top10),
        }

        if not old_pass and new_pass:
            results["recovered"].append(qid)
            detail["status"] = "RECOVERED"
        elif old_pass and not new_pass:
            results["regressed"].append(qid)
            detail["status"] = "REGRESSED"
        elif old_pass:
            results["unchanged_pass"].append(qid)
            detail["status"] = "still_pass"
        else:
            results["unchanged_fail"].append(qid)
            detail["status"] = "still_fail"

        # Show what the new top-3 would be for failing questions
        if not old_pass:
            detail["new_top3"] = [
                {
                    "rank": i + 1,
                    "fact": f["fact"][:100],
                    "semantic": f.get("semantic", 0),
                    "blended": f.get("blended", 0),
                    "supports_correct": f.get("supports_correct", False),
                    "contains_wrong": f.get("contains_wrong", False),
                }
                for i, f in enumerate(new_top10[:3])
            ]

        results["details"][qid] = detail

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ANCHOR EDGE DIAGNOSTIC")
    print("=" * 70)

    results_data = json.load(open(RESULTS_PATH, encoding="utf-8"))
    questions_list = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    questions = {q["id"]: q for q in questions_list}
    per_question = results_data["per_question"]["full"]

    total = len(per_question)
    current_passing = sum(1 for r in per_question if r["av_pass"])
    current_failing = total - current_passing
    print(f"\nCurrent: {current_passing}/{total} passing ({current_passing/total:.1%})")

    # Step 1: Find frequency-based anchors
    print("\n--- Frequency-based anchor detection ---")
    freq_anchors, freq_details = find_frequency_anchors(per_question)
    print(f"Found {len(freq_anchors)} frequency anchors (5+ questions, >0.90 sem, <20% correct)")

    # Step 2: Find pattern-based anchors
    print("\n--- Pattern-based anchor detection ---")
    all_facts = set()
    for r in per_question:
        for f in r.get("top5_facts", []):
            all_facts.add(f["fact"])

    pattern_anchors = {f for f in all_facts if is_pattern_anchor(f)}
    print(f"Found {len(pattern_anchors)} pattern anchors")
    for f in sorted(pattern_anchors):
        print(f"  [P] {f[:100]}")

    # Step 3: Combine
    all_anchors = freq_anchors | pattern_anchors
    print(f"\n--- Combined: {len(all_anchors)} unique anchor edges ---")

    # Show all anchors sorted by frequency
    print("\nAll anchors (sorted by question frequency):")
    anchor_list = []
    for fact in all_anchors:
        detail = freq_details.get(fact, {})
        n_q = detail.get("n_questions", 0)
        avg_sem = detail.get("avg_semantic", 0)
        cr = detail.get("correct_rate", 0)
        is_pat = fact in pattern_anchors
        is_freq = fact in freq_anchors
        tags = []
        if is_pat:
            tags.append("P")
        if is_freq:
            tags.append(f"F:{n_q}q")
        anchor_list.append((fact, n_q, avg_sem, cr, tags))

    anchor_list.sort(key=lambda x: -x[1])
    for fact, n_q, avg_sem, cr, tags in anchor_list:
        tag_str = ",".join(tags)
        print(f"  [{tag_str:>8s}] sem={avg_sem:.3f} cr={cr:.2f} | {fact[:90]}")

    # Step 4: Simulate suppression
    print(f"\n{'=' * 70}")
    print("DRY-RUN SIMULATION: Remove anchors from top-10, re-evaluate")
    print(f"{'=' * 70}")

    sim = simulate_suppression(per_question, all_anchors, questions)

    new_passing = current_passing + len(sim["recovered"]) - len(sim["regressed"])
    print(f"\nRecovered: {len(sim['recovered'])}")
    for qid in sorted(sim["recovered"]):
        q = questions.get(qid, {})
        d = sim["details"][qid]
        print(f"  + {qid} ({q.get('attack_vector', '').split('_')[0]}) "
              f"removed {d['removed_anchors']} anchors")
        for t in d.get("new_top3", []):
            sc = "CORRECT" if t["supports_correct"] else ("STALE" if t["contains_wrong"] else "      ")
            print(f"    #{t['rank']} bl={t['blended']:.3f} [{sc}] {t['fact'][:80]}")

    print(f"\nRegressed: {len(sim['regressed'])}")
    for qid in sorted(sim["regressed"]):
        q = questions.get(qid, {})
        d = sim["details"][qid]
        print(f"  - {qid} ({q.get('attack_vector', '').split('_')[0]}) "
              f"removed {d['removed_anchors']} anchors")

    print(f"\nStill failing: {len(sim['unchanged_fail'])}")
    for qid in sorted(sim["unchanged_fail"]):
        d = sim["details"][qid]
        q = questions.get(qid, {})
        av = q.get("attack_vector", "").split("_")[0]
        print(f"  {qid} ({av}) removed {d['removed_anchors']} anchors, "
              f"top10: {d['old_top10_size']}->{d['new_top10_size']}")
        for t in d.get("new_top3", [])[:2]:
            sc = "CORRECT" if t["supports_correct"] else ("STALE" if t["contains_wrong"] else "      ")
            print(f"    #{t['rank']} bl={t['blended']:.3f} sem={t['semantic']:.3f} [{sc}] {t['fact'][:75]}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Anchors identified: {len(all_anchors)}")
    print(f"  Recovered:   +{len(sim['recovered'])}")
    print(f"  Regressed:   -{len(sim['regressed'])}")
    print(f"  Net gain:    {len(sim['recovered']) - len(sim['regressed']):+d}")
    print(f"  Old: {current_passing}/{total} ({current_passing/total:.1%})")
    print(f"  New: {new_passing}/{total} ({new_passing/total:.1%})")


if __name__ == "__main__":
    main()
