"""
audit_judge_verdicts.py — Trace why correct-in-top-5 edges are still MISS.

For 5 MISS questions where correct edges ARE in top-5 (Q05, Q10, Q11, Q12, Q14),
traces through:
  1. prefilter_for_correct: does the edge pass keyword pre-filtering?
  2. prefilter_for_wrong: does it match wrong indicators?
  3. judge cache: what verdict did the LLM return (if any)?

Usage: python audit_judge_verdicts.py
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

# Import prefilter functions and utilities from evaluate_lifemembench
from evaluate_lifemembench import (
    prefilter_for_correct,
    prefilter_for_wrong,
    extract_keywords,
    extract_numbers,
    QUESTIONS_PATH,
    JUDGE_CACHE_PATH,
)
from neo4j import AsyncGraphDatabase

# The 5 MISS questions where correct edges are in top-5
OK_MISS_IDS = ["priya_q05", "priya_q10", "priya_q11", "priya_q12", "priya_q14"]

# Keywords to identify correct-answer edges (from audit_extraction_ceiling.py)
MISS_KEYWORDS = {
    "priya_q05": ["dog", "landlord", "pet", "golden retriever"],
    "priya_q10": ["neurips", "paper deadline", "june 2026"],
    "priya_q11": ["rock climbing", "climbing", "tamil", "cooking", "travel"],
    "priya_q12": ["tamil", "chennai", "arjun", "seattle", "brother"],
    "priya_q14": ["tokyo", "march 20", "march 2026", "conference"],
}


def trace_prefilter_correct(question: str, correct_answer: str, fact: str):
    """Verbose trace of prefilter_for_correct logic."""
    fact_lower = fact.lower()
    q_kws = extract_keywords(question)
    a_kws = extract_keywords(correct_answer)

    q_hits = [kw for kw in q_kws if kw in fact_lower]
    a_hits = [kw for kw in a_kws if kw in fact_lower]

    a_numbers = extract_numbers(correct_answer)
    f_numbers = extract_numbers(fact)
    number_match = bool(a_numbers and set(a_numbers) & set(f_numbers))

    print(f"      q_kws = {q_kws}")
    print(f"      a_kws = {a_kws}")
    print(f"      q_hits = {q_hits} ({len(q_hits)})")
    print(f"      a_hits = {a_hits} ({len(a_hits)})")
    if a_numbers or f_numbers:
        print(f"      a_numbers = {a_numbers}, f_numbers = {f_numbers}, match = {number_match}")

    # Check conditions in order
    if len(q_hits) >= 1 and len(a_hits) >= 1:
        print(f"      -> PASS (q>=1 AND a>=1)")
        return True
    if len(q_hits) >= 2:
        print(f"      -> PASS (q>=2)")
        return True
    if len(q_hits) >= 1 and number_match:
        print(f"      -> PASS (q>=1 AND number_match)")
        return True
    if len(a_kws) <= 2 and a_kws:
        import re
        for kw in a_kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", fact_lower) and q_hits:
                print(f"      -> PASS (short answer whole-word '{kw}')")
                return True

    reasons = []
    if len(q_hits) == 0:
        reasons.append("q_hits=0 (no question keywords in fact)")
    elif len(q_hits) == 1 and len(a_hits) == 0:
        reasons.append(f"q_hits=1 but a_hits=0 (need both)")
    if not number_match and a_numbers:
        reasons.append("number mismatch")
    print(f"      -> SKIP: {'; '.join(reasons)}")
    return False


def trace_prefilter_wrong(wrong_indicators: list[str], fact: str):
    """Verbose trace of prefilter_for_wrong logic."""
    if not wrong_indicators:
        print(f"      (no wrong indicators)")
        return False

    fact_lower = fact.lower()
    for indicator in wrong_indicators:
        indicator_lower = indicator.lower()
        if indicator_lower in fact_lower:
            print(f"      -> MATCH: '{indicator}' found in fact")
            return True
        indicator_kws = extract_keywords(indicator)
        if indicator_kws:
            hits = sum(1 for kw in indicator_kws if kw in fact_lower)
            threshold = max(1, len(indicator_kws) // 2)
            if hits >= threshold:
                matched = [kw for kw in indicator_kws if kw in fact_lower]
                print(f"      -> MATCH: indicator '{indicator}' kw overlap {matched} ({hits}>={threshold})")
                return True

    print(f"      -> no match")
    return False


async def main():
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    priya_qs = {q["id"]: q for q in questions if q["persona"] == "priya"}

    # Load judge cache
    judge_cache = {}
    if JUDGE_CACHE_PATH.exists():
        judge_cache = json.load(open(JUDGE_CACHE_PATH, encoding="utf-8"))

    # Connect to Neo4j to get all edges
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

    driver = AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "testpassword123"),
        ),
    )

    result = await driver.execute_query(
        """
        MATCH (s)-[e:RELATES_TO]->(t)
        WHERE e.group_id = $gid
        RETURN e.uuid AS uuid, e.fact AS fact
        """,
        gid="lifemembench_priya",
    )
    records = result.records if hasattr(result, "records") else result
    all_edges = {r.data()["uuid"]: r.data()["fact"] for r in records}

    print(f"=== JUDGE VERDICT AUDIT: 5 OK-MISS QUESTIONS ===")
    print(f"Judge cache entries: {len(judge_cache)}")
    print(f"Total edges: {len(all_edges)}\n")

    prefilter_skip_count = 0
    judge_reject_count = 0
    judge_accept_count = 0

    for qid in OK_MISS_IDS:
        q = priya_qs[qid]
        keywords = MISS_KEYWORDS[qid]

        print(f"{'='*70}")
        print(f"{qid} ({q['attack_vector']}): {q['question']}")
        print(f"  correct_answer: {q['correct_answer']}")
        print(f"  wrong_indicators: {q.get('wrong_answer_indicators', [])}")
        print()

        # Find correct-answer edges
        correct_edges = {}
        for uuid, fact in all_edges.items():
            fact_lower = fact.lower()
            matched = [kw for kw in keywords if kw.lower() in fact_lower]
            if matched:
                correct_edges[uuid] = (fact, matched)

        for uuid, (fact, matched_kws) in correct_edges.items():
            print(f"  EDGE [{uuid[:8]}] ({', '.join(matched_kws)})")
            print(f"    \"{fact[:120]}\"")

            # Trace prefilter_for_correct
            print(f"    prefilter_for_correct:")
            pf_correct = trace_prefilter_correct(q["question"], q["correct_answer"], fact)

            # Trace prefilter_for_wrong
            print(f"    prefilter_for_wrong:")
            pf_wrong = trace_prefilter_wrong(q.get("wrong_answer_indicators", []), fact)

            needs_judge = pf_correct or pf_wrong

            # Check judge cache
            cache_key = hashlib.sha256(f"{qid}||{uuid}".encode()).hexdigest()[:16]
            if cache_key in judge_cache:
                entry = judge_cache[cache_key]
                sc = entry["supports_correct"]
                cwi = entry["contains_wrong_indicator"]
                reasoning = entry["reasoning"]
                print(f"    JUDGE VERDICT: supports_correct={sc}, contains_wrong={cwi}")
                print(f"      reasoning: \"{reasoning}\"")
                if sc:
                    judge_accept_count += 1
                else:
                    judge_reject_count += 1
            elif needs_judge:
                print(f"    JUDGE: passed prefilter but NOT IN CACHE (would be judged on next run)")
            else:
                print(f"    JUDGE: NOT IN CACHE (prefilter skipped — never sent to judge)")
                prefilter_skip_count += 1

            print()

    print(f"{'='*70}")
    print(f"SUMMARY across all correct-answer edges:")
    print(f"  Prefilter skipped (never judged): {prefilter_skip_count}")
    print(f"  Judge rejected (supports_correct=false): {judge_reject_count}")
    print(f"  Judge accepted (supports_correct=true): {judge_accept_count}")

    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
