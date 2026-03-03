"""
diagnose_av_failures.py — Root-cause diagnostic for LifeMemBench AV failures.

For each question where av_pass=False in a given config (default: "full"),
prints the top-5 edges with verdicts and supersession status, then classifies
each failure as:

  filter_bug    — stale edge IS marked superseded but still in top-5
  detector_gap  — stale edge is NOT marked superseded (detector missed it)
  retrieval_gap — no stale edge in top-5, just missing correct edge

Usage:
    python diagnose_av_failures.py                       # all failures in "full" config
    python diagnose_av_failures.py --persona priya       # one persona
    python diagnose_av_failures.py --av AV1              # one attack-vector prefix
    python diagnose_av_failures.py --config no_routing   # different config
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from neo4j import AsyncGraphDatabase

PROJECT_ROOT = Path(__file__).parent
LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
ARTIFACTS_DIR = LIFEMEMEVAL_DIR / "artifacts"
RESULTS_PATH = ARTIFACTS_DIR / "lifemembench_results.json"
JUDGE_CACHE_PATH = ARTIFACTS_DIR / "lifemembench_judge_cache.json"
QUESTIONS_PATH = LIFEMEMEVAL_DIR / "lifemembench_questions.json"

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
# Environment
# ---------------------------------------------------------------------------

def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results() -> dict:
    if not RESULTS_PATH.exists():
        print(f"ERROR: Results file not found: {RESULTS_PATH}")
        sys.exit(1)
    return json.load(open(RESULTS_PATH, encoding="utf-8"))


def load_judge_cache() -> dict[str, dict[str, dict]]:
    """Load judge cache and build question_id → {edge_uuid → verdict} index."""
    if not JUDGE_CACHE_PATH.exists():
        print(f"ERROR: Judge cache not found: {JUDGE_CACHE_PATH}")
        sys.exit(1)
    raw = json.load(open(JUDGE_CACHE_PATH, encoding="utf-8"))

    index: dict[str, dict[str, dict]] = defaultdict(dict)
    for _key, entry in raw.items():
        qid = entry.get("question_id")
        uuid = entry.get("edge_uuid")
        if qid and uuid:
            index[qid][uuid] = {
                "supports_correct": entry.get("supports_correct", False),
                "contains_wrong_indicator": entry.get("contains_wrong_indicator", False),
                "reasoning": entry.get("reasoning", ""),
            }
    return dict(index)


def load_questions() -> dict[str, dict]:
    """Load questions JSON, return question_id → question_dict."""
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: Questions file not found: {QUESTIONS_PATH}")
        sys.exit(1)
    raw = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    return {q["id"]: q for q in raw}


async def load_edge_data(driver, group_ids: list[str]) -> dict[str, dict]:
    """Load edge fact text and supersession fields from Neo4j."""
    result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id IN $gids AND e.fr_enriched = true
        RETURN e.uuid AS uuid,
               e.fact AS fact,
               e.fr_superseded_by AS superseded_by,
               e.fr_supersession_confidence AS supersession_confidence,
               e.fr_supersession_reason AS supersession_reason
        """,
        gids=group_ids,
    )
    data = {}
    records = result.records if hasattr(result, "records") else result
    for rec in records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        uuid = d.pop("uuid")
        data[uuid] = d
    return data


# ---------------------------------------------------------------------------
# UUID resolution
# ---------------------------------------------------------------------------

def resolve_top5_uuids(
    top5_facts: list[dict],
    question_id: str,
    judge_index: dict[str, dict[str, dict]],
    edge_data: dict[str, dict],
) -> list[dict]:
    """Match each top5 entry to its edge UUID via judge cache + Neo4j fact text."""
    judged = judge_index.get(question_id, {})

    # Build fact_prefix → uuid mapping for edges judged for this question
    prefix_map: dict[str, str] = {}
    for uuid in judged:
        ed = edge_data.get(uuid)
        if ed and ed.get("fact"):
            prefix = ed["fact"][:120]
            prefix_map[prefix] = uuid

    resolved = []
    for entry in top5_facts:
        uuid = prefix_map.get(entry["fact"])
        sup = edge_data.get(uuid, {}) if uuid else {}
        resolved.append({
            "rank": entry["rank"],
            "uuid": uuid or "???",
            "fact": entry["fact"],
            "supports_correct": entry.get("supports_correct", False),
            "contains_wrong": entry.get("contains_wrong", False),
            "superseded_by": sup.get("superseded_by"),
            "supersession_confidence": sup.get("supersession_confidence"),
            "supersession_reason": sup.get("supersession_reason"),
        })
    return resolved


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_failure(top5_resolved: list[dict]) -> str:
    """Classify a failing question's root cause."""
    stale_edges = [e for e in top5_resolved if e["contains_wrong"]]

    if not stale_edges:
        return "retrieval_gap"

    superseded_stale = [e for e in stale_edges if e["superseded_by"]]
    if superseded_stale:
        return "filter_bug"
    return "detector_gap"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_question_detail(
    question: dict,
    top5_resolved: list[dict],
    classification: str,
):
    label = {
        "filter_bug": "FILTER BUG",
        "detector_gap": "DETECTOR GAP",
        "retrieval_gap": "RETRIEVAL GAP",
    }[classification]

    qid = question["id"]
    av = question["attack_vector"]
    print(f"\n{'='*70}")
    print(f"FAIL  {qid}  {av}  [{label}]")
    print(f"Q: {question['question']}")
    print(f"Correct: {question['correct_answer']}")
    wrong = question.get("wrong_answer_indicators", [])
    print(f"Wrong indicators: {', '.join(wrong) if wrong else '(none)'}")
    print(f"{'-'*70}")

    for e in top5_resolved:
        # Verdict symbol
        if e["supports_correct"] and e["contains_wrong"]:
            sym = "~BOTH"
        elif e["supports_correct"]:
            sym = "+correct"
        elif e["contains_wrong"]:
            sym = "xWRONG"
        else:
            sym = "-"

        uuid_short = e["uuid"][:12] if e["uuid"] != "???" else "???"
        print(f"  #{e['rank']}  {sym:>8s}  uuid={uuid_short}..  {e['fact'][:80]}")

        # Supersession status
        if e["superseded_by"]:
            conf = e["supersession_confidence"] or 0
            reason = (e["supersession_reason"] or "")[:60]
            print(f"       superseded: YES (conf={conf:.2f}) {reason}")
            if e["contains_wrong"]:
                print(f"       ^^ FILTER SHOULD HAVE CAUGHT THIS")
        else:
            if e["contains_wrong"]:
                print(f"       superseded: NO  <-- DETECTOR MISSED THIS")
            else:
                print(f"       superseded: NO")


def print_summary(classifications: list[dict]):
    total = len(classifications)
    counts = Counter(c["classification"] for c in classifications)

    print(f"\n{'='*55}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*55}")
    print(f"Total AV failures:                          {total:>3d}")
    print(f"  Stale edge superseded, still in top-5:    {counts.get('filter_bug', 0):>3d}  (filter bug)")
    print(f"  Stale edge NOT superseded:                {counts.get('detector_gap', 0):>3d}  (detector gap)")
    print(f"  No stale edge, missing correct:           {counts.get('retrieval_gap', 0):>3d}  (retrieval gap)")

    # By attack vector
    av_groups: dict[str, list[str]] = defaultdict(list)
    for c in classifications:
        av_groups[c["attack_vector"]].append(c["classification"])

    print(f"\nBy attack vector:")
    for av in sorted(av_groups):
        cats = Counter(av_groups[av])
        parts = ", ".join(f"{n} {k}" for k, n in cats.most_common())
        print(f"  {av:<35s} {len(av_groups[av]):>2d} failures ({parts})")

    # By persona
    persona_groups: dict[str, list[str]] = defaultdict(list)
    for c in classifications:
        persona_groups[c["persona"]].append(c["classification"])

    print(f"\nBy persona:")
    for p in sorted(persona_groups):
        cats = Counter(persona_groups[p])
        parts = ", ".join(f"{n} {k}" for k, n in cats.most_common())
        print(f"  {p:<15s} {len(persona_groups[p]):>2d} failures ({parts})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    config_name = args.config
    results = load_results()
    judge_index = load_judge_cache()
    questions = load_questions()

    per_question = results.get("per_question", {}).get(config_name)
    if per_question is None:
        print(f"ERROR: Config '{config_name}' not found in results. "
              f"Available: {list(results.get('per_question', {}).keys())}")
        sys.exit(1)

    # Filter to failing questions
    failures = [q for q in per_question if not q["av_pass"]]
    if args.persona:
        failures = [q for q in failures if q["question_id"].startswith(args.persona)]
    if args.av:
        failures = [q for q in failures if q["attack_vector"].startswith(args.av)]

    print(f"Config: {config_name}")
    print(f"Total questions: {len(per_question)}")
    print(f"Total AV failures: {len([q for q in per_question if not q['av_pass']])}")
    print(f"Analyzing: {len(failures)} failures"
          + (f" (filtered: persona={args.persona})" if args.persona else "")
          + (f" (filtered: av={args.av})" if args.av else ""))

    # Determine which group_ids we need
    needed_personas = set()
    for f in failures:
        # Extract persona name from question_id (e.g., "priya_q01" → "priya")
        persona = f["question_id"].rsplit("_q", 1)[0]
        needed_personas.add(persona)

    group_ids = [PERSONAS[p] for p in needed_personas if p in PERSONAS]
    if not group_ids:
        print("No matching personas found.")
        return

    # Connect to Neo4j
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    try:
        print(f"\nLoading edge data from Neo4j for {len(group_ids)} persona(s)...")
        edge_data = await load_edge_data(driver, group_ids)
        print(f"Loaded {len(edge_data)} edges")

        # Count superseded edges
        n_superseded = sum(1 for e in edge_data.values() if e.get("superseded_by"))
        print(f"Edges marked superseded: {n_superseded}")

        classifications = []

        for f in failures:
            qid = f["question_id"]
            q = questions.get(qid)
            if not q:
                print(f"WARNING: Question {qid} not found in questions JSON, skipping")
                continue

            top5_resolved = resolve_top5_uuids(
                f.get("top5_facts", []), qid, judge_index, edge_data
            )
            classification = classify_failure(top5_resolved)

            classifications.append({
                "question_id": qid,
                "attack_vector": f["attack_vector"],
                "persona": qid.rsplit("_q", 1)[0],
                "classification": classification,
            })

            print_question_detail(q, top5_resolved, classification)

        if classifications:
            print_summary(classifications)
        else:
            print("\nNo failures to analyze!")

    finally:
        await driver.close()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose LifeMemBench AV failures: identify filter bugs, "
                    "detector gaps, and retrieval gaps."
    )
    parser.add_argument("--config", default="full",
                        help="Which config to analyze (default: full)")
    parser.add_argument("--persona", default=None,
                        help="Filter to one persona (e.g., priya)")
    parser.add_argument("--av", default=None,
                        help="Filter to one AV prefix (e.g., AV1)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
