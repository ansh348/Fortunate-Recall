"""
audit_extraction_ceiling.py — Check if correct answers exist in Neo4j edges.

For each MISS question, searches ALL edges for keyword matches against the
correct answer. Reports RETRIEVABLE vs EXTRACTION_CEILING per question.

Usage: python audit_extraction_ceiling.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
QUESTIONS_PATH = PROJECT_ROOT / "LifeMemEval" / "lifemembench_questions.json"

# --- Env ---
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

from neo4j import AsyncGraphDatabase

# --- Search keywords per MISS question ---
MISS_KEYWORDS = {
    "priya_q01": ["pescatarian", "fish", "vegetarian"],
    "priya_q03": ["adhd", "attention deficit", "attention"],
    "priya_q04": ["tokyo", "march 20", "conference"],
    "priya_q05": ["dog", "landlord", "pet", "golden retriever"],
    "priya_q07": ["rock climbing", "climbing", "bouldering"],
    "priya_q09": ["migraine", "omega-3", "omega"],
    "priya_q10": ["neurips", "paper deadline", "june 2026"],
    "priya_q11": ["rock climbing", "climbing", "tamil", "cooking", "travel"],
    "priya_q12": ["tamil", "chennai", "arjun", "seattle", "brother"],
    "priya_q14": ["tokyo", "march 20", "march 2026", "conference"],
}


async def main():
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    priya_qs = {q["id"]: q for q in questions if q["persona"] == "priya"}

    driver = AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "testpassword123"),
        ),
    )

    # Fetch ALL edges
    result = await driver.execute_query(
        """
        MATCH (s)-[e:RELATES_TO]->(t)
        WHERE e.group_id = $gid
        RETURN e.uuid AS uuid, e.fact AS fact
        """,
        gid="lifemembench_priya",
    )
    records = result.records if hasattr(result, "records") else result
    edges = [(r.data()["uuid"], r.data()["fact"]) for r in records]

    print(f"=== EXTRACTION CEILING AUDIT: PRIYA ===")
    print(f"Total edges: {len(edges)}\n")

    retrievable = 0
    ceiling = 0

    for qid, keywords in MISS_KEYWORDS.items():
        q = priya_qs[qid]
        print(f"{qid} ({q['attack_vector'][:3]}): {q['question']}")
        print(f"  Correct: {q['correct_answer']}")
        print(f"  Keywords: {', '.join(keywords)}")

        matches = []
        for uuid, fact in edges:
            fact_lower = fact.lower()
            matched_kws = [kw for kw in keywords if kw.lower() in fact_lower]
            if matched_kws:
                matches.append((uuid, fact, matched_kws))

        if matches:
            print(f"  MATCHES ({len(matches)}):")
            for uuid, fact, kws in matches:
                kw_str = ", ".join(kws)
                print(f"    [{uuid[:8]}] ({kw_str}) \"{fact[:120]}\"")
            print(f"  VERDICT: RETRIEVABLE\n")
            retrievable += 1
        else:
            print(f"  MATCHES (0)")
            print(f"  VERDICT: *** EXTRACTION_CEILING ***\n")
            ceiling += 1

    print(f"{'='*50}")
    print(f"SUMMARY: {retrievable} retrievable, {ceiling} extraction ceiling")
    print(f"  Ceiling rate: {ceiling}/{retrievable + ceiling} "
          f"({ceiling / (retrievable + ceiling) * 100:.0f}%)")

    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
