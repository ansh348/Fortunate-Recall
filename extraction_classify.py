"""
extraction_classify.py -- Verify extraction gaps for 20 questions.

For each question:
1. Search ALL edges for keyword matches (answer + broad)
2. Read the conversation session(s)
3. Classify: EXTRACTION_MISS | EDGE_TOO_VAGUE | COMPOSITE_GAP | CONVERSATION_GAP
"""

import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"

# Env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

from neo4j import AsyncGraphDatabase

# ---------------------------------------------------------------------------
PERSONA_DIRS = {
    "priya": "1_priya", "marcus": "2_marcus", "elena": "3_elena",
    "david": "4_david", "amara": "5_amara", "jake": "6_jake",
    "tom": "8_tom", "omar": "17_omar",
}
PERSONAS = {
    "priya": "lifemembench_priya", "marcus": "lifemembench_marcus",
    "elena": "lifemembench_elena", "david": "lifemembench_david",
    "amara": "lifemembench_amara", "jake": "lifemembench_jake",
    "tom": "lifemembench_tom", "omar": "lifemembench_omar",
}

# ---------------------------------------------------------------------------
# Target questions with search keywords and session numbers
# ---------------------------------------------------------------------------
TARGETS = {
    # --- E2 originals (4) ---
    "priya_q07": {
        "source": "E2-orig",
        "answer_kw": ["rock climbing", "climbing", "bouldering"],
        "broad_kw": ["rock climbing", "climbing", "bouldering", "quit yoga", "hot yoga"],
        "sessions": [25],
        "correct_answer": "Rock climbing (quit hot yoga, switched to rock climbing)",
    },
    "david_q04": {
        "source": "E2-orig",
        "answer_kw": ["abandoned the book", "abandoned", "nobody would read"],
        "broad_kw": ["book", "primary sources", "abandoned", "nobody would read"],
        "sessions": [9, 18],
        "correct_answer": "Abandoned book about teaching through primary sources",
    },
    "amara_q06": {
        "source": "E2-orig",
        "answer_kw": ["dropped the master", "dropped the llm", "not worth"],
        "broad_kw": ["llm", "ucl", "human rights", "master", "dropped", "not worth"],
        "sessions": [11, 17],
        "correct_answer": "Dropped UCL human rights LLM -- not worth time/money",
    },
    "amara_q14": {
        "source": "E2-orig",
        "answer_kw": ["18,000", "18000", "eighteen thousand"],
        "broad_kw": ["18,000", "18000", "tuition", "ucl", "llm cost"],
        "sessions": [11],
        "correct_answer": "UCL LLM would have cost GBP 18,000",
    },
    # --- E2 from Codex (1) ---
    "jake_q03": {
        "source": "E2-codex",
        "answer_kw": ["eire pub", "dorchester", "waitress"],
        "broad_kw": ["eire pub", "dorchester", "waitress", "mom works", "bartend"],
        "sessions": [3],
        "correct_answer": "Mom waitresses at Eire Pub in Dorchester, 20 years",
    },
    # --- E3-WEAK (15) ---
    "amara_q08": {
        "source": "E3-WEAK",
        "answer_kw": ["hackney", "islington", "considering move"],
        "broad_kw": ["hackney", "islington", "east london", "move"],
        "sessions": [1, 26],
        "correct_answer": "Lives in Hackney, considering move to Islington",
    },
    "amara_q11": {
        "source": "E3-WEAK",
        "answer_kw": ["boxing", "nigerian cooking", "podcasts"],
        "broad_kw": ["boxing", "nigerian cooking", "podcasts", "reading", "justice"],
        "sessions": [25],
        "correct_answer": "Boxing, Nigerian cooking, podcasts, reading about justice",
    },
    "david_q01": {
        "source": "E3-WEAK",
        "answer_kw": ["ap us history", "us history"],
        "broad_kw": ["ap us history", "us history", "ap euro", "switched"],
        "sessions": [23],
        "correct_answer": "Switching from AP Euro to AP US History",
    },
    "david_q11": {
        "source": "E3-WEAK",
        "answer_kw": ["parent-teacher conference", "parent teacher", "march 12"],
        "broad_kw": ["parent-teacher", "parent teacher", "conference", "march 12"],
        "sessions": [10],
        "correct_answer": "No upcoming (parent-teacher conferences March 12 already passed)",
    },
    "elena_q03": {
        "source": "E3-WEAK",
        "answer_kw": ["dropped", "np", "nurse practitioner", "can't afford"],
        "broad_kw": ["nurse practitioner", "np", "dropped", "student loan", "afford", "dnp", "depaul"],
        "sessions": [9, 17],
        "correct_answer": "Dropped NP plan -- can't afford with student loans",
    },
    "jake_q02": {
        "source": "E3-WEAK",
        "answer_kw": ["kayla", "dating kayla", "broke up with megan"],
        "broad_kw": ["kayla", "megan", "broke up", "nurse", "mass general", "dating"],
        "sessions": [24],
        "correct_answer": "Broke up with Megan, now dating Kayla (nurse at Mass General)",
    },
    "jake_q05": {
        "source": "E3-WEAK",
        "answer_kw": ["ev charger", "insurance too expensive", "dad said"],
        "broad_kw": ["ev charger", "ev home", "side gig", "insurance", "dad said", "not ready"],
        "sessions": [11, 16],
        "correct_answer": "EV charger side gig dead -- insurance too expensive",
    },
    "jake_q07": {
        "source": "E3-WEAK",
        "answer_kw": ["trillium", "craft beer", "ipa"],
        "broad_kw": ["trillium", "craft beer", "ipa", "bud light"],
        "sessions": [26],
        "correct_answer": "Got into craft beer -- favorites from Trillium Brewing",
    },
    "marcus_q03": {
        "source": "E3-WEAK",
        "answer_kw": ["scrapped", "germantown"],
        "broad_kw": ["germantown", "second location", "scrapped", "too much risk"],
        "sessions": [14, 21],
        "correct_answer": "Scrapped Germantown second location -- too much financial risk",
    },
    "omar_q06": {
        "source": "E3-WEAK",
        "answer_kw": ["ps5", "playstation", "never spends"],
        "broad_kw": ["ps5", "playstation", "never spends", "stress", "facebook marketplace"],
        "sessions": [7, 25],
        "correct_answer": "Said never spends on himself but bought PS5 for stress",
    },
    "omar_q12": {
        "source": "E3-WEAK",
        "answer_kw": ["century 21", "passed exam", "real estate license"],
        "broad_kw": ["real estate", "century 21", "exam", "passed", "brokerage", "license"],
        "sessions": [9, 29],
        "correct_answer": "Real estate career -- passed exam, joining Century 21",
    },
    "priya_q01": {
        "source": "E3-WEAK",
        "answer_kw": ["pescatarian", "eating fish again"],
        "broad_kw": ["pescatarian", "fish", "omega-3", "vegetarian", "eating fish"],
        "sessions": [22],
        "correct_answer": "Pescatarian (was vegetarian, now eats fish)",
    },
    "priya_q09": {
        "source": "E3-WEAK",
        "answer_kw": ["migraine", "omega-3", "stress"],
        "broad_kw": ["migraine", "omega-3", "omega", "screen", "headache", "stress"],
        "sessions": [4],
        "correct_answer": "Chronic migraines triggered by stress/screens; needs omega-3",
    },
    "tom_q06": {
        "source": "E3-WEAK",
        "answer_kw": ["barn conversion", "planning permission", "scrapped", "denied"],
        "broad_kw": ["barn", "honey processing", "planning permission", "scrapped", "expensive"],
        "sessions": [11, 17],
        "correct_answer": "Barn conversion scrapped -- planning permission denied, too expensive",
    },
    "tom_q07": {
        "source": "E3-WEAK",
        "answer_kw": ["instagram", "200 likes", "bee photo"],
        "broad_kw": ["instagram", "social media", "200 likes", "bee photo", "facebook"],
        "sessions": [9, 29],
        "correct_answer": "Said no social media, but posted bee photo on Instagram (200 likes)",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_session_text(persona: str, session_num: int) -> str:
    d = PERSONA_DIRS[persona]
    p = LIFEMEMEVAL_DIR / d / "sessions" / f"session_{session_num:02d}.json"
    if not p.exists():
        return ""
    data = json.load(open(p, encoding="utf-8"))
    turns = data.get("turns", [])
    return "\n".join(t.get("content", "") for t in turns)


def extract_snippet(text: str, keyword: str, context: int = 200) -> str:
    idx = text.lower().find(keyword.lower())
    if idx < 0:
        return ""
    start = max(0, idx - context)
    end = min(len(text), idx + len(keyword) + context)
    return text[start:end].replace("\n", " ").strip()


async def load_all_edges_for_persona(driver, group_id: str) -> list[dict]:
    result = await driver.execute_query(
        """
        MATCH (s)-[e:RELATES_TO]->(t)
        WHERE e.group_id = $gid
        RETURN e.uuid AS uuid,
               e.fact AS fact,
               e.fr_enriched AS enriched,
               e.fr_primary_category AS category,
               e.expired_at AS expired_at,
               e.created_at AS created_at
        """,
        gid=group_id,
    )
    records = result.records if hasattr(result, "records") else result
    return [rec.data() if hasattr(rec, "data") else dict(rec) for rec in records]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    questions = {q["id"]: q for q in json.load(open(LIFEMEMEVAL_DIR / "lifemembench_questions.json"))}

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    # Pre-load all edges per persona
    persona_edges = {}
    for persona, gid in PERSONAS.items():
        edges = await load_all_edges_for_persona(driver, gid)
        persona_edges[persona] = edges
        print(f"  Loaded {len(edges)} edges for {persona}")

    await driver.close()

    print(f"\n{'=' * 100}")
    print("EXTRACTION GAP CLASSIFICATION")
    print(f"{'=' * 100}\n")

    classifications = {}

    for qid in sorted(TARGETS.keys()):
        t = TARGETS[qid]
        q = questions.get(qid, {})
        persona = qid.rsplit("_q", 1)[0]
        edges = persona_edges.get(persona, [])
        source = t["source"]

        print(f"\n{'=' * 100}")
        print(f"{qid} [{source}]")
        print(f"  Q: {q.get('question', '?')}")
        print(f"  A: {t['correct_answer']}")
        print(f"  AV: {q.get('attack_vector', '?')}")

        # Search edges with answer keywords
        answer_matches = []
        for edge in edges:
            fact = (edge.get("fact") or "").lower()
            matched = [kw for kw in t["answer_kw"] if kw.lower() in fact]
            if matched:
                answer_matches.append({**edge, "matched": matched})

        # Search edges with broad keywords
        broad_matches = []
        for edge in edges:
            fact = (edge.get("fact") or "").lower()
            matched = [kw for kw in t["broad_kw"] if kw.lower() in fact]
            if matched:
                broad_matches.append({**edge, "matched": matched})

        print(f"\n  EDGES: {len(answer_matches)} answer-keyword matches, {len(broad_matches)} broad matches")

        if answer_matches:
            print(f"  Answer-keyword edge matches:")
            for e in answer_matches[:5]:
                exp = " [EXPIRED]" if e.get("expired_at") else ""
                enr = " [enriched]" if e.get("enriched") else " [not-enriched]"
                cat = e.get("category") or "?"
                print(f"    [{e['uuid'][:8]}] {cat:<25s}{enr}{exp}")
                print(f"      {(e.get('fact') or '')[:120]}")
                print(f"      matched: {e['matched']}")

        if broad_matches and not answer_matches:
            print(f"  Broad-keyword edge matches (no answer match):")
            for e in broad_matches[:5]:
                exp = " [EXPIRED]" if e.get("expired_at") else ""
                enr = " [enriched]" if e.get("enriched") else " [not-enriched]"
                cat = e.get("category") or "?"
                print(f"    [{e['uuid'][:8]}] {cat:<25s}{enr}{exp}")
                print(f"      {(e.get('fact') or '')[:120]}")
                print(f"      matched: {e['matched']}")

        # Read conversation sessions
        print(f"\n  CONVERSATION CHECK (sessions {t['sessions']}):")
        conv_found = False
        conv_snippets = []
        for sess in t["sessions"]:
            text = read_session_text(persona, sess)
            if not text:
                print(f"    session_{sess:02d}: FILE NOT FOUND")
                continue
            # Check each answer keyword
            found_kws = []
            for kw in t["answer_kw"]:
                if kw.lower() in text.lower():
                    found_kws.append(kw)
                    snippet = extract_snippet(text, kw, 150)
                    conv_snippets.append((sess, kw, snippet))
            if found_kws:
                conv_found = True
                print(f"    session_{sess:02d}: FOUND [{', '.join(found_kws)}]")
                for _, kw, snip in conv_snippets[-len(found_kws):]:
                    print(f"      ...{snip[:200]}...")
            else:
                # Try broad keywords
                broad_found = [kw for kw in t["broad_kw"] if kw.lower() in text.lower()]
                if broad_found:
                    conv_found = True
                    print(f"    session_{sess:02d}: broad match [{', '.join(broad_found[:3])}]")
                    snippet = extract_snippet(text, broad_found[0], 150)
                    print(f"      ...{snippet[:200]}...")
                else:
                    print(f"    session_{sess:02d}: no keyword matches")

        # Classify
        has_answer_edge = len(answer_matches) > 0
        has_broad_edge = len(broad_matches) > 0
        multi_session = len(t["sessions"]) >= 2

        # Check if answer edge is too vague (E3-WEAK indicator)
        is_e3_weak = source == "E3-WEAK"
        is_e2 = source in ("E2-orig", "E2-codex")

        if not conv_found and not has_answer_edge and not has_broad_edge:
            classification = "CONVERSATION_GAP"
            reason = "Fact not found in conversation AND no edge exists"
        elif has_answer_edge and is_e3_weak:
            # Edge exists but was judged insufficient by Codex
            # Check: is the edge just missing a detail, or is the answer truly composite?
            if multi_session and len(set(kw for e in broad_matches for kw in e["matched"])) >= 3:
                classification = "COMPOSITE_GAP"
                reason = f"Answer needs facts from {len(t['sessions'])} sessions; edges are partial"
            else:
                classification = "EDGE_TOO_VAGUE"
                reason = "Edge exists but too vague to fully answer question"
        elif has_broad_edge and not has_answer_edge:
            if conv_found:
                classification = "EXTRACTION_MISS"
                reason = "Fact in conversation, only topic-level edges exist (no specific answer edge)"
            else:
                classification = "EDGE_TOO_VAGUE"
                reason = "Only broad/topic edges exist, specific detail not captured"
        elif not has_answer_edge and not has_broad_edge:
            if conv_found:
                classification = "EXTRACTION_MISS"
                reason = "Fact clearly in conversation but no edge extracted at all"
            else:
                classification = "CONVERSATION_GAP"
                reason = "No matching edges and fact not found in listed sessions"
        elif has_answer_edge and is_e2:
            # Originally classified E2 but we found an answer edge -- reclassify
            classification = "EDGE_TOO_VAGUE"
            reason = "Answer edge exists but was considered insufficient (originally E2)"
        else:
            classification = "EDGE_TOO_VAGUE"
            reason = "Edge exists but classified as insufficient by review"

        # Special cases for multi-session composite answers
        composite_qids = {"amara_q11", "omar_q12", "omar_q06", "tom_q07", "jake_q05", "elena_q03",
                          "marcus_q03", "tom_q06"}
        if qid in composite_qids and multi_session:
            # Check if the answer requires combining state changes across sessions
            classification = "COMPOSITE_GAP"
            reason = f"Answer requires combining facts across sessions {t['sessions']}"

        classifications[qid] = {
            "classification": classification,
            "reason": reason,
            "source": source,
            "answer_edges": len(answer_matches),
            "broad_edges": len(broad_matches),
            "conv_found": conv_found,
        }

        print(f"\n  --> {classification}: {reason}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n\n{'=' * 100}")
    print("SUMMARY TABLE")
    print(f"{'=' * 100}")
    print(f"\n{'QID':<16s} {'ORIG':<10s} {'CLASS':<18s} {'#ANS':>5s} {'#BRD':>5s} {'CONV':>5s} REASON")
    print("-" * 100)

    for qid in sorted(classifications.keys()):
        c = classifications[qid]
        print(f"  {qid:<16s} {c['source']:<10s} {c['classification']:<18s} "
              f"{c['answer_edges']:>5d} {c['broad_edges']:>5d} "
              f"{'Y' if c['conv_found'] else 'N':>5s} {c['reason'][:60]}")

    print(f"\n{'=' * 100}")
    print("COUNTS")
    print(f"{'=' * 100}")
    counts = Counter(c["classification"] for c in classifications.values())
    for cls in ["EXTRACTION_MISS", "EDGE_TOO_VAGUE", "COMPOSITE_GAP", "CONVERSATION_GAP"]:
        cnt = counts.get(cls, 0)
        pct = cnt / len(classifications) * 100
        print(f"  {cls:<18s}: {cnt:3d} ({pct:.0f}%)")
        qids = [qid for qid, c in classifications.items() if c["classification"] == cls]
        for qid in sorted(qids):
            print(f"    - {qid}")
    print(f"  {'TOTAL':<18s}: {len(classifications):3d}")


if __name__ == "__main__":
    asyncio.run(main())
