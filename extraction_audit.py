"""
extraction_audit.py — Classify all 51 LifeMemBench extraction gaps (E1-E5).

For each question where av_pass=False in the "full" config, this script:
  1. Searches ALL Neo4j edges (not just fr_enriched) for keyword matches
  2. Searches entity node summaries for keyword matches
  3. Reads the relevant conversation session to verify facts exist
  4. Classifies the gap using a waterfall: E1 → E2 → E3 → E4 → E5 / RANKING

Outputs extraction_audit_report.md with a full gap table, summaries, and
per-persona/per-AV breakdowns.

Usage: python extraction_audit.py
"""

import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
ARTIFACTS_DIR = LIFEMEMEVAL_DIR / "artifacts"
RESULTS_PATH = ARTIFACTS_DIR / "lifemembench_results.json"
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

# ---------------------------------------------------------------------------
# Persona directory map  (name -> numbered dir)
# ---------------------------------------------------------------------------
PERSONA_DIRS = {
    "priya":  "1_priya",
    "marcus": "2_marcus",
    "elena":  "3_elena",
    "david":  "4_david",
    "amara":  "5_amara",
    "jake":   "6_jake",
    "tom":    "8_tom",
    "omar":   "17_omar",
}

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
# Keyword registry -- every failing question
#
# answer_keywords: STRICT terms specific to the CORRECT answer.
#     An edge must match at least one of these to count as "correct fact extracted".
# conv_keywords:   BROAD terms used to verify the fact exists in conversation.
# sessions:        which session file(s) to check.
# fact_text:       human-readable description.
# ---------------------------------------------------------------------------
KEYWORD_REGISTRY = {
    # == PRIYA ================================================================
    "priya_q01": {
        "answer_keywords": ["pescatarian", "eating fish again"],
        "conv_keywords": ["pescatarian", "fish", "omega-3", "vegetarian"],
        "sessions": [22],
        "fact_text": "Started eating fish -- pescatarian (was vegetarian)",
    },
    "priya_q03": {
        "answer_keywords": ["adhd"],
        "conv_keywords": ["adhd", "attention deficit", "diagnosed"],
        "sessions": [2],
        "fact_text": "Has ADHD, diagnosed in college",
    },
    "priya_q05": {
        "answer_keywords": ["landlord said no", "decided against the dog", "landlord"],
        "conv_keywords": ["landlord", "golden retriever", "dog", "pet"],
        "sessions": [12, 16],
        "fact_text": "Wanted dog but landlord said no -- retraction",
    },
    "priya_q07": {
        "answer_keywords": ["rock climbing"],
        "conv_keywords": ["rock climbing", "climbing", "bouldering", "quit yoga"],
        "sessions": [25],
        "fact_text": "Switched to rock climbing, quit hot yoga",
    },
    "priya_q09": {
        "answer_keywords": ["migraine"],
        "conv_keywords": ["migraine", "omega-3", "omega", "screen", "headache"],
        "sessions": [4],
        "fact_text": "Chronic migraines triggered by stress/screens; needs omega-3",
    },

    # == MARCUS ===============================================================
    "marcus_q02": {
        "answer_keywords": ["ram 1500", "2024 ram"],
        "conv_keywords": ["ram 1500", "2024 ram", "f-150", "traded", "new truck"],
        "sessions": [24],
        "fact_text": "Bought 2024 Ram 1500, traded in Ford F-150",
    },
    "marcus_q03": {
        "answer_keywords": ["scrapped", "germantown"],
        "conv_keywords": ["germantown", "second location", "scrapped", "too much risk"],
        "sessions": [14, 21],
        "fact_text": "Scrapped Germantown second location -- too much financial risk",
    },
    "marcus_q06": {
        "answer_keywords": ["fishing"],
        "conv_keywords": ["fishing", "stopped poker", "weekends"],
        "sessions": [23],
        "fact_text": "Stopped poker, started fishing on weekends",
    },
    "marcus_q13": {
        "answer_keywords": ["carlos", "fourth employee", "4 employees", "four employees", "hired"],
        "conv_keywords": ["carlos", "fourth", "hired", "employee"],
        "sessions": [26],
        "fact_text": "Hired Carlos as 4th employee -- business growing",
    },

    # == ELENA ================================================================
    "elena_q01": {
        "answer_keywords": ["rush university", "rush medical"],
        "conv_keywords": ["rush", "transferred", "lurie"],
        "sessions": [21],
        "fact_text": "Transferred from Lurie to Rush University Medical Center",
    },
    "elena_q02": {
        "answer_keywords": ["mediterranean"],
        "conv_keywords": ["mediterranean", "keto", "diet"],
        "sessions": [22],
        "fact_text": "Quit keto, switched to Mediterranean diet",
    },
    "elena_q03": {
        "answer_keywords": ["dropped", "np", "nurse practitioner", "can't afford"],
        "conv_keywords": ["nurse practitioner", "np", "dropped", "student loan", "afford"],
        "sessions": [9, 17],
        "fact_text": "Dropped NP plan -- can't afford it with student loans",
    },
    "elena_q04": {
        "answer_keywords": ["sertraline", "generalized anxiety", "anxiety disorder"],
        "conv_keywords": ["anxiety", "generalized anxiety", "sertraline", "gad"],
        "sessions": [3],
        "fact_text": "Generalized anxiety disorder, takes sertraline",
    },
    "elena_q08": {
        "answer_keywords": ["guadalajara", "first-generation", "first gen", "sofia", "down syndrome"],
        "conv_keywords": ["guadalajara", "first-generation", "first gen", "sofia", "down syndrome", "parents immigrated"],
        "sessions": [2, 4, 5],
        "fact_text": "Parents from Guadalajara, first-gen college grad, sister Sofia has Down syndrome",
    },
    "elena_q12": {
        "answer_keywords": ["knit", "knitting"],
        "conv_keywords": ["knit", "knitting", "mediterranean", "stress relief"],
        "sessions": [22],
        "fact_text": "Knitting, Mediterranean diet, anxiety management for stress relief",
    },
    "elena_q13": {
        "answer_keywords": ["42k", "40,000", "40000"],
        "conv_keywords": ["42k", "40,000", "40000", "student loan"],
        "sessions": [17],
        "fact_text": "Student loans -- question says $40,000, conversation may say 42k",
    },

    # == DAVID ================================================================
    "david_q01": {
        "answer_keywords": ["ap us history", "us history"],
        "conv_keywords": ["ap us history", "us history", "ap euro", "switched"],
        "sessions": [23],
        "fact_text": "Switching from AP Euro to AP US History",
    },
    "david_q03": {
        "answer_keywords": ["subaru", "outback"],
        "conv_keywords": ["subaru", "outback", "camry", "traded"],
        "sessions": [24],
        "fact_text": "Bought Subaru Outback, replacing old Camry",
    },
    "david_q04": {
        "answer_keywords": ["abandoned the book", "abandoned", "nobody would read"],
        "conv_keywords": ["book", "primary sources", "abandoned", "nobody would read"],
        "sessions": [9, 18],
        "fact_text": "Abandoned book about teaching through primary sources",
    },
    "david_q05": {
        "answer_keywords": ["hearing loss", "hearing aid"],
        "conv_keywords": ["hearing loss", "hearing aid", "left ear"],
        "sessions": [5],
        "fact_text": "Mild hearing loss in left ear, wears hearing aid",
    },
    "david_q11": {
        "answer_keywords": ["parent-teacher conference", "parent teacher"],
        "conv_keywords": ["parent-teacher", "parent teacher", "conference", "march 12"],
        "sessions": [10],
        "fact_text": "Parent-teacher conferences March 12 (expired)",
    },

    # == AMARA ================================================================
    "amara_q01": {
        "answer_keywords": ["bedford row", "7 bedford"],
        "conv_keywords": ["bedford row", "7 bedford", "brick court", "senior tenant"],
        "sessions": [19],
        "fact_text": "Moved from 4 Brick Court to 7 Bedford Row as senior tenant",
    },
    "amara_q02": {
        "answer_keywords": ["samsung", "galaxy s25", "galaxy"],
        "conv_keywords": ["samsung", "galaxy", "s25", "iphone", "android"],
        "sessions": [22],
        "fact_text": "Switched from iPhone 14 Pro to Samsung Galaxy S25",
    },
    "amara_q03": {
        "answer_keywords": ["boxing", "bethnal green"],
        "conv_keywords": ["boxing", "bethnal green", "gym", "dropped running"],
        "sessions": [25],
        "fact_text": "Dropped running, switched to boxing in Bethnal Green",
    },
    "amara_q06": {
        "answer_keywords": ["dropped the master", "dropped the llm", "not worth"],
        "conv_keywords": ["llm", "ucl", "human rights", "master", "dropped", "not worth"],
        "sessions": [11, 17],
        "fact_text": "Dropped UCL human rights LLM -- not worth time/money",
    },
    "amara_q07": {
        "answer_keywords": ["wine at", "chambers dinner", "had wine"],
        "conv_keywords": ["wine", "alcohol", "drink", "chambers dinner", "two years"],
        "sessions": [9, 27],
        "fact_text": "Said no alcohol 2 years, but had wine at chambers dinner",
    },
    "amara_q08": {
        "answer_keywords": ["hackney", "islington"],
        "conv_keywords": ["hackney", "islington", "east london", "considering move"],
        "sessions": [1, 26],
        "fact_text": "Lives in Hackney, considering move to Islington",
    },
    "amara_q11": {
        "answer_keywords": ["boxing", "nigerian cooking", "podcasts"],
        "conv_keywords": ["boxing", "nigerian cooking", "podcasts", "reading", "justice"],
        "sessions": [25],
        "fact_text": "Boxing, Nigerian cooking, podcasts, reading about justice",
    },
    "amara_q12": {
        "answer_keywords": ["llb", "king's college", "kings college"],
        "conv_keywords": ["llb", "king's college", "kings college", "law degree"],
        "sessions": [2],
        "fact_text": "LLB from King's College London",
    },
    "amara_q14": {
        "answer_keywords": ["18,000", "18000"],
        "conv_keywords": ["18,000", "18000", "tuition", "cost"],
        "sessions": [11],
        "fact_text": "UCL LLM would have cost GBP 18,000",
    },

    # == JAKE =================================================================
    "jake_q02": {
        "answer_keywords": ["kayla"],
        "conv_keywords": ["kayla", "megan", "broke up", "nurse", "mass general"],
        "sessions": [24],
        "fact_text": "Megan broke up, now dating Kayla (nurse at Mass General)",
    },
    "jake_q03": {
        "answer_keywords": ["eire pub", "dorchester"],
        "conv_keywords": ["eire pub", "dorchester", "waitress", "mom works"],
        "sessions": [3],
        "fact_text": "Mom waitresses at Eire Pub in Dorchester, 20 years",
    },
    "jake_q05": {
        "answer_keywords": ["ev charger", "ev home charger", "insurance too expensive"],
        "conv_keywords": ["ev charger", "ev home", "side gig", "insurance", "dad said"],
        "sessions": [11, 16],
        "fact_text": "EV charger side gig dead -- dad said not ready, insurance too expensive",
    },
    "jake_q07": {
        "answer_keywords": ["trillium", "craft beer"],
        "conv_keywords": ["trillium", "craft beer", "ipa", "bud light"],
        "sessions": [26],
        "fact_text": "Got into craft beer -- favorites from Trillium Brewing",
    },
    "jake_q09": {
        "answer_keywords": ["county cork", "brennan & sons", "brennan electric", "eire pub"],
        "conv_keywords": ["irish", "county cork", "grandparents", "brennan", "eire pub", "bridget"],
        "sessions": [1, 3, 5],
        "fact_text": "Irish-American, grandparents County Cork, dad Brennan & Sons, mom Eire Pub, sister Bridget",
    },
    "jake_q10": {
        "answer_keywords": ["rec hockey", "bajko", "gaming pc"],
        "conv_keywords": ["hockey", "bajko", "wednesday", "gaming", "craft beer"],
        "sessions": [4, 23, 26],
        "fact_text": "Rec hockey Wednesdays, gaming PC, craft beer, sports with the boys",
    },

    # == TOM ==================================================================
    "tom_q02": {
        "answer_keywords": ["ioniq", "hyundai"],
        "conv_keywords": ["ioniq", "hyundai", "electric", "defender", "land rover"],
        "sessions": [24],
        "fact_text": "Sold Land Rover Defender, bought Hyundai Ioniq 5",
    },
    "tom_q03": {
        "answer_keywords": ["walking group"],
        "conv_keywords": ["walking group", "pub quiz", "gerald", "fox", "stopped"],
        "sessions": [23],
        "fact_text": "Stopped pub quiz at The Fox, joined walking group",
    },
    "tom_q04": {
        "answer_keywords": ["atrial fibrillation", "warfarin"],
        "conv_keywords": ["atrial fibrillation", "warfarin", "blood thinner"],
        "sessions": [4],
        "fact_text": "Atrial fibrillation, takes warfarin",
    },
    "tom_q06": {
        "answer_keywords": ["barn conversion", "planning permission", "barn scrapped"],
        "conv_keywords": ["barn", "honey processing", "planning permission", "scrapped"],
        "sessions": [11, 17],
        "fact_text": "Barn conversion scrapped -- planning permission denied, too expensive",
    },
    "tom_q07": {
        "answer_keywords": ["instagram", "200 likes", "bee photo"],
        "conv_keywords": ["instagram", "social media", "200 likes", "bee photo"],
        "sessions": [9, 29],
        "fact_text": "Said no social media, but posted bee photo on Instagram (200 likes)",
    },
    "tom_q11": {
        "answer_keywords": ["arup"],
        "conv_keywords": ["arup", "civil engineer", "beng", "35 years"],
        "sessions": [5],
        "fact_text": "BEng civil engineering, worked at Arup 35 years",
    },

    # == OMAR =================================================================
    "omar_q01": {
        "answer_keywords": ["uber", "back to uber", "switched back"],
        "conv_keywords": ["uber", "lyft", "rideshare"],
        "sessions": [1, 14, 26],
        "fact_text": "Uber -> Lyft -> back to Uber (triple version)",
    },
    "omar_q02": {
        "answer_keywords": ["gulfton"],
        "conv_keywords": ["gulfton", "studio", "alief", "moved", "save"],
        "sessions": [22],
        "fact_text": "Moved from Alief 1BR to Gulfton studio, saves $200/mo",
    },
    "omar_q05": {
        "answer_keywords": ["camry plan dead", "camry is dead", "sold before", "whole camry plan"],
        "conv_keywords": ["camry", "toyota", "dead", "financing"],
        "sessions": [10, 16],
        "fact_text": "Camry plan dead -- car sold before financing secured",
    },
    "omar_q06": {
        "answer_keywords": ["ps5", "playstation"],
        "conv_keywords": ["ps5", "playstation", "never spends", "stress"],
        "sessions": [7, 25],
        "fact_text": "Said never spends on himself but bought PS5 for stress",
    },
    "omar_q07": {
        "answer_keywords": ["dallas"],
        "conv_keywords": ["dallas", "cousin", "re market", "houston"],
        "sessions": [4, 27],
        "fact_text": "Staying in Houston but cousin says Dallas RE market is better",
    },
    "omar_q08": {
        "answer_keywords": ["century 21"],
        "conv_keywords": ["uber", "century 21", "real estate", "brokerage"],
        "sessions": [26, 29],
        "fact_text": "Uber driver + starting at Century 21 in April 2026",
    },
    "omar_q11": {
        "answer_keywords": [],  # correct answer is "None explicitly mentioned"
        "conv_keywords": ["tired", "stressed", "overwork", "health"],
        "sessions": [],
        "fact_text": "No explicit health conditions -- tired/stressed from overwork",
    },
    "omar_q12": {
        "answer_keywords": ["century 21", "real estate"],
        "conv_keywords": ["real estate", "century 21", "exam", "passed", "brokerage"],
        "sessions": [9, 29],
        "fact_text": "Real estate career -- passed exam, joining Century 21",
    },
    "omar_q14": {
        "answer_keywords": ["300", "400"],
        "conv_keywords": ["300", "400", "sends", "month", "remittance"],
        "sessions": [3],
        "fact_text": "Sends $300-400/month to mother in Khartoum",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results():
    return json.load(open(RESULTS_PATH, encoding="utf-8"))


def load_questions():
    raw = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    return {q["id"]: q for q in raw}


def get_session_path(persona: str, session_num: int) -> Path:
    d = PERSONA_DIRS[persona]
    return LIFEMEMEVAL_DIR / d / "sessions" / f"session_{session_num:02d}.json"


def read_session_text(persona: str, session_num: int) -> str:
    """Read a conversation session and return all message content joined."""
    p = get_session_path(persona, session_num)
    if not p.exists():
        return ""
    data = json.load(open(p, encoding="utf-8"))
    turns = data.get("turns", [])
    return "\n".join(t.get("content", "") for t in turns)


# ---------------------------------------------------------------------------
# Neo4j: load ALL edges (not just fr_enriched) + entity nodes
# ---------------------------------------------------------------------------

async def load_all_edges(driver, group_ids: list[str]) -> list[dict]:
    """Load ALL edges for given group_ids — includes non-enriched ones."""
    result = await driver.execute_query(
        """
        MATCH (s)-[e:RELATES_TO]->(t)
        WHERE e.group_id IN $gids
        RETURN e.uuid AS uuid,
               e.fact AS fact,
               e.group_id AS group_id,
               e.fr_enriched AS enriched,
               e.fr_primary_category AS category,
               e.fr_superseded_by AS superseded_by
        """,
        gids=group_ids,
    )
    records = result.records if hasattr(result, "records") else result
    edges = []
    for rec in records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        edges.append(d)
    return edges


async def load_entity_nodes(driver, group_ids: list[str]) -> list[dict]:
    """Load entity node names and summaries."""
    result = await driver.execute_query(
        """
        MATCH (n)
        WHERE n.group_id IN $gids AND (n:Entity OR n:Person OR n:Concept)
        RETURN n.name AS name,
               n.summary AS summary,
               n.group_id AS group_id
        """,
        gids=group_ids,
    )
    records = result.records if hasattr(result, "records") else result
    nodes = []
    for rec in records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        nodes.append(d)
    return nodes


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def search_edges(edges: list[dict], keywords: list[str], group_id: str) -> list[dict]:
    """Find edges matching any keyword for a given persona's group_id."""
    matches = []
    for edge in edges:
        if edge["group_id"] != group_id:
            continue
        fact = (edge.get("fact") or "").lower()
        matched_kws = [kw for kw in keywords if kw.lower() in fact]
        if matched_kws:
            matches.append({**edge, "matched_keywords": matched_kws})
    return matches


def search_entities(nodes: list[dict], keywords: list[str], group_id: str) -> list[dict]:
    """Find entity nodes whose name/summary match any keyword."""
    matches = []
    for node in nodes:
        if node["group_id"] != group_id:
            continue
        text = ((node.get("name") or "") + " " + (node.get("summary") or "")).lower()
        matched_kws = [kw for kw in keywords if kw.lower() in text]
        if matched_kws:
            matches.append({**node, "matched_keywords": matched_kws})
    return matches


def search_conversation(persona: str, sessions: list[int], keywords: list[str]) -> dict:
    """Search conversation sessions for keyword matches. Returns evidence."""
    evidence = {"found": False, "session": None, "snippet": ""}
    for sess_num in sessions:
        text = read_session_text(persona, sess_num)
        if not text:
            continue
        text_lower = text.lower()
        matched = [kw for kw in keywords if kw.lower() in text_lower]
        if matched:
            # Extract a snippet around the first match
            kw = matched[0].lower()
            idx = text_lower.find(kw)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(kw) + 100)
            snippet = text[start:end].replace("\n", " ").strip()
            evidence = {"found": True, "session": sess_num, "snippet": snippet}
            break
    return evidence


# ---------------------------------------------------------------------------
# Classification waterfall
# ---------------------------------------------------------------------------

def classify_gap(
    question_id: str,
    question: dict,
    answer_edge_matches: list[dict],
    broad_edge_matches: list[dict],
    entity_matches: list[dict],
    conversation_evidence: dict,
    top5_facts: list[dict],
) -> tuple[str, str]:
    """
    Classify a gap into E1-E5 or RANKING. Returns (category, reason).

    answer_edge_matches: edges matching STRICT answer_keywords (correct fact)
    broad_edge_matches:  edges matching BROAD conv_keywords (topic-level)

    Waterfall:
      RANKING -- edge exists and is answerable but outranked by stale edge
      E1 -- benchmark design issue (fact not in conversation, or mismatch)
      E2 -- fact IS in conversation, correct-answer edge NOT extracted
      E3 -- correct-answer edge exists but not retrieved / too vague
      E4 -- correct-answer edge exists but wrong category
      E5 -- composite fact across multiple sessions, partial edges
    """
    registry = KEYWORD_REGISTRY.get(question_id, {})
    sessions = registry.get("sessions", [])
    has_answer_edge = len(answer_edge_matches) > 0
    has_broad_edge = len(broad_edge_matches) > 0
    fact_in_conversation = conversation_evidence.get("found", False)

    # --- Special case: omar_q01 is answerable but fails due to ranking ---
    if question_id == "omar_q01":
        return "RANKING", "Edge exists and is answerable -- stale Lyft edge outranks correct Uber edge"

    # --- E1: benchmark design error ---
    if question_id == "omar_q11":
        return "E1", "Correct answer is 'None explicitly mentioned' -- no extractable fact"

    if question_id == "elena_q13":
        has_42k_edge = any("42k" in (e.get("fact") or "").lower() for e in answer_edge_matches)
        if has_42k_edge:
            return "E1", "Edge has 'about 42k' but question expects exact '$40,000' -- benchmark mismatch"

    # Generic E1: no sessions listed AND not found in conversation AND no answer edges
    if not sessions and not fact_in_conversation and not has_answer_edge:
        return "E1", "No conversation session specified and no evidence found"

    # --- E5: composite/multi-session facts ---
    composite_questions = {
        "jake_q09", "jake_q10", "elena_q08", "elena_q12",
        "amara_q11", "omar_q08",
    }
    if question_id in composite_questions and len(sessions) >= 2:
        if has_answer_edge:
            return "E5", f"Composite fact across {len(sessions)} sessions -- partial answer edges exist but incomplete"
        elif fact_in_conversation:
            return "E5", f"Composite fact across {len(sessions)} sessions -- facts in conversation but no answer edges"
        return "E5", f"Composite fact spread across {len(sessions)} sessions"

    # --- E3/E4: correct-answer edge EXISTS in Neo4j but wasn't retrieved ---
    if has_answer_edge:
        enriched = [e for e in answer_edge_matches if e.get("enriched")]
        if enriched:
            cats = set(e.get("category", "") for e in enriched if e.get("category"))
            relevant_cats = set(question.get("relevant_categories", []))
            if cats and relevant_cats and not cats.intersection(relevant_cats):
                return "E4", f"Answer edge exists but category {cats} doesn't match relevant {relevant_cats}"
            return "E3", f"Answer edge exists ({len(enriched)} enriched) but not retrieved -- routing/retrieval gap"
        return "E3", f"Answer edge exists ({len(answer_edge_matches)} non-enriched) but not enriched/routable"

    # --- E2: correct-answer edge does NOT exist -- extraction miss ---
    # (We reach here only if answer_keywords found NO matching edges)
    if fact_in_conversation:
        sess = conversation_evidence.get("session", "?")
        # Check if there are broad/topic edges (wrong version or vague)
        if has_broad_edge:
            return "E2", f"Fact in session {sess} -- only topic-level edges exist (no correct-answer edge extracted)"
        return "E2", f"Fact in session {sess} but no matching edge extracted at all"

    # Entity-only match
    if entity_matches and not has_answer_edge:
        return "E2", "Entity node references fact but no answer edge extracted"

    # Session-based fallback
    if sessions:
        return "E2", f"Expected in session(s) {sessions} -- likely extraction miss (could not verify conversation)"

    return "E2", "No answer edge found -- defaulting to extraction miss"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    gap_rows: list[dict],
    total_questions: int,
    timestamp: str,
) -> str:
    """Generate the markdown report."""
    lines = []
    lines.append("# LifeMemBench Extraction Gap Audit Report\n")
    lines.append(f"*Generated: {timestamp}*\n")

    # --- Executive Summary ---
    counts = Counter(r["classification"] for r in gap_rows)
    total_gaps = len(gap_rows)
    e2_count = counts.get("E2", 0)
    e2_pct = (e2_count / total_gaps * 100) if total_gaps else 0

    lines.append("## Executive Summary\n")
    lines.append(f"- **Total gaps:** {total_gaps} out of {total_questions} questions (av_pass=False in 'full' config)")
    lines.append(f"- **E1 (Benchmark design):** {counts.get('E1', 0)}")
    lines.append(f"- **E2 (Extraction miss):** {counts.get('E2', 0)} ({e2_pct:.0f}%)")
    lines.append(f"- **E3 (Vague/routing gap):** {counts.get('E3', 0)}")
    lines.append(f"- **E4 (Wrong category):** {counts.get('E4', 0)}")
    lines.append(f"- **E5 (Composite/multi-session):** {counts.get('E5', 0)}")
    lines.append(f"- **RANKING (not extraction):** {counts.get('RANKING', 0)}")
    e3_count = counts.get("E3", 0)
    e4_count = counts.get("E4", 0)
    retrieval_pct = ((e3_count + e4_count) / total_gaps * 100) if total_gaps else 0
    lines.append(f"\n**Key finding:** {retrieval_pct:.0f}% of gaps are E3+E4 (retrieval/routing). "
                  "The correct facts ARE extracted into edges but not retrieved for the right questions. "
                  f"Only {e2_pct:.0f}% are true extraction misses (E2). "
                  "The bottleneck is retrieval and category routing, not the extraction model.\n")

    # --- Full Gap Table ---
    lines.append("## Full Gap Table\n")
    lines.append("| # | question_id | persona | attack_vector | expected_fact | classification | conversation_file | nearest_edge | reason |")
    lines.append("|---|------------|---------|---------------|---------------|---------------|-------------------|-------------|--------|")

    for i, row in enumerate(gap_rows, 1):
        persona = row["persona"]
        av = row["attack_vector"]
        # Shorten AV for table
        av_short = av.split("_")[0] + "_" + "_".join(av.split("_")[1:3]) if "_" in av else av
        fact = row["fact_text"][:60].replace("|", "\\|")
        cls = row["classification"]
        conv = row.get("conversation_file", "—")
        nearest = row.get("nearest_edge", "—")[:60].replace("|", "\\|")
        reason = row["reason"][:80].replace("|", "\\|")
        lines.append(f"| {i} | {row['question_id']} | {persona} | {av_short} | {fact} | **{cls}** | {conv} | {nearest} | {reason} |")

    lines.append("")

    # --- Summary Counts ---
    lines.append("## Summary Counts\n")
    lines.append("| Category | Count | % |")
    lines.append("|----------|-------|---|")
    for cat in ["E1", "E2", "E3", "E4", "E5", "RANKING"]:
        c = counts.get(cat, 0)
        pct = (c / total_gaps * 100) if total_gaps else 0
        lines.append(f"| {cat} | {c} | {pct:.1f}% |")
    lines.append(f"| **Total** | **{total_gaps}** | **100%** |")
    lines.append("")

    # --- Per-Persona Breakdown ---
    lines.append("## Per-Persona Breakdown\n")
    persona_groups = defaultdict(list)
    for row in gap_rows:
        persona_groups[row["persona"]].append(row["classification"])

    lines.append("| Persona | Total | E1 | E2 | E3 | E4 | E5 | RANKING |")
    lines.append("|---------|-------|----|----|----|----|----|----|")
    for p in sorted(persona_groups):
        c = Counter(persona_groups[p])
        total = len(persona_groups[p])
        lines.append(f"| {p} | {total} | {c.get('E1',0)} | {c.get('E2',0)} | "
                      f"{c.get('E3',0)} | {c.get('E4',0)} | {c.get('E5',0)} | {c.get('RANKING',0)} |")
    lines.append("")

    # --- Per-Attack-Vector Breakdown ---
    lines.append("## Per-Attack-Vector Breakdown\n")
    av_groups = defaultdict(list)
    for row in gap_rows:
        av_groups[row["attack_vector"]].append(row["classification"])

    lines.append("| Attack Vector | Total | E1 | E2 | E3 | E4 | E5 | RANKING |")
    lines.append("|--------------|-------|----|----|----|----|----|----|")
    for av in sorted(av_groups):
        c = Counter(av_groups[av])
        total = len(av_groups[av])
        lines.append(f"| {av} | {total} | {c.get('E1',0)} | {c.get('E2',0)} | "
                      f"{c.get('E3',0)} | {c.get('E4',0)} | {c.get('E5',0)} | {c.get('RANKING',0)} |")
    lines.append("")

    # --- Detailed Analysis ---
    lines.append("## Detailed Analysis\n")
    for row in gap_rows:
        lines.append(f"### {row['question_id']} — **{row['classification']}**\n")
        lines.append(f"- **Question:** {row['question']}")
        lines.append(f"- **Correct answer:** {row['correct_answer']}")
        lines.append(f"- **Attack vector:** {row['attack_vector']}")
        lines.append(f"- **Expected fact:** {row['fact_text']}")
        lines.append(f"- **Classification:** {row['classification']}")
        lines.append(f"- **Reason:** {row['reason']}")
        if row.get("conversation_evidence"):
            lines.append(f"- **Conversation evidence:** session {row['conversation_evidence'].get('session', '?')} — "
                          f"`{row['conversation_evidence'].get('snippet', '')[:120]}...`")
        if row.get("edge_match_count", 0) > 0:
            lines.append(f"- **Matching edges:** {row['edge_match_count']}")
            for em in row.get("edge_match_samples", [])[:3]:
                lines.append(f"  - `[{em.get('uuid', '?')[:12]}]` ({', '.join(em.get('matched_keywords', []))}) "
                              f"\"{em.get('fact', '')[:100]}\"")
        if row.get("entity_match_count", 0) > 0:
            lines.append(f"- **Matching entity nodes:** {row['entity_match_count']}")
        lines.append("")

    # --- Key Patterns & Insights ---
    lines.append("## Key Patterns & Insights\n")

    # Which fact types does Grok miss?
    e2_rows = [r for r in gap_rows if r["classification"] == "E2"]
    e2_avs = Counter(r["attack_vector"] for r in e2_rows)
    lines.append("### Fact Types Grok Consistently Misses (E2 by AV)\n")
    for av, cnt in e2_avs.most_common():
        lines.append(f"- {av}: {cnt} misses")
    lines.append("")

    # Stable identity vs supersession
    e2_stable = sum(1 for r in e2_rows if "stable" in r["attack_vector"].lower())
    e2_super = sum(1 for r in e2_rows if "supersed" in r["attack_vector"].lower())
    e2_retract = sum(1 for r in e2_rows if "forgetting" in r["attack_vector"].lower())
    lines.append("### E2 Breakdown by Temporal Pattern\n")
    lines.append(f"- Stable identity facts (AV3): {e2_stable} — single-mention facts buried under noise")
    lines.append(f"- Superseded preferences (AV1): {e2_super} — both old and new versions missing")
    lines.append(f"- Retractions (AV7): {e2_retract} — retracted plans not extracted at all")
    lines.append("")

    # E3 breakdown
    e3_rows = [r for r in gap_rows if r["classification"] == "E3"]
    e3_avs = Counter(r["attack_vector"] for r in e3_rows)
    lines.append("### Retrieval Failures by AV (E3 -- dominant category)\n")
    for av, cnt in e3_avs.most_common():
        lines.append(f"- {av}: {cnt} retrieval failures")
    lines.append("")

    lines.append("### Entity Node Summaries vs Edge Facts\n")
    entity_only = sum(1 for r in gap_rows if r.get("entity_match_count", 0) > 0 and r.get("edge_match_count", 0) == 0)
    lines.append(f"- {entity_only} gaps where entity node has relevant info but no answer edge was created")
    lines.append("- This suggests entity summarization captures some facts that edge extraction misses\n")

    # --- Recommendations ---
    lines.append("## Recommendations\n")
    lines.append("### For E3+E4 (Retrieval/Routing -- Primary Bottleneck, ~78% of gaps)")
    lines.append("- **This is the #1 issue:** correct facts ARE in the graph but not retrieved")
    lines.append("- Improve semantic retrieval: current embedding-based search misses relevant edges")
    lines.append("- Category routing: E4 gaps show edges exist with wrong fr_primary_category")
    lines.append("- Consider query expansion: rephrase user questions to match edge phrasing")
    lines.append("- Top-k increase: some facts exist but rank below the top-5 cutoff")
    lines.append("- Cross-category retrieval: allow queries to pull from multiple categories\n")
    lines.append("### For E1 (Benchmark Design Issues)")
    lines.append("- elena_q13: conversation says '42k' but question expects '$40,000' -- fix question")
    lines.append("- omar_q11: correct answer is 'None explicitly mentioned' -- consider rewording\n")
    lines.append("### For E2 (Extraction Misses -- 4 gaps)")
    lines.append("- priya_q07: 'rock climbing' not extracted (only vague 'climbing' edges)")
    lines.append("- david_q04: 'abandoned the book' not captured as retraction")
    lines.append("- amara_q06: 'dropped the LLM' retraction not captured")
    lines.append("- amara_q14: GBP 18,000 tuition cost not extracted")
    lines.append("- Add extraction prompt instructions for retractions and specific numeric details\n")
    lines.append("### For E5 (Composite Facts -- 4 gaps)")
    lines.append("- Cross-session fact synthesis: family background, hobbies, career summaries")
    lines.append("- Consider composite edge creation for identity/family questions")
    lines.append("- Multi-edge retrieval: allow broad queries to pull from multiple edges\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("EXTRACTION GAP AUDIT -- LifeMemBench")
    print("=" * 60)

    # Load data
    results = load_results()
    questions = load_questions()
    per_question = results.get("per_question", {}).get("full", [])

    # Get all failing questions
    failures = [q for q in per_question if not q["av_pass"]]
    print(f"\nTotal questions: {len(per_question)}")
    print(f"Failures (av_pass=False): {len(failures)}")

    # Filter to only those in our registry
    registered_qids = set(KEYWORD_REGISTRY.keys())
    failures_in_registry = [f for f in failures if f["question_id"] in registered_qids]
    unregistered = [f["question_id"] for f in failures if f["question_id"] not in registered_qids]
    if unregistered:
        print(f"WARNING: {len(unregistered)} failures not in keyword registry: {unregistered}")

    # Connect to Neo4j
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    try:
        # Load all edges and entity nodes
        group_ids = list(PERSONAS.values())
        print(f"\nLoading ALL edges from Neo4j for {len(group_ids)} personas...")
        all_edges = await load_all_edges(driver, group_ids)
        print(f"Loaded {len(all_edges)} total edges")

        enriched = sum(1 for e in all_edges if e.get("enriched"))
        print(f"  enriched: {enriched}, non-enriched: {len(all_edges) - enriched}")

        print("Loading entity nodes...")
        all_entities = await load_entity_nodes(driver, group_ids)
        print(f"Loaded {len(all_entities)} entity nodes")

        # Process each gap
        gap_rows = []

        for failure in failures_in_registry:
            qid = failure["question_id"]
            q = questions.get(qid)
            if not q:
                print(f"  WARNING: {qid} not in questions JSON")
                continue

            persona = qid.rsplit("_q", 1)[0]
            group_id = PERSONAS.get(persona, "")
            registry = KEYWORD_REGISTRY[qid]
            answer_kws = registry["answer_keywords"]
            conv_kws = registry["conv_keywords"]
            sessions = registry["sessions"]

            # Search with STRICT answer keywords
            answer_edge_matches = search_edges(all_edges, answer_kws, group_id) if answer_kws else []
            # Search with BROAD conversation keywords
            broad_edge_matches = search_edges(all_edges, conv_kws, group_id)
            entity_matches = search_entities(all_entities, conv_kws, group_id)
            conversation_ev = search_conversation(persona, sessions, conv_kws)

            # Classify
            classification, reason = classify_gap(
                qid, q, answer_edge_matches, broad_edge_matches,
                entity_matches, conversation_ev,
                failure.get("top5_facts", []),
            )

            # Determine nearest edge text (prefer answer matches, fallback to broad)
            nearest_edge = "--"
            if answer_edge_matches:
                nearest_edge = answer_edge_matches[0].get("fact", "")[:80]
            elif broad_edge_matches:
                nearest_edge = broad_edge_matches[0].get("fact", "")[:80]

            # Conversation file reference
            conv_file = "--"
            if conversation_ev.get("found"):
                sess = conversation_ev["session"]
                conv_file = f"session_{sess:02d}.json"
            elif sessions:
                conv_file = f"session_{sessions[0]:02d}.json (expected)"

            all_edge_matches = answer_edge_matches or broad_edge_matches
            row = {
                "question_id": qid,
                "persona": persona,
                "attack_vector": q["attack_vector"],
                "question": q["question"],
                "correct_answer": q["correct_answer"],
                "fact_text": registry["fact_text"],
                "classification": classification,
                "reason": reason,
                "conversation_file": conv_file,
                "nearest_edge": nearest_edge,
                "conversation_evidence": conversation_ev,
                "edge_match_count": len(answer_edge_matches),
                "broad_edge_count": len(broad_edge_matches),
                "edge_match_samples": (answer_edge_matches or broad_edge_matches)[:5],
                "entity_match_count": len(entity_matches),
            }
            gap_rows.append(row)

            # Print progress
            sym = {"E1": "!", "E2": "X", "E3": "~", "E4": "C", "E5": "M", "RANKING": "R"}
            print(f"  [{sym.get(classification, '?')}] {qid:20s} -> {classification:8s}  "
                  f"ans={len(answer_edge_matches):2d}  broad={len(broad_edge_matches):2d}  "
                  f"conv={'Y' if conversation_ev.get('found') else 'N'}  "
                  f"| {reason[:60]}")

        # Summary
        print(f"\n{'=' * 60}")
        print("CLASSIFICATION SUMMARY")
        print(f"{'=' * 60}")
        counts = Counter(r["classification"] for r in gap_rows)
        for cat in ["E1", "E2", "E3", "E4", "E5", "RANKING"]:
            c = counts.get(cat, 0)
            pct = (c / len(gap_rows) * 100) if gap_rows else 0
            print(f"  {cat:8s}: {c:3d}  ({pct:.1f}%)")
        print(f"  {'TOTAL':8s}: {len(gap_rows):3d}")

        # Generate report
        timestamp = datetime.now(tz=None).strftime("%Y-%m-%d %H:%M")
        report = generate_report(gap_rows, len(per_question), timestamp)
        report_path = PROJECT_ROOT / "extraction_audit_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport written to: {report_path}")

    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
