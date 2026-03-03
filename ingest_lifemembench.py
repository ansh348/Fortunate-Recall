r"""
ingest_lifemembench.py - Ingest LifeMemBench personas into Neo4j via Graphiti.

Ingests 8 POC personas (35 sessions each, 280 total) into separate group_ids
so they stay isolated from the existing LongMemEval graph (group_id=full_234).

Each persona gets: group_id = lifemembench_{short_name}

After ingestion, edges are enriched with behavioral classification (10+1
categories, world-knowledge detection) using the same v8 prompt as
reenrich_edges_v7.py (which is actually v8).

Prerequisites:
    - Neo4j running: docker start neo4j-graphiti
    - .env with OPENAI_API_KEY, XAI_API_KEY, NEO4J_* vars
    - Graphiti installed in editable mode

Usage:
    python ingest_lifemembench.py --all                 # Ingest + enrich all 8
    python ingest_lifemembench.py --persona 1_priya     # Single persona
    python ingest_lifemembench.py --enrich-only         # Skip ingestion, just enrich
    python ingest_lifemembench.py --status              # Show progress table
"""

import argparse
import asyncio
import json
import os
import sys
import time as time_module
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment setup (MUST be before graphiti_core import)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

# Cap Graphiti's internal concurrency to stay under xAI's 480 RPM limit.
os.environ.setdefault('SEMAPHORE_LIMIT', '8')


def load_env():
    """Load .env from project root."""
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), val)


load_env()

for var in ['OPENAI_API_KEY', 'XAI_API_KEY']:
    if not os.environ.get(var):
        print(f"ERROR: {var} not set. Check your .env file.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Imports (after env is loaded)
# ---------------------------------------------------------------------------

from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.utils.bulk_utils import RawEpisode
from neo4j import AsyncGraphDatabase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POC_PERSONAS = {
    "1_priya":  "priya",
    "2_marcus": "marcus",
    "3_elena":  "elena",
    "4_david":  "david",
    "5_amara":  "amara",
    "6_jake":   "jake",
    "8_tom":    "tom",
    "17_omar":  "omar",
}

LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
ARTIFACTS_DIR = PROJECT_ROOT / "LifeMemEval" / "artifacts"
CHECKPOINT_FILE = ARTIFACTS_DIR / "lifemembench_ingest_checkpoint.json"

# Cost estimates (from ingest_full.py actuals)
COST_PER_SESSION = 0.014
COST_PER_EDGE = 0.001


def group_id_for(short_name: str) -> str:
    return f"lifemembench_{short_name}"


CUSTOM_EXTRACTION_INSTRUCTIONS = """
CRITICAL EXTRACTION PRIORITY:
This is a PERSONAL MEMORY system for a specific individual. Your #1 job is extracting
facts about the USER — their experiences, preferences, relationships, plans, possessions,
health, schedule, and opinions.

PRIORITIZE extracting:
- Facts where the user is the subject (e.g., "user works at Google", "user has ADHD")
- User's relationships with people (e.g., "user's brother Arjun lives in Seattle")
- User's preferences, habits, and routines (e.g., "user does hot yoga 4x/week")
- Specific numbers tied to the user (e.g., "lease expires August 2025")
- User's schedule, plans, and obligations (e.g., "dentist appointment June 12")
- Changes in user's situation (e.g., "user left Google, joined Anthropic")

DEPRIORITIZE (extract ONLY if directly relevant to user's life):
- General knowledge about places, landmarks, historical facts
- Assistant's explanations or recommendations that aren't about the user
- Facts between two non-user entities

NUMERIC PRESERVATION (CRITICAL):
When the user mentions ANY number, quantity, amount, count, time, or measurement, you MUST
preserve it exactly in the fact text.

TEMPORAL FACTS (CRITICAL):
When the user mentions dates, deadlines, or temporal anchors, preserve them exactly.
These are essential for the memory system's temporal reasoning.
"""


# ---------------------------------------------------------------------------
# Graphiti client
# ---------------------------------------------------------------------------

def get_graphiti_client() -> Graphiti:
    """Create Graphiti client with split xAI/OpenAI architecture."""
    xai_client = AsyncOpenAI(
        api_key=os.environ['XAI_API_KEY'],
        base_url="https://api.x.ai/v1",
    )
    llm_client = OpenAIClient(
        client=xai_client,
        config=LLMConfig(
            model="grok-4-1-fast-non-reasoning",
            small_model="grok-4-1-fast-non-reasoning",
        ),
        max_tokens=16384,
    )
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig())

    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'testpassword123')

    return Graphiti(
        neo4j_uri, neo4j_user, neo4j_password,
        llm_client=llm_client, embedder=embedder,
    )


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

def parse_session_date(date_str: str) -> datetime:
    """Parse session date string to timezone-aware UTC datetime.

    Session dates are ISO format like '2025-01-08'.
    Set to noon UTC to avoid date boundary issues.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=12, minute=0, second=0, tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        print(f"  WARNING: Could not parse date '{date_str}', using now()")
        return datetime.now(timezone.utc)


def load_persona_sessions(persona_dir: str) -> list[dict]:
    """Load all session JSON files for a persona, sorted by session_id."""
    sessions_dir = LIFEMEMEVAL_DIR / persona_dir / "sessions"
    if not sessions_dir.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")

    sessions = []
    for session_file in sorted(sessions_dir.glob("session_*.json")):
        with open(session_file, encoding="utf-8") as f:
            sessions.append(json.load(f))

    sessions.sort(key=lambda s: s["session_id"])

    if not sessions:
        raise ValueError(f"No session files found in {sessions_dir}")
    if len(sessions) != 35:
        print(f"  WARNING: Expected 35 sessions for {persona_dir}, found {len(sessions)}")

    return sessions


def build_raw_episodes(sessions: list[dict], persona_dir: str) -> list[RawEpisode]:
    """Convert session dicts to RawEpisode objects for add_episode_bulk."""
    short_name = POC_PERSONAS[persona_dir]
    episodes = []

    for session in sessions:
        sid = session["session_id"]
        session_type = session.get("type", "unknown")

        content = "\n".join(
            f"{t['role']}: {t['content']}" for t in session["turns"]
        )

        episodes.append(RawEpisode(
            name=f"lmb_{short_name}_s{sid:02d}",
            content=content,
            source_description=(
                f"LifeMemBench {short_name} session {sid} ({session_type})"
            ),
            source=EpisodeType.message,
            reference_time=parse_session_date(session["date"]),
        ))

    return episodes


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    """Load ingestion checkpoint."""
    if CHECKPOINT_FILE.exists():
        return json.load(open(CHECKPOINT_FILE, encoding='utf-8'))
    return {
        'started_at': None,
        'personas': {},
    }


def save_checkpoint(checkpoint: dict):
    """Save ingestion checkpoint atomically."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)
    tmp.replace(CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# Edge enrichment — v8 classifier (copied from reenrich_edges_v7.py)
# ---------------------------------------------------------------------------

V7_CATEGORIES = [
    "OBLIGATIONS", "RELATIONAL_BONDS", "HEALTH_WELLBEING", "IDENTITY_SELF_CONCEPT",
    "HOBBIES_RECREATION", "PREFERENCES_HABITS", "INTELLECTUAL_INTERESTS",
    "LOGISTICAL_CONTEXT", "PROJECTS_ENDEAVORS", "FINANCIAL_MATERIAL", "OTHER",
]

V8_TO_DECAY = {cat: cat for cat in V7_CATEGORIES}

SYSTEM_PROMPT = """You are a memory classification system for a conversational AI. You classify extracted facts from a personal knowledge graph into behavioral categories that determine how the memory should age over time.

IMPORTANT CONTEXT: Every fact you see was extracted from a real conversation between a user and an AI assistant. Even facts that LOOK like general knowledge were discussed because they are relevant to the user's life, projects, hobbies, or decisions. Default to treating facts as personal memories unless they are PURELY encyclopedic with zero connection to the user.

You will receive:
- **Source entity** and **Target entity**: the two nodes this fact connects in the graph.
- **Fact**: the extracted claim stored on the edge between those entities.
- **Is speaker edge**: whether the user or assistant is one of the entities. If true, this is ALWAYS a personal memory — skip Step 0.

## Step 0: Personal vs World Knowledge

BEFORE classifying into behavioral categories, determine whether this fact is a PERSONAL MEMORY or GENERAL WORLD KNOWLEDGE.

**CRITICAL RULES:**
- If `is_speaker_edge` is true, this is ALWAYS personal. Skip to Step 1.
- If the fact describes something the user DOES, OWNS, BOUGHT, PLANS, VISITED, LIKES, or IS — it is personal, even if the entities are objects/places.
- Only tag as world knowledge if the fact is a standalone encyclopedic statement that would be true regardless of who the user is AND has no actionable relevance to the user's life.
- **When in doubt, classify as personal.** False negatives (missing a WK tag) are harmless. False positives (tagging personal facts as WK) destroy memory.

Examples of world knowledge (→ OTHER):
- "Tokyo Tower was built in 1958" (pure historical trivia, no user connection)
- "Python is a programming language" (encyclopedic definition)
- "LeBron James played for the LA Lakers" (celebrity fact)
- "Nice has a bike-sharing system called Vélo Bleu" (city infrastructure fact)

Examples that LOOK like world knowledge but are PERSONAL (→ classify normally):
- "humidifier maintains humidity for plants" → user is growing plants, this is their setup (HOBBIES_RECREATION)
- "Humpback Rocks is a challenging hike in the Blue Ridge Mountains" → user hikes there (HOBBIES_RECREATION)
- "50-inch 4K TV was picked up for $350" → user's purchase (FINANCIAL_MATERIAL)
- "shade cloth protects plants from excessive heat" → user's greenhouse project (PROJECTS_ENDEAVORS or HOBBIES_RECREATION)
- "Call of Duty game was discounted by 20%" → user's purchase context (FINANCIAL_MATERIAL)
- "Saunders-Monticello Trail offers beautiful views" → user's local trail (HOBBIES_RECREATION or PREFERENCES_HABITS)

Key signal: Could this fact help the assistant remember something useful about the user's life? → personal memory. Is it a random encyclopedia entry with no user connection? → world knowledge.

## Categories (for personal memories only)

1. OBLIGATIONS — Tasks, deadlines, appointments, promises, action items. Time-bound, lose relevance after completion.
2. RELATIONAL_BONDS — Relationships with people: family, partners, friends, colleagues. Status, dynamics, emotional quality.
3. HEALTH_WELLBEING — Physical/mental health, medications, diagnoses, chronic conditions, fitness metrics, mental health, diet/nutrition, explicitly health-motivated behaviors.
4. IDENTITY_SELF_CONCEPT — Core stable traits: ethnicity, name, occupation, heritage, values, beliefs, personality. Things that wouldn't change if you woke up tomorrow with amnesia about your preferences. Stable across years. Near-zero decay.
5. HOBBIES_RECREATION — Active leisure involving accumulated skill or equipment investment: fishing, coin collecting, cycling, painting, gardening, cooking, photography. Can go dormant but reactivate. Slow decay (months to years).
6. PREFERENCES_HABITS — Current tastes and consumption patterns: media preferences (TV shows, movies, documentaries), food choices, subscription services, lifestyle routines, favorite restaurants. What you LIKE right now. Moderate decay (weeks to months), easily superseded.
7. INTELLECTUAL_INTERESTS — Curiosities, academic fascinations, active learning goals. Exploration WITHOUT a concrete deliverable. If actively producing/building/delivering something -> PROJECTS_ENDEAVORS instead.
8. LOGISTICAL_CONTEXT — Transient scheduling details: appointment times, one-time locations, errands, travel logistics. Recurring activities are NOT logistical even if they have dates attached.
9. PROJECTS_ENDEAVORS — Ongoing works: startups, research papers, creative projects, long-term goals with milestones and timelines.
10. FINANCIAL_MATERIAL — Budget, income, expenses, purchases, debts, assets, owned devices, hardware specs, tech upgrades.
11. OTHER — World knowledge, or genuinely does not fit categories 1-10.

## Classification Rules (for personal memories)

- **CLASSIFY THE FACT, NOT THE CONTEXT.** Determine the category based on the nature of the information being stored (what this fact reveals about someone's life), NOT the conversational context surrounding it.
- When the fact is a price, cost, or financial amount -> FINANCIAL_MATERIAL.
- When the fact is a transient scheduling detail (appointment time, location visited once, errand) -> LOGISTICAL_CONTEXT. Recurring activities, hobbies, and events are NOT logistical even if they have dates attached.
- When the fact is about who someone IS at their core (name, role, ethnicity, heritage, core values) -> IDENTITY_SELF_CONCEPT.
- When the fact is about an active leisure activity involving skill or equipment (fishing, photography, cycling, cooking, gardening) -> HOBBIES_RECREATION.
- When the fact is about current tastes, media consumption, food preferences, subscriptions, or lifestyle routines -> PREFERENCES_HABITS.
- **Key boundary test:** Does it involve accumulated SKILL? -> HOBBIES_RECREATION. Is it a consumption CHOICE that could flip tomorrow? -> PREFERENCES_HABITS. Is it a core trait unchanged across years? -> IDENTITY_SELF_CONCEPT.
- **PREFERENCES_HABITS GUARDRAILS:** PREFERENCES_HABITS is ONLY for consumption patterns that could change next week with no loss of identity, skill, or health impact.
  - If someone is LEARNING a language, skill, or subject -> INTELLECTUAL_INTERESTS.
  - If someone declares "I am a [role/aspiration]" -> IDENTITY_SELF_CONCEPT.
  - If a behavior is health/wellbeing-motivated -> HEALTH_WELLBEING.
  - If the fact is about device/hardware ownership, cost, or purchase -> FINANCIAL_MATERIAL.
  - If the fact is about usage frequency or wearing pattern -> PREFERENCES_HABITS.
  - If the fact is about activity the item supports (art supplies for art hobby) -> HOBBIES_RECREATION.
- **FINANCIAL_MATERIAL GUARDRAIL:** A physical item being mentioned does NOT automatically make it FINANCIAL_MATERIAL. Classify by what the fact IS ABOUT: item value/cost/ownership -> FINANCIAL_MATERIAL. Usage frequency -> PREFERENCES_HABITS. Activity it supports -> HOBBIES_RECREATION.
- **DELIVERABLE TEST (MANDATORY OVERRIDE):** If the fact involves creating, editing, or revising a NAMED artifact -> primary category MUST be PROJECTS_ENDEAVORS.
- **NAMED PERSON DISAMBIGUATION:** When a named person appears, ask: "Is this fact ABOUT the relationship itself, or does the person merely provide context?" Timing/counts/events involving a person -> classify by WHAT is measured. Relationship origin or dynamics -> RELATIONAL_BONDS.
- Assign membership weights across ALL relevant categories. Weights must sum to 1.0.
- Primary category gets the STRICTLY highest weight (minimum 0.3). No ties.
- OTHER must NEVER exceed 0.5 weight for personal memories. If uncertain, distribute across top 2-3 plausible categories.
- For world knowledge, OTHER should be 1.0.

## Output Format (strict JSON, no commentary)

{"primary_category": "CATEGORY_NAME", "is_world_knowledge": false, "weights": {"OBLIGATIONS": 0.0, "RELATIONAL_BONDS": 0.0, "HEALTH_WELLBEING": 0.0, "IDENTITY_SELF_CONCEPT": 0.0, "HOBBIES_RECREATION": 0.0, "PREFERENCES_HABITS": 0.0, "INTELLECTUAL_INTERESTS": 0.0, "LOGISTICAL_CONTEXT": 0.0, "PROJECTS_ENDEAVORS": 0.0, "FINANCIAL_MATERIAL": 0.0, "OTHER": 0.0}, "emotional_loading": {"detected": false, "type": null, "intensity": null}, "confidence": "high"}"""


def validate_and_normalize(raw: dict) -> dict:
    """Normalize classifier output. Fixes casing, missing keys, renormalizes weights."""

    is_world_knowledge = raw.get("is_world_knowledge", False)

    primary = raw.get("primary_category", "OTHER")
    primary_upper = primary.upper().replace(" ", "_")

    best_match = None
    for canon in V7_CATEGORIES:
        if primary_upper == canon.upper():
            best_match = canon
            break
    if best_match is None:
        for canon in V7_CATEGORIES:
            if canon.upper() in primary_upper or primary_upper in canon.upper():
                best_match = canon
                break
    if best_match is None:
        best_match = "OTHER"
    primary = best_match

    raw_weights = raw.get("weights", {})
    weights = {}
    for canon in V7_CATEGORIES:
        found = False
        for k, v in raw_weights.items():
            if k.upper().replace(" ", "_") == canon.upper():
                weights[canon] = float(v)
                found = True
                break
        if not found:
            weights[canon] = 0.0

    # World knowledge: force OTHER=1.0, skip all guardrails
    if is_world_knowledge:
        weights = {k: 0.0 for k in V7_CATEGORIES}
        weights["OTHER"] = 1.0
        primary = "OTHER"
        emo = raw.get("emotional_loading", {})
        if not isinstance(emo, dict):
            emo = {"detected": False, "type": None, "intensity": None}
        return {
            "primary_category": primary,
            "is_world_knowledge": True,
            "weights": weights,
            "emotional_loading": emo,
            "confidence": raw.get("confidence", "low"),
        }

    # OTHER suppression (personal memories only)
    if primary == "OTHER" and weights.get("OTHER", 0) < 0.4:
        non_other = {k: v for k, v in weights.items() if k != "OTHER" and v > 0}
        if non_other:
            runner_up = max(non_other, key=non_other.get)
            if weights["OTHER"] - non_other[runner_up] < 0.1:
                primary = runner_up

    # Ensure primary has strictly highest weight
    max_weight = max(weights.values()) if weights else 0.3
    if weights[primary] < max_weight:
        weights[primary] = max_weight + 0.05
    for k in weights:
        if k != primary and weights[k] >= weights[primary]:
            weights[primary] = weights[k] + 0.05

    # Renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
    else:
        weights[primary] = 1.0

    emo = raw.get("emotional_loading", {})
    if not isinstance(emo, dict):
        emo = {"detected": False, "type": None, "intensity": None}

    return {
        "primary_category": primary,
        "is_world_knowledge": False,
        "weights": weights,
        "emotional_loading": emo,
        "confidence": raw.get("confidence", "unknown"),
    }


def map_to_decay_engine(v8_result: dict) -> dict:
    """Map v8 category names to decay engine category names."""
    v8_primary = v8_result["primary_category"]
    decay_primary = V8_TO_DECAY.get(v8_primary, "OTHER")

    decay_weights = {}
    for cat, weight in v8_result["weights"].items():
        decay_cat = V8_TO_DECAY.get(cat, "OTHER")
        decay_weights[decay_cat] = decay_weights.get(decay_cat, 0.0) + weight

    emo = v8_result.get("emotional_loading", {})
    return {
        "primary_category": decay_primary,
        "membership_weights": decay_weights,
        "is_world_knowledge": v8_result.get("is_world_knowledge", False),
        "confidence": 0.9 if v8_result.get("confidence") == "high" else 0.7,
        "emotional_loading": emo.get("detected", False) if isinstance(emo, dict) else False,
    }


def _is_speaker_entity(name: str) -> bool:
    """Check if an entity name represents a speaker (user or assistant)."""
    lower = name.lower().strip()
    return lower in ('user', 'assistant') or lower.startswith('user ') or lower.startswith('assistant ')


async def classify_edge_fact(
    fact_text: str,
    src_name: str,
    tgt_name: str,
    xai_client: AsyncOpenAI,
) -> dict:
    """Classify a single edge fact using the v8 entity-aware prompt."""

    fact_display = fact_text if fact_text else "(empty)"
    is_speaker = _is_speaker_entity(src_name or '') or _is_speaker_entity(tgt_name or '')

    prompt = f"""Classify this knowledge graph edge into behavioral categories.

**Source entity:** {src_name or 'unknown'}
**Target entity:** {tgt_name or 'unknown'}
**Fact:** {fact_display}
**Is speaker edge:** {is_speaker}

{"This edge directly involves the user/assistant — it is ALWAYS a personal memory. Skip Step 0 and go straight to classification." if is_speaker else "Step 0: Is this a personal memory about the user, or general world knowledge? Remember: these facts come from personal conversations, so lean toward personal unless purely encyclopedic."}
Step 1: Classify WHAT this fact IS (relationship? hobby? schedule? identity? etc.)

Respond with JSON only."""

    for attempt in range(3):
        try:
            resp = await xai_client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
            )
            raw = json.loads(resp.choices[0].message.content)
            # Hard override: speaker edges are NEVER world knowledge
            if is_speaker and raw.get("is_world_knowledge"):
                raw["is_world_knowledge"] = False
            validated = validate_and_normalize(raw)
            mapped = map_to_decay_engine(validated)
            return mapped
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 * (2 ** attempt))
            else:
                return {
                    "primary_category": "OTHER",
                    "membership_weights": {"OTHER": 1.0},
                    "is_world_knowledge": False,
                    "confidence": 0.0,
                    "emotional_loading": False,
                    "error": str(e),
                }


# ---------------------------------------------------------------------------
# Core pipeline: ingest
# ---------------------------------------------------------------------------

async def ingest_persona(
    persona_dir: str,
    graphiti: Graphiti,
) -> dict:
    """Ingest a single persona's sessions into Neo4j.

    Returns dict with stats: {sessions_ingested, time_s, error}
    """
    short_name = POC_PERSONAS[persona_dir]
    gid = group_id_for(short_name)

    sessions = load_persona_sessions(persona_dir)
    episodes = build_raw_episodes(sessions, persona_dir)

    print(f"    Loading {len(sessions)} sessions -> group '{gid}'")
    print(f"    Date range: {sessions[0]['date']} to {sessions[-1]['date']}")

    t0 = time_module.time()

    try:
        await graphiti.add_episode_bulk(
            bulk_episodes=episodes,
            group_id=gid,
            custom_extraction_instructions=CUSTOM_EXTRACTION_INSTRUCTIONS,
        )
    except Exception as e:
        elapsed = time_module.time() - t0
        return {"sessions_ingested": 0, "time_s": elapsed, "error": str(e)}

    elapsed = time_module.time() - t0
    return {"sessions_ingested": len(sessions), "time_s": elapsed, "error": None}


# ---------------------------------------------------------------------------
# Core pipeline: enrich edges
# ---------------------------------------------------------------------------

async def enrich_persona_edges(
    persona_dir: str,
    driver,
    xai_client: AsyncOpenAI,
) -> dict:
    """Classify all RELATES_TO edges for a persona's group_id.

    Returns dict with stats: {total_edges, personal, world_knowledge, errors, time_s}
    """
    short_name = POC_PERSONAS[persona_dir]
    gid = group_id_for(short_name)

    batch_size = 200
    sem = asyncio.Semaphore(40)
    classified = 0
    errors = 0
    wk_count = 0
    personal_count = 0
    t0 = time_module.time()

    # Count total edges for this persona
    count_result = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid RETURN count(e) AS total",
        gid=gid,
    )
    total_edges = count_result.records[0]["total"]
    print(f"    Total edges for {short_name}: {total_edges}")

    # Count already enriched
    already_result = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid AND e.fr_enriched = true RETURN count(e) AS cnt",
        gid=gid,
    )
    already_enriched = already_result.records[0]["cnt"]
    if already_enriched > 0:
        print(f"    Already enriched: {already_enriched}, processing remaining...")

    while True:
        # Fetch unenriched edges
        batch_result = await driver.execute_query(
            """
            MATCH (src)-[e:RELATES_TO]->(tgt)
            WHERE e.group_id = $gid AND (e.fr_enriched IS NULL OR e.fr_enriched = false)
            RETURN e.uuid AS uuid, e.fact AS fact,
                   src.name AS src_name, tgt.name AS tgt_name
            LIMIT $limit
            """,
            gid=gid,
            limit=batch_size,
        )

        records = batch_result.records if hasattr(batch_result, "records") else batch_result
        if not records:
            break

        async def classify_and_write(rec):
            nonlocal errors, wk_count, personal_count
            d = rec.data() if hasattr(rec, "data") else dict(rec)
            uuid = d["uuid"]
            fact = d.get("fact", "") or ""
            src_name = d.get("src_name", "") or ""
            tgt_name = d.get("tgt_name", "") or ""

            async with sem:
                result = await classify_edge_fact(fact, src_name, tgt_name, xai_client)

            if result.get("error"):
                errors += 1

            is_wk = result.get("is_world_knowledge", False)
            if is_wk:
                wk_count += 1
            else:
                personal_count += 1

            # Write to Neo4j
            await driver.execute_query(
                """
                MATCH ()-[e:RELATES_TO]->()
                WHERE e.uuid = $uuid
                SET e.fr_enriched = true,
                    e.fr_primary_category = $cat,
                    e.fr_membership_weights = $weights,
                    e.fr_confidence = $conf,
                    e.fr_emotional_loading = $emo,
                    e.fr_is_world_knowledge = $wk,
                    e.fr_classified_ts = $ts
                """,
                uuid=uuid,
                cat=result["primary_category"],
                weights=json.dumps(result.get("membership_weights", {})),
                conf=result.get("confidence", 0.7),
                emo=result.get("emotional_loading", False),
                wk=is_wk,
                ts=time_module.time(),
            )

        tasks = [classify_and_write(rec) for rec in records]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count gather-level exceptions
        for r in results:
            if isinstance(r, Exception):
                errors += 1
                print(f"    Enrichment error: {r}")

        classified += len(records)
        elapsed = time_module.time() - t0
        print(f"    Enriched {classified}/{total_edges} edges "
              f"({personal_count} personal, {wk_count} WK, {errors} errors) "
              f"[{elapsed:.0f}s]")

    elapsed = time_module.time() - t0
    return {
        "total_edges": classified + already_enriched,
        "world_knowledge": wk_count,
        "personal": personal_count,
        "errors": errors,
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format seconds as Xm Ys."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def print_summary_table(checkpoint: dict, total_elapsed: float = 0):
    """Print a formatted summary table of all personas."""
    print(f"\n{'=' * 85}")
    print("  LIFEMEMBENCH INGESTION SUMMARY")
    print(f"{'=' * 85}")
    header = (f"  {'Persona':<12} {'Ingest':<10} {'Sessions':>8}  {'Edges':>6}  "
              f"{'Personal':>8}  {'WK':>4}  {'Group ID':<24} {'Time':>8}")
    print(header)
    print(f"  {'-' * 12} {'-' * 10} {'-' * 8}  {'-' * 6}  {'-' * 8}  {'-' * 4}  {'-' * 24} {'-' * 8}")

    total_sessions = 0
    total_edges = 0
    total_personal = 0
    total_wk = 0

    for persona_dir, short_name in POC_PERSONAS.items():
        p = checkpoint.get("personas", {}).get(short_name, {})
        ingest = p.get("ingest_status", "pending")
        sessions = p.get("ingest_sessions", 0)
        edges = p.get("enrich_edges", 0)
        personal = p.get("enrich_personal", 0)
        wk = p.get("enrich_world_knowledge", 0)
        t = p.get("ingest_time_s", 0) + p.get("enrich_time_s", 0)
        gid = group_id_for(short_name)

        total_sessions += sessions
        total_edges += edges
        total_personal += personal
        total_wk += wk

        print(f"  {short_name:<12} {ingest:<10} {sessions:>8}  {edges:>6}  "
              f"{personal:>8}  {wk:>4}  {gid:<24} {format_time(t):>8}")

    print(f"  {'-' * 12} {'-' * 10} {'-' * 8}  {'-' * 6}  {'-' * 8}  {'-' * 4}  {'-' * 24} {'-' * 8}")
    print(f"  {'TOTAL':<12} {'':10} {total_sessions:>8}  {total_edges:>6}  "
          f"{total_personal:>8}  {total_wk:>4}")

    if total_elapsed > 0:
        print(f"\n  Total time: {format_time(total_elapsed)}")
    est_cost = total_sessions * COST_PER_SESSION + total_edges * COST_PER_EDGE
    if est_cost > 0:
        print(f"  Est. cost:  ~${est_cost:.2f}")


def show_status():
    """Show current ingestion status from checkpoint."""
    checkpoint = load_checkpoint()
    if not checkpoint.get("personas"):
        print("No ingestion started yet. Run with --all or --persona <name>.")
        return
    print_summary_table(checkpoint)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def run_pipeline(personas: list[str], enrich_only: bool):
    """Main orchestration: ingest then enrich for each persona."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    if not checkpoint['started_at']:
        checkpoint['started_at'] = datetime.now(timezone.utc).isoformat()

    graphiti = get_graphiti_client()
    print("Building graph indices and constraints...")
    await graphiti.build_indices_and_constraints()

    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'testpassword123')
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    total_start = time_module.time()
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    for persona_dir in personas:
        short_name = POC_PERSONAS[persona_dir]
        persona_ckpt = checkpoint["personas"].setdefault(short_name, {})

        # Phase 1: Ingest
        if not enrich_only and persona_ckpt.get("ingest_status") != "completed":
            print(f"\n{'=' * 60}")
            print(f"  INGESTING: {persona_dir} -> lifemembench_{short_name}")
            print(f"{'=' * 60}")

            result = await ingest_persona(persona_dir, graphiti)

            persona_ckpt["ingest_status"] = "completed" if result["error"] is None else "error"
            persona_ckpt["ingest_sessions"] = result["sessions_ingested"]
            persona_ckpt["ingest_time_s"] = result["time_s"]
            persona_ckpt["ingest_error"] = result["error"]
            save_checkpoint(checkpoint)

            if result["error"]:
                print(f"  ERROR: {result['error']}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\n  {MAX_CONSECUTIVE_FAILURES} consecutive failures — halting.")
                    print("  Fix the issue and re-run. Completed personas will be skipped.")
                    break
                print(f"  Skipping enrichment for {short_name}, moving to next persona.")
                continue
            else:
                consecutive_failures = 0
                print(f"  Ingested {result['sessions_ingested']} sessions in {format_time(result['time_s'])}")
        elif not enrich_only:
            print(f"\n  {short_name}: ingestion already completed, skipping.")

        # Phase 2: Enrich edges
        if persona_ckpt.get("enrich_status") != "completed":
            print(f"\n  Enriching edges for {short_name}...")
            result = await enrich_persona_edges(persona_dir, driver, xai_client)

            persona_ckpt["enrich_status"] = "completed" if result["errors"] == 0 else "partial"
            persona_ckpt["enrich_edges"] = result["total_edges"]
            persona_ckpt["enrich_world_knowledge"] = result["world_knowledge"]
            persona_ckpt["enrich_personal"] = result["personal"]
            persona_ckpt["enrich_errors"] = result["errors"]
            persona_ckpt["enrich_time_s"] = result["time_s"]
            persona_ckpt["completed_at"] = datetime.now(timezone.utc).isoformat()
            save_checkpoint(checkpoint)

            print(f"  Enriched: {result['total_edges']} edges "
                  f"({result['personal']} personal, {result['world_knowledge']} WK)")
        else:
            print(f"\n  {short_name}: enrichment already completed, skipping.")

    total_elapsed = time_module.time() - total_start
    print_summary_table(checkpoint, total_elapsed)

    await graphiti.close()
    await driver.close()


async def main():
    parser = argparse.ArgumentParser(
        description="LifeMemBench ingestion pipeline — 8 POC personas into Neo4j/Graphiti"
    )
    parser.add_argument("--all", action="store_true",
                        help="Ingest all 8 POC personas")
    parser.add_argument("--persona", type=str, metavar="DIR_NAME",
                        help="Single persona directory name (e.g., 1_priya)")
    parser.add_argument("--enrich-only", action="store_true",
                        help="Skip ingestion, run edge enrichment only")
    parser.add_argument("--status", action="store_true",
                        help="Show progress for all personas")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.persona:
        if args.persona not in POC_PERSONAS:
            print(f"ERROR: Unknown persona '{args.persona}'")
            print(f"Valid personas: {', '.join(POC_PERSONAS.keys())}")
            sys.exit(1)
        personas = [args.persona]
    elif args.all:
        personas = list(POC_PERSONAS.keys())
    else:
        parser.print_help()
        return

    await run_pipeline(personas, enrich_only=args.enrich_only)


if __name__ == "__main__":
    asyncio.run(main())
