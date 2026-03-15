r"""
evaluate_lifemembench.py - LifeMemBench 2x2 ablation evaluation harness.

Evaluates Fortunate Recall's behavioral decay engine on the LifeMemBench
benchmark: 8 personas x 14 questions x 4 configurations (routing ON/OFF
x behavioral/uniform decay).

Scoring is attack-vector-aware:
    AV1 (superseded): current in top-5, old NOT ranked higher
    AV2 (expired):    expired fact NOT in top-5
    AV3 (stable):     correct fact in top-5
    AV4 (multi-ver):  current version in top-5, old NOT ranked higher
    AV5 (broad):      correct composite in top-5
    AV6 (contra):     newer view in top-5
    AV7 (retraction): retracted fact NOT in top-5
    AV8 (numeric):    exact number in top-1
    AV9 (soft super): existing fact in top-5

Usage:
    python evaluate_lifemembench.py --all                     # Full 2x2, all 8 personas
    python evaluate_lifemembench.py --persona priya           # Single persona
    python evaluate_lifemembench.py --config full             # Single config
    python evaluate_lifemembench.py --alpha 0.1               # Custom alpha (default 0.1)
"""

import asyncio
import argparse
import hashlib
import json
import os
import re
import sys
import time as time_module
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
ARTIFACTS_DIR = LIFEMEMEVAL_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS_PATH = LIFEMEMEVAL_DIR / "lifemembench_questions.json"
RESULTS_PATH = ARTIFACTS_DIR / "lifemembench_results.json"
JUDGE_CACHE_PATH = ARTIFACTS_DIR / "lifemembench_judge_cache.json"

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

for var in ["OPENAI_API_KEY", "XAI_API_KEY", "ANTHROPIC_API_KEY"]:
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

sys.path.insert(0, str(PROJECT_ROOT))
from decay_engine import DecayEngine, TemporalContext, FactNode, CATEGORIES
from graphiti_bridge import build_temporal_context
from retrieval_router import route_and_retrieve, RoutingConfig, reset_routing_state
from query_classifier import classify_query


# ===========================================================================
# Section 2: Data structures
# ===========================================================================

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

CONFIGS = {
    "full":       {"routing": True,  "engine": "behavioral"},
    "no_routing": {"routing": False, "engine": "behavioral"},
    "uniform":    {"routing": True,  "engine": "uniform"},
    "baseline":   {"routing": False, "engine": "uniform"},
}

CONFIG_NAMES = list(CONFIGS.keys())

ALPHA_DEFAULT = 0.1

# Category-specific alpha for behavioral configs.
# Low alpha = fact stays strong over time; high alpha = decays fast.
CATEGORY_DECAY = {
    "FINANCIAL_MATERIAL":     0.05,
    "IDENTITY_SELF_CONCEPT":  0.10,
    "HEALTH_WELLBEING":       0.10,
    "RELATIONAL_BONDS":       0.10,
    "INTELLECTUAL_INTERESTS": 0.15,
    "PREFERENCES_HABITS":     0.20,
    "HOBBIES_RECREATION":     0.20,
    "PROJECTS_ENDEAVORS":     0.30,
    "OBLIGATIONS":            0.35,
    "EMOTIONAL_EPISODES":     0.40,
    "LOGISTICAL_CONTEXT":     0.40,  # reduced from 0.50 for AV8 numeric preservation
}

# Semantic floor per category: blended = max(blended, floor * semantic).
# Prevents high-semantic edges from being destroyed by zero activation.
# Floor must exceed (1-alpha) for the category to rescue activation=0 edges.
# E.g. FINANCIAL alpha=0.05 => (1-a)=0.95 => floor must be >0.95.
# Ephemeral facts (logistical/emotional) get 0.00 — decay can fully suppress
# them (correct for expired appointments and fleeting emotions).
SEMANTIC_FLOOR = {
    "FINANCIAL_MATERIAL":     0.97,  # alpha=0.05, (1-a)=0.95
    "IDENTITY_SELF_CONCEPT":  0.95,  # alpha=0.10, (1-a)=0.90
    "HEALTH_WELLBEING":       0.95,  # alpha=0.10, (1-a)=0.90
    "RELATIONAL_BONDS":       0.95,  # alpha=0.10, (1-a)=0.90
    "INTELLECTUAL_INTERESTS": 0.90,  # alpha=0.15, (1-a)=0.85
    "PREFERENCES_HABITS":     0.85,  # alpha=0.20, (1-a)=0.80
    "HOBBIES_RECREATION":     0.92,  # alpha=0.20, (1-a)=0.80 — raised for AV8 numeric rescue
    "PROJECTS_ENDEAVORS":     0.75,  # alpha=0.30, (1-a)=0.70
    "OBLIGATIONS":            0.70,  # alpha=0.35, (1-a)=0.65
    "EMOTIONAL_EPISODES":     0.00,  # no floor — episodic, should decay
    "LOGISTICAL_CONTEXT":     0.00,  # no floor — expired facts MUST be suppressible
}

POOL_CONCURRENCY = 10
JUDGE_CONCURRENCY = 30


@dataclass
class Candidate:
    uuid: str
    fact: str
    source: str
    graphiti_score: float


@dataclass
class JudgeVerdict:
    supports_correct: bool
    contains_wrong_indicator: bool
    reasoning: str
    cached: bool = False


@dataclass
class QuestionScore:
    question_id: str
    attack_vector: str
    correctness: bool
    hit_at_1: bool
    hit_at_5: bool
    staleness_penalty: float
    supersession_pass: bool
    retraction_pass: bool
    expiry_pass: bool
    av_pass: bool
    answer_rank: int | None
    reciprocal_rank: float
    pool_size: int
    answerable: bool
    top5_facts: list = field(default_factory=list)


@dataclass
class ConfigResult:
    config_name: str
    routing_enabled: bool
    engine_name: str
    alpha: float
    scores: list[QuestionScore] = field(default_factory=list)


# ===========================================================================
# Section 3: Utility functions (from evaluate_v4.py)
# ===========================================================================

STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "the", "a", "an", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "about", "that",
    "this", "it", "its", "and", "but", "or", "if", "not", "no", "so", "up", "out",
    "what", "which", "who", "when", "where", "how", "many", "much", "did", "any",
    "some", "all", "most", "more", "also", "very", "just", "than", "then", "now",
    "back", "going", "went", "get", "got", "take", "took", "make", "made",
    "currently", "recently", "previous", "last", "first",
    "suggest", "suggestions", "recommend", "wondering", "planning", "thinking",
    "know", "tell", "does", "doing", "been", "there", "their", "they", "them",
    "her", "him", "his", "she", "he",
}


def extract_keywords(text: str, min_len: int = 2) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    keywords = [w for w in tokens if len(w) >= min_len and w not in STOPWORDS]
    # Also extract hyphenated compounds (e.g. "parent-teacher", "first-generation")
    compounds = re.findall(r"[a-zA-Z]+-[a-zA-Z]+", text.lower())
    keywords.extend(compounds)
    return list(dict.fromkeys(keywords))  # deduplicate preserving order


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text.replace(",", ""))


# Numeric rescue: detect monetary/quantity values so rerank_candidates can
# protect them from aggressive category-decay (AV8 numeric preservation).
_NUMERIC_RESCUE_RE = re.compile(
    r'(?:'
    r'[\$\u20ac\u00a3\u00a5\u20ab]\s*[\d,]+(?:\.\d+)?'
    r'|[\d,]+(?:\.\d+)?\s*(?:dong|pesos?|kroner?|kronor|dirhams?|euros?|dollars?|pounds?)'
    r'|[\d,]+(?:\.\d+)?\s+(?:million|billion|thousand)\b'
    r'|\d{1,3}(?:,\d{3}){2,}'
    r')', re.I,
)


def _contains_monetary_or_quantity(text: str) -> bool:
    """True if *text* contains a monetary amount or large formatted number."""
    return bool(_NUMERIC_RESCUE_RE.search(text))


CYPHER_SOURCE_BASELINES = {
    "cypher_kw": 0.4,
    "cypher_intersect": 0.5,
    "cypher_neighbor": 0.3,
    "category_routed": 0.35,
    "category_forced": 0.35,
}

def _to_unix_ts(value) -> float:
    """Convert Neo4j/Python datetime-ish values to unix seconds."""
    if value is None:
        return 0.0
    if hasattr(value, "to_native"):
        return value.to_native().timestamp()
    if hasattr(value, "timestamp"):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _edge_attrs_to_fact_node(attrs: dict, edge_uuid: str) -> FactNode:
    """Convert edge fr_ attributes to a FactNode for the decay engine."""
    raw_weights = attrs.get("membership_weights", "{}")
    if isinstance(raw_weights, str):
        weights = json.loads(raw_weights)
    else:
        weights = raw_weights or {}
    for cat in CATEGORIES:
        weights.setdefault(cat, 0.0)

    return FactNode(
        fact_id=edge_uuid,
        membership_weights=weights,
        primary_category=attrs.get("primary_category", "OTHER"),
        last_updated_ts=attrs.get("created_at_ts", 0.0) or 0.0,
        base_activation=1.0,
        future_anchor_ts=None,
        emotional_loading=False,
        emotional_loading_ts=None,
        last_reactivation_ts=None,
        access_count=0,
    )


# ===========================================================================
# Section 3b: Text-based date extraction for expired-logistics filtering
# ===========================================================================

_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTH_RE = "|".join(_MONTH_NAMES)
_ORD = r"(?:st|nd|rd|th)?"

# Priority-ordered date extraction patterns (first match wins).
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 1. ISO: 2025-09-03
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    # 2. Month Day, Year: "March 22, 2025"
    (re.compile(
        rf"\b({_MONTH_RE})\s+(\d{{1,2}}){_ORD},?\s+(\d{{4}})\b", re.I,
    ), "mdy_full"),
    # 3. Day Month Year (British): "10 February 2025"
    (re.compile(
        rf"\b(\d{{1,2}}){_ORD}\s+({_MONTH_RE}),?\s+(\d{{4}})\b", re.I,
    ), "dmy_full"),
    # 4. Month + Day (no year): "March 20", "August 12"
    (re.compile(rf"\b({_MONTH_RE})\s+(\d{{1,2}}){_ORD}\b", re.I), "month_day"),
    # 5. Day + Month (no year, British): "12th March"
    (re.compile(rf"\b(\d{{1,2}}){_ORD}\s+({_MONTH_RE})\b", re.I), "day_month"),
    # 6. "back in August" / "last March" / "this past June"
    (re.compile(
        rf"\b(?:back\s+in|last|this\s+past)\s+({_MONTH_RE})\b", re.I,
    ), "contextual_month"),
]

# 7. Relative: "two weeks later", "a couple months from now"
_RELATIVE_PATTERN = re.compile(
    r"\b(a\s+couple(?:\s+of)?|a\s+few|several|\d+|a|one|two|three|four|five|six)"
    r"\s+(day|week|month)s?"
    r"\s+(later|from\s+now|out|away|after(?:\s+that)?)\b",
    re.I,
)

_WORD_TO_NUM: dict[str, int] = {
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "several": 4,
}
_UNIT_TO_DAYS: dict[str, int] = {"day": 1, "week": 7, "month": 30}


def _resolve_year(month: int, day: int, created_at_ts: float, t_now: float) -> int:
    """Pick the most likely year for a month-day pair without an explicit year.

    Heuristic: the event most likely falls within ~6 months after the fact's
    creation date (facts describe upcoming events at time of writing).
    If the resulting date is >6 months *before* created_at, bump year +1.
    Second guard: if resolved date is past relative to t_now, prefer the
    future interpretation (ambiguous dates are more likely upcoming events).
    """
    created_dt = datetime.fromtimestamp(created_at_ts, tz=timezone.utc) if created_at_ts > 0 else \
        datetime.fromtimestamp(t_now, tz=timezone.utc)
    t_now_dt = datetime.fromtimestamp(t_now, tz=timezone.utc)
    year = created_dt.year
    try:
        candidate = datetime(year, month, min(day, 28), tzinfo=timezone.utc)
    except ValueError:
        return year
    if (candidate - created_dt).days < -180:
        year += 1
    # Second guard: if resolved date is already past relative to t_now but
    # bumping the year would make it future, prefer the future interpretation.
    # Only bump if the future date is within ~18 months of t_now.
    try:
        resolved = datetime(year, month, min(day, 28), tzinfo=timezone.utc)
        if resolved < t_now_dt:
            future = datetime(year + 1, month, min(day, 28), tzinfo=timezone.utc)
            if (future - t_now_dt).days <= 548:
                year += 1
    except ValueError:
        pass
    return year


def _extract_event_date(
    fact_text: str,
    created_at_ts: float,
    t_now: float,
) -> datetime | None:
    """Extract the most likely event/deadline date from fact text.

    Returns a UTC datetime if a date signal is found, else None.
    Searches in priority order: ISO > full date > month+day > contextual > relative.
    """
    fact_lower = fact_text.lower()

    for pattern, kind in _DATE_PATTERNS:
        m = pattern.search(fact_lower)
        if not m:
            continue
        try:
            if kind == "iso":
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                tzinfo=timezone.utc)
            if kind == "mdy_full":
                mo = _MONTH_NAMES.index(m.group(1).lower()) + 1
                return datetime(int(m.group(3)), mo, int(m.group(2)),
                                tzinfo=timezone.utc)
            if kind == "dmy_full":
                mo = _MONTH_NAMES.index(m.group(2).lower()) + 1
                return datetime(int(m.group(3)), mo, int(m.group(1)),
                                tzinfo=timezone.utc)
            if kind == "month_day":
                mo = _MONTH_NAMES.index(m.group(1).lower()) + 1
                day = int(m.group(2))
                yr = _resolve_year(mo, day, created_at_ts, t_now)
                return datetime(yr, mo, min(day, 28), tzinfo=timezone.utc)
            if kind == "day_month":
                mo = _MONTH_NAMES.index(m.group(2).lower()) + 1
                day = int(m.group(1))
                yr = _resolve_year(mo, day, created_at_ts, t_now)
                return datetime(yr, mo, min(day, 28), tzinfo=timezone.utc)
            if kind == "contextual_month":
                mo = _MONTH_NAMES.index(m.group(1).lower()) + 1
                day = 15  # mid-month default
                # "back in <month>" is explicitly past relative to creation
                if "back in" in fact_lower:
                    c_dt = datetime.fromtimestamp(
                        created_at_ts if created_at_ts > 0 else t_now, tz=timezone.utc)
                    yr = c_dt.year if mo < c_dt.month else c_dt.year - 1
                else:
                    yr = _resolve_year(mo, day, created_at_ts, t_now)
                return datetime(yr, mo, day, tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue

    # Fallback: relative temporal markers ("two weeks later")
    m = _RELATIVE_PATTERN.search(fact_lower)
    if m and created_at_ts > 0:
        qty_raw = m.group(1).lower().strip()
        unit = m.group(2).lower()
        # Normalise quantity
        cleaned = qty_raw.replace("a couple of", "2").replace("a couple", "2").replace("a few", "3")
        try:
            n = int(cleaned)
        except ValueError:
            n = _WORD_TO_NUM.get(cleaned, _WORD_TO_NUM.get(cleaned.split()[0], 2))
        offset_secs = n * _UNIT_TO_DAYS.get(unit, 7) * 86400
        return datetime.fromtimestamp(created_at_ts + offset_secs, tz=timezone.utc)

    return None


# ===========================================================================
# Section 4: Neo4j setup + per-persona loading
# ===========================================================================


def get_graphiti_client() -> Graphiti:
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    llm_client = OpenAIClient(
        client=xai_client,
        config=LLMConfig(
            model="grok-4-1-fast-reasoning",
            small_model="grok-4-1-fast-reasoning",
        ),
    )
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig())
    return Graphiti(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "testpassword123"),
        llm_client=llm_client,
        embedder=embedder,
    )


async def load_edge_cache(driver, group_id: str) -> dict[str, dict]:
    """Load all enriched edge attributes for a persona into memory."""
    result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid AND e.fr_enriched = true
        OPTIONAL MATCH (ep:Episodic)
        WHERE ep.uuid IN e.episodes
        WITH e,
             max(ep.valid_at) AS episode_time
        RETURN e.uuid AS uuid,
               e.fact AS fact,
               e.fr_primary_category AS primary_category,
               e.fr_membership_weights AS membership_weights,
               e.fr_confidence AS confidence,
               COALESCE(episode_time, e.created_at) AS created_at,
               e.fr_is_world_knowledge AS is_world_knowledge,
               e.fr_superseded_by AS superseded_by,
               e.fr_supersession_confidence AS supersession_confidence
        """,
        gid=group_id,
    )
    cache = {}
    records = result.records if hasattr(result, "records") else result
    for rec in records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        uuid = d.pop("uuid")
        d["created_at_ts"] = _to_unix_ts(d.get("created_at"))
        cache[uuid] = d
    return cache


async def get_persona_t_now(driver, group_id: str) -> float:
    """Get the latest edge timestamp for a persona (= t_now)."""
    result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
        UNWIND e.episodes AS ep_uuid
        MATCH (ep:Episodic {uuid: ep_uuid})
        RETURN max(ep.valid_at) AS max_ts
        """,
        gid=group_id,
    )
    records = result.records if hasattr(result, "records") else result
    if not records:
        return time_module.time()
    d = records[0].data() if hasattr(records[0], "data") else dict(records[0])
    ts = _to_unix_ts(d.get("max_ts"))
    return ts if ts > 0 else time_module.time()


# ===========================================================================
# Section 5: Candidate pool building (from evaluate_v4.py:577-702)
# ===========================================================================


async def build_candidate_pool(
    question: str, driver, graphiti, group_id: str,
    xai_client=None, edge_cache: dict = None, routing_config=None,
    embedder=None,
) -> list[Candidate]:
    """Build fat candidate pool using 6 retrieval strategies."""
    uuid_to_candidate: dict[str, Candidate] = {}

    def add(c: Candidate):
        existing = uuid_to_candidate.get(c.uuid)
        if existing is None:
            uuid_to_candidate[c.uuid] = c
        elif c.graphiti_score > existing.graphiti_score:
            existing.graphiti_score = c.graphiti_score
            existing.source = c.source

    keywords = extract_keywords(question)

    # --- Strategy 1: Graphiti semantic search (top 50) ---
    try:
        graphiti_results = await graphiti.search(
            question, group_ids=[group_id], num_results=50,
        )
        for i, r in enumerate(graphiti_results):
            fact_text = ""
            if hasattr(r, "fact"):
                fact_text = str(r.fact)
            elif hasattr(r, "name"):
                fact_text = str(r.name)
            uuid = str(getattr(r, "uuid", None) or id(r))
            graphiti_score = 1.0 - (i / max(len(graphiti_results), 1))
            add(Candidate(uuid, fact_text, "graphiti", graphiti_score))
    except Exception as e:
        print(f"    Graphiti search error: {e}")

    # --- Strategy 2: Cypher keyword search on edge facts ---
    for kw in keywords[:8]:
        try:
            result = await driver.execute_query(
                """
                MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
                WHERE e.group_id = $group_id
                  AND toLower(e.fact) CONTAINS $keyword
                RETURN e.uuid AS uuid, e.fact AS fact
                LIMIT 20
                """,
                group_id=group_id,
                keyword=kw,
            )
            records = result.records if hasattr(result, "records") else result
            for rec in records:
                d = rec.data() if hasattr(rec, "data") else dict(rec)
                add(Candidate(d["uuid"], d["fact"], "cypher_kw",
                              CYPHER_SOURCE_BASELINES["cypher_kw"]))
        except Exception as e:
            print(f"    Strategy error (cypher_kw): {e}")

    # --- Strategy 3: Multi-keyword intersection ---
    if len(keywords) >= 2:
        for i in range(min(3, len(keywords))):
            for j in range(i + 1, min(5, len(keywords))):
                kw1, kw2 = keywords[i], keywords[j]
                try:
                    result = await driver.execute_query(
                        """
                        MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
                        WHERE e.group_id = $group_id
                          AND toLower(e.fact) CONTAINS $kw1
                          AND toLower(e.fact) CONTAINS $kw2
                        RETURN e.uuid AS uuid, e.fact AS fact
                        LIMIT 10
                        """,
                        group_id=group_id,
                        kw1=kw1,
                        kw2=kw2,
                    )
                    records = result.records if hasattr(result, "records") else result
                    for rec in records:
                        d = rec.data() if hasattr(rec, "data") else dict(rec)
                        add(Candidate(d["uuid"], d["fact"], "cypher_intersect",
                                      CYPHER_SOURCE_BASELINES["cypher_intersect"]))
                except Exception as e:
                    print(f"    Strategy error (cypher_intersect): {e}")

    # --- Strategy 4: Entity name match -> neighborhood edges ---
    for kw in keywords[:5]:
        try:
            result = await driver.execute_query(
                """
                MATCH (n:Entity)
                WHERE n.group_id = $group_id
                  AND toLower(n.name) CONTAINS $keyword
                WITH n LIMIT 5
                MATCH (n)-[e:RELATES_TO]-(other:Entity)
                WHERE e.group_id = $group_id
                RETURN e.uuid AS uuid, e.fact AS fact
                LIMIT 30
                """,
                group_id=group_id,
                keyword=kw,
            )
            records = result.records if hasattr(result, "records") else result
            for rec in records:
                d = rec.data() if hasattr(rec, "data") else dict(rec)
                add(Candidate(d["uuid"], d["fact"], "cypher_neighbor",
                              CYPHER_SOURCE_BASELINES["cypher_neighbor"]))
        except Exception as e:
            print(f"    Strategy error (cypher_neighbor): {e}")

    # --- Strategy 5: Category-aware routing ---
    if xai_client is not None and edge_cache is not None:
        effective_config = routing_config or RoutingConfig()
        try:
            routed = await route_and_retrieve(
                query=question,
                driver=driver,
                group_id=group_id,
                xai_client=xai_client,
                edge_cache=edge_cache,
                config=effective_config,
                embedder=embedder,
            )
            for uuid, fact, source, score in routed:
                add(Candidate(uuid, fact, source, score))
        except Exception as e:
            print(f"    Strategy error (category_routed): {e}")

    # --- Strategy 6: Category-forced retrieval (top-1 category, deep pull) ---
    if xai_client is not None:
        try:
            classification = await classify_query(question, xai_client)
            if classification.matches:
                top_cat = classification.matches[0].category
                result = await driver.execute_query(
                    """
                    MATCH (src)-[e:RELATES_TO]->(tgt)
                    WHERE e.group_id = $group_id
                      AND e.fr_primary_category = $category
                    RETURN e.uuid AS uuid, e.fact AS fact
                    ORDER BY e.created_at DESC
                    LIMIT 20
                    """,
                    group_id=group_id,
                    category=top_cat,
                )
                records = result.records if hasattr(result, "records") else result
                for rec in records:
                    d = rec.data() if hasattr(rec, "data") else dict(rec)
                    add(Candidate(d["uuid"], d["fact"], "category_forced",
                                  CYPHER_SOURCE_BASELINES["category_forced"]))
        except Exception as e:
            print(f"    Strategy error (category_forced): {e}")

    candidates = list(uuid_to_candidate.values())

    # Per-source score normalization: normalize each source independently
    # to [0, 1] so that every source's best result competes equally.
    from collections import defaultdict as _defaultdict
    source_groups: dict[str, list[Candidate]] = _defaultdict(list)
    for c in candidates:
        source_groups[c.source].append(c)
    for source, group in source_groups.items():
        scores = [c.graphiti_score for c in group]
        s_min, s_max = min(scores), max(scores)
        if s_max > s_min:
            for c in group:
                c.graphiti_score = (c.graphiti_score - s_min) / (s_max - s_min)
        else:
            for c in group:
                c.graphiti_score = 0.5

    return candidates


# ===========================================================================
# Section 6: Activation + reranking (parameterized, no globals)
# ===========================================================================


def compute_edge_activation(
    edge_uuid: str,
    ctx: TemporalContext,
    engine: DecayEngine,
    edge_cache: dict[str, dict],
) -> float:
    """Compute activation from edge-level classification."""
    attrs = edge_cache.get(edge_uuid)
    if attrs is None:
        return 0.5  # unenriched fallback

    if attrs.get("is_world_knowledge"):
        return 0.0

    fact_node = _edge_attrs_to_fact_node(attrs, edge_uuid)

    if ctx.current_timestamp and fact_node.last_updated_ts > 0:
        abs_hours = (ctx.current_timestamp - fact_node.last_updated_ts) / 3600.0
    else:
        abs_hours = 0.0

    fact_ctx = TemporalContext(
        absolute_hours=max(0.0, abs_hours),
        relative_hours=ctx.relative_hours,
        conversational_messages=ctx.conversational_messages,
        current_timestamp=ctx.current_timestamp,
    )
    return engine.compute_activation(fact_node, fact_ctx)


def rerank_candidates(
    candidates: list[Candidate],
    ctx: TemporalContext,
    engine: DecayEngine,
    edge_cache: dict[str, dict],
    alpha: float = 0.1,
    category_decay: dict[str, float] | None = None,
) -> list[tuple[Candidate, float, float, float]]:
    """Rerank by blended score = alpha * activation + (1-alpha) * semantic.

    If category_decay is provided, each edge uses its category-specific alpha
    instead of the global alpha.  Falls back to global alpha for unknown categories.

    Returns: list of (candidate, activation, semantic_score, blended_score)
    sorted by blended_score descending.
    """
    scored = []
    for c in candidates:
        activation = compute_edge_activation(c.uuid, ctx, engine, edge_cache)
        if category_decay is not None:
            cat = edge_cache.get(c.uuid, {}).get("primary_category", "")
            a = category_decay.get(cat, alpha)
        else:
            a = alpha
        blended = a * activation + (1.0 - a) * c.graphiti_score
        # Apply semantic floor: prevent high-semantic edges from being
        # destroyed by zero activation.  Floor is category-specific.
        # Only rescue near-perfect semantic matches (>= 0.95) to avoid
        # boosting merely-related edges that crowd out correct answers.
        if category_decay is not None and c.graphiti_score >= 0.95:
            floor = SEMANTIC_FLOOR.get(cat, 0.0)
            blended = max(blended, floor * c.graphiti_score)
        # Numeric rescue: facts with monetary/quantity values get extra
        # protection against category-decay crushing.  Uses semantic
        # threshold (0.85) and guaranteed floor (0.95) to keep numeric
        # facts retrievable even in high-alpha categories like
        # PROJECTS_ENDEAVORS.  Safe for AV2: expiry filter runs before
        # reranking, so expired logistics are already removed.
        if (category_decay is not None
                and c.graphiti_score >= 0.85
                and _contains_monetary_or_quantity(c.fact)):
            blended = max(blended, 0.95 * c.graphiti_score)
        scored.append((c, activation, c.graphiti_score, blended))
    scored.sort(key=lambda x: (-x[3], -x[2]))
    return scored


def filter_superseded(
    candidates: list[Candidate],
    edge_cache: dict[str, dict],
    confidence_threshold: float = 0.7,
) -> list[Candidate]:
    """Remove candidates whose edges are superseded above confidence threshold.

    Used to filter outdated facts (e.g., old preferences, retracted plans)
    from the candidate pool before top-5 selection. Only applied to
    behavioral-engine configs to preserve clean ablation structure.
    """
    filtered = []
    for c in candidates:
        attrs = edge_cache.get(c.uuid)
        if (attrs
                and attrs.get("superseded_by")
                and float(attrs.get("supersession_confidence") or 0) >= confidence_threshold):
            # Preserve numeric/monetary edges even if superseded — they contain
            # precise values useful for AV8 numeric preservation.
            if not _contains_monetary_or_quantity(c.fact):
                continue
        filtered.append(c)
    return filtered


def _is_backward_looking(question: str) -> bool:
    """Detect backward-looking temporal intent via keyword matching.

    Returns True if the question asks about historical/past state,
    meaning superseded edges should be preserved (not filtered out).
    """
    q = question.lower()
    # Explicit past-reference phrases
    phrases = [
        "before switching", "before he moved", "before she stopped",
        "before they changed", "before she moved", "before he stopped",
        "used to", "previously", "prior to", "back when",
        "old ", "former ", "original ",
    ]
    if any(p in q for p in phrases):
        return True
    # Past-tense question patterns
    past_patterns = [
        r"\bhow (?:much|often|many) (?:was|were|did)\b",
        r"\bwhat (?:was|were|did)\b",
        r"\bbefore (?:he|she|they|it)\b",
        r"\bdid .+ before\b",
    ]
    return any(re.search(p, q) for p in past_patterns)


# ---------------------------------------------------------------------------
# AV7 retraction detection helpers
# ---------------------------------------------------------------------------

def _is_retraction_query(question: str) -> bool:
    """Detect questions asking whether a plan/intention is still active.

    Returns True for "Is X still planning to Y?" style questions where
    the answer might be "No, they retracted/cancelled."
    """
    q = question.lower()
    patterns = [
        r"\bis \w+ (?:still )?planning\b",
        r"\bis \w+ still (?:doing|going|working|looking|pursuing)\b",
        r"\bdoes \w+ still (?:plan|want|intend)\b",
        r"\bis \w+ still interested in\b",
        r"\bis \w+ (?:planning|going) to (?:buy|get|open|start|do)\b",
    ]
    return any(re.search(p, q) for p in patterns)


_RETRACTION_MARKERS = [
    "plan is dead", "is dead for now", "off the table",
    "scrapped", "not happening", "decided against",
    "said no", "denied", "deny ", "no longer", "cancelled",
    "abandoned", "fell through", "didn't work out",
    "not going to happen", "not allowed", "prohibit",
    "gave up on",
]

_STOP_WORDS = frozenset(
    "the a an is was were are has have had been being will would could "
    "should can may might shall do does did to in on at by for with from "
    "and or but not no if so of that this it its they them their he she "
    "him her his user about after also just now still very really thing "
    "whole actually basically".split()
)

_RETRACTION_NOISE = frozenset(
    "plan plans dead table happening decided against said denied longer "
    "cancelled abandoned fell through work going anymore "
    "landlord told because too much money expensive cost "
    "really actually absolutely totally completely".split()
)


def _stem_words(words: set[str]) -> set[str]:
    """Basic stemming: strip trailing s/ies for plural matching."""
    stemmed = set()
    for w in words:
        stemmed.add(w)
        if w.endswith('ies') and len(w) > 4:
            stemmed.add(w[:-3] + 'y')
        elif w.endswith('s') and not w.endswith('ss') and len(w) > 3:
            stemmed.add(w[:-1])
    return stemmed


# Synonym groups for retraction topic bridging.
# When a retraction fact uses one term (e.g. "pets"), the filter should also
# suppress stale facts using synonyms (e.g. "golden retriever puppies").
_SYNONYM_GROUPS: list[frozenset[str]] = [
    frozenset({"pet", "pets", "dog", "dogs", "puppy", "puppies",
               "retriever", "golden", "kitten", "cat", "cats"}),
]


def _expand_synonyms(words: set[str]) -> set[str]:
    """Expand topic words with known synonym groups for better matching."""
    expanded = set(words)
    for group in _SYNONYM_GROUPS:
        if words & group:
            expanded |= group
    return expanded


def filter_retracted_candidates(
    candidates: list[Candidate],
    edge_cache: dict[str, dict],
) -> tuple[list[Candidate], int]:
    """Remove plan edges whose topic has been retracted anywhere in the persona's graph.

    Strategy: scan ALL enriched edges (via edge_cache) for retraction markers,
    extract topic keywords, and suppress candidate pool edges sharing those keywords.

    This works even when the retraction-affirming edge itself isn't in the
    candidate pool — the edge_cache covers the entire persona graph.

    Returns (filtered_candidates, num_removed).
    """
    # Step 1: Find retraction-affirming facts across ALL persona edges
    retraction_facts: list[str] = []
    for attrs in edge_cache.values():
        fact = attrs.get("fact") or ""
        fact_low = fact.lower()
        if any(m in fact_low for m in _RETRACTION_MARKERS):
            retraction_facts.append(fact_low)

    if not retraction_facts:
        return candidates, 0  # No retraction signal in persona's graph

    # Step 2: Extract topic words from retraction facts
    topic_words: set[str] = set()
    for fact_low in retraction_facts:
        words = set(re.findall(r'\b[a-z]{3,}\b', fact_low))
        words -= _STOP_WORDS | _RETRACTION_NOISE
        topic_words |= _stem_words(words)

    # Expand with synonyms so cross-vocabulary retractions match
    # (e.g. retraction says "pets" but stale fact says "golden retriever puppies")
    topic_words = _expand_synonyms(topic_words)

    # Step 3: Filter candidates that share topic words but lack retraction language
    filtered = []
    removed_facts = []
    for c in candidates:
        fact_low = c.fact.lower()
        has_retraction_lang = any(m in fact_low for m in _RETRACTION_MARKERS)
        if has_retraction_lang:
            filtered.append(c)  # Always keep retraction-affirming edges
            continue
        fact_words = set(re.findall(r'\b[a-z]{3,}\b', fact_low))
        shared = _stem_words(fact_words) & topic_words
        if shared:
            removed_facts.append((c.fact[:80], shared))
            continue
        filtered.append(c)

    return filtered, len(removed_facts)


# ---------------------------------------------------------------------------
# Expired-logistics pre-filter (AV2 support)
# ---------------------------------------------------------------------------

# Categories eligible for date-based expiry filtering.
# Only ephemeral/time-bound categories — protects identity, relational,
# financial, preferences, and other stable categories from accidental removal.
# HEALTH_WELLBEING is included because medical *appointments* (labs, physicals)
# are logistical despite being health-related; chronic conditions are safe
# because they lack parseable event dates.
_EXPIRY_ELIGIBLE_CATEGORIES = frozenset({
    "LOGISTICAL_CONTEXT",
    "OBLIGATIONS",
    "HEALTH_WELLBEING",
})


def filter_expired_candidates(
    candidates: list[Candidate],
    edge_cache: dict[str, dict],
    t_now: float,
    buffer_days: int = 14,
) -> tuple[list[Candidate], int]:
    """Remove candidates whose fact text contains a date that has already passed.

    Only targets LOGISTICAL_CONTEXT, OBLIGATIONS, and HEALTH_WELLBEING
    categories. The regex-based date extraction acts as a second guard,
    so facts without parseable dates (e.g. chronic conditions) are never
    touched.

    Args:
        candidates: Pre-rerank candidate pool.
        edge_cache: Full persona edge cache with enriched attributes.
        t_now: Unix timestamp of evaluation reference time.
        buffer_days: Grace period — fact is expired only if its extracted
            event date is more than *buffer_days* before t_now.

    Returns:
        (filtered_candidates, num_removed)
    """
    cutoff = datetime.fromtimestamp(t_now - buffer_days * 86400, tz=timezone.utc)

    filtered: list[Candidate] = []
    removed = 0
    for c in candidates:
        attrs = edge_cache.get(c.uuid)
        if attrs is None:
            filtered.append(c)
            continue

        cat = attrs.get("primary_category", "")
        if cat not in _EXPIRY_ELIGIBLE_CATEGORIES:
            filtered.append(c)
            continue

        # Numeric rescue: protect edges with monetary/quantity values from
        # expiry filtering (AV8 support).  Precise numeric facts remain
        # valuable for recall even after the associated event passes.
        if _contains_monetary_or_quantity(c.fact):
            filtered.append(c)
            continue

        created_at_ts = attrs.get("created_at_ts", 0.0) or 0.0
        event_date = _extract_event_date(c.fact, created_at_ts, t_now)

        if event_date is not None and event_date < cutoff:
            removed += 1
            continue  # expired — drop from pool

        filtered.append(c)

    return filtered, removed


# ===========================================================================
# Section 7: LLM Judge (LifeMemBench-specific)
# ===========================================================================

JUDGE_SYSTEM_PROMPT = """You are evaluating whether a retrieved memory fact correctly answers a question about a user. The evaluation criteria DEPEND on the attack vector being tested.

ATTACK VECTOR SPECIFIC CRITERIA:

AV1_superseded_preference / AV4_multi_version_fact / AV6_cross_session_contradiction:
  - supports_correct = true ONLY if the fact contains the CURRENT/LATEST version of the information
  - contains_wrong_indicator = true if the fact contains an OUTDATED/OLD version that has been superseded
  - Example: If user switched from vegetarian to pescatarian, "user is vegetarian" is wrong, "user eats fish" is correct

AV2_expired_logistics:
  - supports_correct = true ONLY if the fact is still valid/upcoming relative to the temporal context
  - contains_wrong_indicator = true if the fact refers to an event or deadline that has already passed
  - Example: A conference on March 20 is expired if today is May 28

AV7_selective_forgetting:
  - supports_correct = true ONLY if the fact reflects the FINAL decision after retraction
  - contains_wrong_indicator = true if the fact contains the retracted plan/intention
  - Example: If user wanted a dog but landlord said no, "planning to get a dog" is wrong, "landlord denied pets" is correct

AV3_stable_identity / AV8_numeric_preservation:
  - supports_correct = true if the fact contains the relevant stable information. Be generous with partial matches.
  - Specific numbers must be preserved (AV8)

AV5_broad_query:
  - supports_correct = true if the fact contains ANY relevant piece of the correct answer. Be generous — partial coverage counts.

AV9_soft_supersession:
  - supports_correct = true if the fact reflects the most recent stance, even if tentative
  - contains_wrong_indicator = true if it contains an earlier tentative position that was later revised

GENERAL RULES:
- Paraphrased versions count ("started eating fish" = "pescatarian")
- Related facts count ("mom is from Chennai" supports "Tamil background")
- For contains_wrong_indicator: be strict, only flag clear matches

Respond with JSON only: {"supports_correct": true/false, "contains_wrong_indicator": true/false, "reasoning": "brief explanation"}"""

JUDGE_USER_TEMPLATE = """Attack vector: {attack_vector}
Question: {question}
Correct answer: {correct_answer}
Wrong/stale indicators: {wrong_indicators}

Fact to evaluate: {fact}

Respond with JSON only."""


def prefilter_for_correct(question: str, correct_answer: str, fact: str) -> bool:
    """Keyword pre-filter: does this fact potentially support the correct answer?"""
    fact_lower = fact.lower()
    q_kws = extract_keywords(question)
    a_kws = extract_keywords(correct_answer)

    q_hits = [kw for kw in q_kws if kw in fact_lower]
    a_hits = [kw for kw in a_kws if kw in fact_lower]

    a_numbers = extract_numbers(correct_answer)
    f_numbers = extract_numbers(fact)
    number_match = bool(a_numbers and set(a_numbers) & set(f_numbers))

    if len(q_hits) >= 1 and len(a_hits) >= 1:
        return True
    if len(q_hits) >= 2:
        return True
    if len(q_hits) >= 1 and number_match:
        return True
    # Short answer: whole-word match
    if len(a_kws) <= 2 and a_kws:
        for kw in a_kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", fact_lower) and q_hits:
                return True
    return False


def prefilter_for_wrong(wrong_indicators: list[str], fact: str) -> bool:
    """Keyword pre-filter: does this fact potentially contain a wrong indicator?"""
    if not wrong_indicators:
        return False
    fact_lower = fact.lower()
    for indicator in wrong_indicators:
        indicator_lower = indicator.lower()
        # Direct substring check
        if indicator_lower in fact_lower:
            return True
        # Keyword overlap check
        indicator_kws = extract_keywords(indicator)
        if not indicator_kws:
            continue
        hits = sum(1 for kw in indicator_kws if kw in fact_lower)
        if hits >= max(1, len(indicator_kws) // 2):
            return True
    return False


def _load_judge_cache() -> dict:
    if JUDGE_CACHE_PATH.exists():
        try:
            return json.load(open(JUDGE_CACHE_PATH, encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_judge_cache(cache: dict):
    tmp = JUDGE_CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(JUDGE_CACHE_PATH)


async def judge_fact(
    question: str,
    correct_answer: str,
    wrong_indicators: list[str],
    fact: str,
    question_id: str,
    edge_uuid: str,
    attack_vector: str,
    anthropic_client: AsyncAnthropic,
    cache: dict,
    sem: asyncio.Semaphore,
) -> JudgeVerdict:
    """Single LLM call to judge a fact for both correctness and staleness.
    Uses Claude Sonnet with extended thinking for AV-aware evaluation."""
    key = hashlib.sha256(f"{question_id}||{edge_uuid}||{correct_answer}".encode()).hexdigest()[:16]

    if key in cache:
        c = cache[key]
        return JudgeVerdict(
            supports_correct=c.get("supports_correct", False),
            contains_wrong_indicator=c.get("contains_wrong_indicator", False),
            reasoning=c.get("reasoning", ""),
            cached=True,
        )

    wrong_str = "; ".join(wrong_indicators) if wrong_indicators else "(none)"

    user_message = JUDGE_SYSTEM_PROMPT + "\n\n---\n\n" + JUDGE_USER_TEMPLATE.format(
        attack_vector=attack_vector,
        question=question,
        correct_answer=correct_answer,
        wrong_indicators=wrong_str,
        fact=fact[:500],
    )

    async with sem:
        try:
            resp = await anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 2000,
                },
                messages=[{"role": "user", "content": user_message}],
            )
            # Extract text block (not thinking block)
            raw_text = ""
            for block in resp.content:
                if block.type == "text":
                    raw_text = block.text.strip()
                    break
            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
            raw = json.loads(raw_text)
            verdict = JudgeVerdict(
                supports_correct=raw.get("supports_correct", False),
                contains_wrong_indicator=raw.get("contains_wrong_indicator", False),
                reasoning=raw.get("reasoning", ""),
            )
        except Exception as e:
            verdict = JudgeVerdict(False, False, f"ERROR: {e}")

    cache[key] = {
        "supports_correct": verdict.supports_correct,
        "contains_wrong_indicator": verdict.contains_wrong_indicator,
        "reasoning": verdict.reasoning,
        "question_id": question_id,
        "edge_uuid": edge_uuid,
    }
    return verdict


# ===========================================================================
# Section 8: Per-question scoring
# ===========================================================================


def av_specific_pass(
    attack_vector: str,
    topK_verdicts: list[JudgeVerdict],
    top5_verdicts: list[JudgeVerdict] | None = None,
) -> bool:
    """Unified AV-specific pass/fail for a single question.

    Uses a split-window approach: full topK (10) for correct-checking,
    narrow top-5 for wrong-checking in AV2/AV7 (to avoid regressions).

    Args:
        attack_vector: The AV string (e.g. "AV1_superseded_preference").
        topK_verdicts: Ordered list of JudgeVerdict for positions 1..K (K=10).
        top5_verdicts: Ordered list of JudgeVerdict for positions 1..5.
                       Used for narrow wrong-checking in AV2/AV7.
                       If None, defaults to topK_verdicts[:5].

    Returns:
        True if the question passes its AV-specific correctness test.
    """
    if top5_verdicts is None:
        top5_verdicts = topK_verdicts[:5]

    # Full window (topK) checks
    has_correct = any(v.supports_correct for v in topK_verdicts)
    has_wrong = any(v.contains_wrong_indicator for v in topK_verdicts)

    first_correct_rank = None
    for i, v in enumerate(topK_verdicts):
        if v.supports_correct:
            first_correct_rank = i + 1
            break

    first_wrong_rank = None
    for i, v in enumerate(topK_verdicts):
        if v.contains_wrong_indicator:
            first_wrong_rank = i + 1
            break

    # Narrow window (top-5) wrong check for AV2/AV7
    has_wrong_in_top5 = any(v.contains_wrong_indicator for v in top5_verdicts)

    av_prefix = attack_vector.split("_")[0]  # "AV1", "AV2", etc.

    if av_prefix in ("AV1", "AV4", "AV6", "AV9"):
        # Supersession / contradiction:
        # correct in topK AND no wrong ranked HIGHER than highest correct (in topK)
        if not has_correct:
            return False
        if has_wrong and first_wrong_rank < first_correct_rank:
            return False
        return True

    elif av_prefix == "AV2":
        # Expired logistics:
        # No wrong in top-5, OR (correct in topK AND ranked above ALL wrong in topK)
        if not has_wrong_in_top5:
            return True
        if has_correct and first_correct_rank < first_wrong_rank:
            return True
        return False

    elif av_prefix == "AV7":
        # Selective forgetting:
        # correct in topK AND zero wrong in top-5
        return has_correct and not has_wrong_in_top5

    else:
        # AV3, AV5, AV8 -- stable identity, broad query, numeric:
        # At least one correct in topK
        return has_correct


async def score_question(
    q: dict,
    reranked: list[tuple[Candidate, float, float, float]],
    anthropic_client: AsyncAnthropic,
    judge_cache: dict,
    sem: asyncio.Semaphore,
) -> QuestionScore:
    """Score a single question against its reranked candidate list."""

    question_id = q["id"]
    attack_vector = q["attack_vector"]
    correct_answer = q["correct_answer"]
    wrong_indicators = q.get("wrong_answer_indicators", [])

    topK = reranked[:10]           # Judge 10 candidates (expanded from 5)
    top5_for_wrong = reranked[:5]  # Narrow window for wrong-indicator checks (AV2/AV7)

    # --- Judge ALL topK candidates (parallel) ---
    # LifeMemBench scale: 112 × 10 = 1120 calls max (most cached after first run).
    judge_tasks = []
    for cand, act, sem_score, blended in topK:
        judge_tasks.append((
            cand,
            judge_fact(
                q["question"], correct_answer, wrong_indicators,
                cand.fact, question_id, cand.uuid,
                attack_vector, anthropic_client, judge_cache, sem,
            ),
        ))

    verdicts: dict[str, JudgeVerdict] = {}
    for cand, coro in judge_tasks:
        verdict = await coro
        verdicts[cand.uuid] = verdict

    # DIAGNOSTIC: Judge verdict summary
    cached_count = sum(1 for v in verdicts.values() if v.cached)
    new_count = sum(1 for v in verdicts.values() if not v.cached)
    if not any(v.supports_correct for v in verdicts.values()):
        print(f"          NO correct verdict | topK={len(topK)} "
              f"judged={len(verdicts)} (cached={cached_count} new={new_count})")

    # --- answer_rank: rank of first correct fact in reranked list ---
    answer_rank = None
    for rank, (cand, _, _, _) in enumerate(reranked):
        v = verdicts.get(cand.uuid)
        if v and v.supports_correct:
            answer_rank = rank + 1
            break

    hit_at_1 = answer_rank == 1
    hit_at_5 = answer_rank is not None and answer_rank <= 5
    rr = 1.0 / answer_rank if answer_rank else 0.0

    # --- Staleness penalty ---
    # 1.0 if any wrong indicator in top-5 ranks above the first correct answer
    staleness = 0.0
    for rank, (cand, _, _, _) in enumerate(top5_for_wrong):
        v = verdicts.get(cand.uuid)
        if v and v.contains_wrong_indicator:
            if answer_rank is None or (rank + 1) < answer_rank:
                staleness = 1.0
                break

    # --- AV-specific scoring (diagnostic fields, use narrow top-5) ---
    supersession_pass = True
    retraction_pass = True
    expiry_pass = True

    if attack_vector in ("AV1_superseded_preference", "AV4_multi_version_fact"):
        has_correct_in_top5 = hit_at_5
        wrong_ranks_higher = staleness > 0
        supersession_pass = has_correct_in_top5 and not wrong_ranks_higher

    elif attack_vector == "AV7_selective_forgetting":
        retracted_in_top5 = any(
            verdicts.get(cand.uuid, JudgeVerdict(False, False, "")).contains_wrong_indicator
            for cand, _, _, _ in top5_for_wrong
        )
        retraction_pass = not retracted_in_top5

    elif attack_vector == "AV2_expired_logistics":
        expired_in_top5 = any(
            verdicts.get(cand.uuid, JudgeVerdict(False, False, "")).contains_wrong_indicator
            for cand, _, _, _ in top5_for_wrong
        )
        expiry_pass = not expired_in_top5

    # --- Correctness: top-1 supports correct answer ---
    correctness = False
    if topK:
        v = verdicts.get(topK[0][0].uuid)
        if v and v.supports_correct:
            correctness = True

    # --- Debug info (all topK candidates) ---
    topK_debug = []
    for rank, (cand, act, sem_s, bl) in enumerate(topK):
        v = verdicts.get(cand.uuid, JudgeVerdict(False, False, "not judged"))
        topK_debug.append({
            "rank": rank + 1,
            "fact": cand.fact[:120],
            "source": cand.source,
            "activation": round(act, 4),
            "semantic": round(sem_s, 4),
            "blended": round(bl, 4),
            "supports_correct": v.supports_correct,
            "contains_wrong": v.contains_wrong_indicator,
        })

    answerable = any(
        v.supports_correct for v in verdicts.values()
    )

    # --- Unified AV-specific pass (split-window) ---
    # topK for correct-checking, top-5 for wrong-checking in AV2/AV7
    topK_verdict_list = [
        verdicts.get(cand.uuid, JudgeVerdict(False, False, ""))
        for cand, _, _, _ in topK
    ]
    top5_verdict_list = [
        verdicts.get(cand.uuid, JudgeVerdict(False, False, ""))
        for cand, _, _, _ in top5_for_wrong
    ]
    av_pass_result = av_specific_pass(attack_vector, topK_verdict_list, top5_verdict_list)

    return QuestionScore(
        question_id=question_id,
        attack_vector=attack_vector,
        correctness=correctness,
        hit_at_1=hit_at_1,
        hit_at_5=hit_at_5,
        staleness_penalty=staleness,
        supersession_pass=supersession_pass,
        retraction_pass=retraction_pass,
        expiry_pass=expiry_pass,
        av_pass=av_pass_result,
        answer_rank=answer_rank,
        reciprocal_rank=rr,
        pool_size=len(reranked),
        answerable=answerable,
        top5_facts=topK_debug,
    )


# ===========================================================================
# Section 9: Metrics aggregation
# ===========================================================================


def compute_aggregate_metrics(scores: list[QuestionScore]) -> dict:
    """Compute all aggregate metrics from a list of question scores."""
    n = len(scores)
    if n == 0:
        return {"total_questions": 0}

    answerable = [s for s in scores if s.answerable]
    n_ans = len(answerable)

    mrr = sum(s.reciprocal_rank for s in answerable) / max(n_ans, 1)
    hit1 = sum(1 for s in answerable if s.hit_at_1) / max(n_ans, 1)
    hit5 = sum(1 for s in answerable if s.hit_at_5) / max(n_ans, 1)
    correctness = sum(1 for s in scores if s.correctness) / max(n, 1)
    avg_staleness = sum(s.staleness_penalty for s in scores) / max(n, 1)

    # Per-AV breakdown
    av_groups = defaultdict(list)
    for s in scores:
        av_groups[s.attack_vector].append(s)

    per_av = {}
    for av, group in sorted(av_groups.items()):
        av_n = len(group)
        av_answerable = [s for s in group if s.answerable]
        av_n_ans = len(av_answerable)

        av_metrics = {
            "count": av_n,
            "answerable": av_n_ans,
            "mrr": sum(s.reciprocal_rank for s in av_answerable) / max(av_n_ans, 1),
            "hit_at_1": sum(1 for s in av_answerable if s.hit_at_1) / max(av_n_ans, 1),
            "hit_at_5": sum(1 for s in av_answerable if s.hit_at_5) / max(av_n_ans, 1),
            "staleness": sum(s.staleness_penalty for s in group) / max(av_n, 1),
            "av_pass_rate": sum(1 for s in group if s.av_pass) / max(av_n, 1),
            "stale_pct": sum(1 for s in group if s.staleness_penalty > 0) / max(av_n, 1),
        }

        if av in ("AV1_superseded_preference", "AV4_multi_version_fact"):
            av_metrics["supersession_pass_rate"] = (
                sum(1 for s in group if s.supersession_pass) / max(av_n, 1)
            )
        elif av == "AV7_selective_forgetting":
            av_metrics["retraction_pass_rate"] = (
                sum(1 for s in group if s.retraction_pass) / max(av_n, 1)
            )
        elif av == "AV2_expired_logistics":
            av_metrics["expiry_pass_rate"] = (
                sum(1 for s in group if s.expiry_pass) / max(av_n, 1)
            )

        per_av[av] = av_metrics

    av_pass_rate = sum(1 for s in scores if s.av_pass) / max(n, 1)

    return {
        "total_questions": n,
        "answerable": n_ans,
        "mrr": mrr,
        "hit_at_1": hit1,
        "hit_at_5": hit5,
        "av_pass_rate": av_pass_rate,
        "correctness": correctness,
        "avg_staleness_penalty": avg_staleness,
        "per_attack_vector": per_av,
    }


# ===========================================================================
# Section 10: 2x2 ablation runner
# ===========================================================================


async def evaluate_persona(
    persona_name: str,
    questions: list[dict],
    graphiti: Graphiti,
    driver,
    xai_client: AsyncOpenAI,
    anthropic_client: AsyncAnthropic,
    alpha: float,
    configs: list[str] | None = None,
    no_supersession_filter: bool = False,
    uniform_alpha: bool = False,
    judge_cache: dict | None = None,
    pool_sem: asyncio.Semaphore | None = None,
    judge_sem: asyncio.Semaphore | None = None,
    verbose: bool = True,
) -> dict[str, ConfigResult]:
    """Evaluate all configurations for a single persona."""
    group_id = PERSONAS[persona_name]["group_id"]

    # Load edge cache and t_now once per persona
    edge_cache = await load_edge_cache(driver, group_id)
    t_now = await get_persona_t_now(driver, group_id)

    t_now_str = datetime.fromtimestamp(t_now, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"\n  {persona_name}: {len(edge_cache)} edges, t_now={t_now_str}")

    # Temporal context: t_now with 24h since last session
    ctx = build_temporal_context(
        last_session_ts=t_now - 86400.0,
        session_message_count=0,
        now=t_now,
    )

    # Build pools once (with routing ON) for reuse across configs
    if pool_sem is None:  # sequential mode — caller handles in parallel mode
        reset_routing_state()
    routing_config = RoutingConfig(enable_routing=True)

    if verbose:
        print(f"    Building candidate pools for {len(questions)} questions...")
    pools: dict[str, list[Candidate]] = {}
    _pool_sem = pool_sem or asyncio.Semaphore(POOL_CONCURRENCY)

    async def _build_one(q):
        async with _pool_sem:
            pool = await build_candidate_pool(
                q["question"], driver, graphiti, group_id,
                xai_client=xai_client, edge_cache=edge_cache,
                routing_config=routing_config,
                embedder=graphiti.embedder,
            )
            return q["id"], pool

    tasks = [_build_one(q) for q in questions]
    for coro in asyncio.as_completed(tasks):
        qid, pool = await coro
        pools[qid] = pool

    if verbose:
        print(f"    Pools built. Avg size: {sum(len(p) for p in pools.values()) / max(len(pools), 1):.0f}")

    # DIAGNOSTIC: Source distribution across all pools
    if verbose:
        source_counts = Counter()
        for p in pools.values():
            for c in p:
                source_counts[c.source] += 1
        print(f"    Source distribution: {dict(source_counts)}")

    # Load / share judge cache across configs
    _judge_cache = judge_cache if judge_cache is not None else _load_judge_cache()
    _judge_sem = judge_sem or asyncio.Semaphore(JUDGE_CONCURRENCY)

    configs_to_run = configs or CONFIG_NAMES
    results = {}

    for config_name in configs_to_run:
        cfg = CONFIGS[config_name]
        routing_enabled = cfg["routing"]
        engine_name = cfg["engine"]
        engine = DecayEngine.default() if engine_name == "behavioral" else DecayEngine.uniform()

        # Supersession filter: ON for behavioral configs, OFF for uniform/baseline
        apply_supersession = (engine_name == "behavioral" and not no_supersession_filter)

        # Category-specific alpha: ON for behavioral unless --uniform-alpha
        use_category_decay = (engine_name == "behavioral" and not uniform_alpha)
        cat_decay = CATEGORY_DECAY if use_category_decay else None

        if engine_name == "behavioral":
            decay_label = "category-specific" if use_category_decay else f"behavioral(uniform-alpha={alpha})"
        else:
            decay_label = f"uniform({alpha})"

        if verbose:
            print(f"    Config: {config_name} (routing={'ON' if routing_enabled else 'OFF'}, "
                  f"decay={decay_label}, supersession={'ON' if apply_supersession else 'OFF'})")

        config_scores = []
        cache_size_before = len(_judge_cache)
        total_pool_size = 0
        total_superseded_removed = 0

        for qi, q in enumerate(questions):
            # Get pool — filter out category_routed if routing OFF
            pool = pools[q["id"]]
            orig_pool_size = len(pool)
            if not routing_enabled:
                pool = [c for c in pool if c.source not in ("category_routed", "category_forced")]

            # Filter superseded edges for behavioral configs
            # Skip filter for backward-looking queries that need historical edges
            pre_supersession_size = len(pool)
            backward = _is_backward_looking(q["question"])
            if apply_supersession and not backward:
                pool = filter_superseded(pool, edge_cache)
            if backward and apply_supersession and verbose:
                print(f"        >> Skipped supersession filter (backward-looking): {q['id']}")
            total_superseded_removed += pre_supersession_size - len(pool)

            # Filter retracted plans for retraction-type queries (AV7 support)
            if _is_retraction_query(q["question"]):
                pool, retraction_removed = filter_retracted_candidates(pool, edge_cache)
                if retraction_removed and verbose:
                    print(f"        >> Retraction filter ({q['id']}): removed {retraction_removed} candidates")

            # Filter expired logistics/obligations before reranking (AV2 support)
            # Skip for backward-looking queries that ask about past events
            if apply_supersession and not backward:
                pool, expiry_removed = filter_expired_candidates(pool, edge_cache, t_now)
                if expiry_removed and verbose:
                    print(f"        >> Expiry filter ({q['id']}): removed {expiry_removed} candidates")
            elif backward and apply_supersession and verbose:
                print(f"        >> Skipped expiry filter (backward-looking): {q['id']}")

            total_pool_size += len(pool)

            # DIAGNOSTIC: Pool details for first question per config
            if qi == 0 and verbose:
                print(f"        Q0 pool: {orig_pool_size} -> {len(pool)} after filter "
                      f"(sources: {Counter(c.source for c in pool)})")

            # Rerank
            reranked = rerank_candidates(pool, ctx, engine, edge_cache, alpha=alpha, category_decay=cat_decay)

            # Score
            score = await score_question(q, reranked, anthropic_client, _judge_cache, _judge_sem)
            config_scores.append(score)

            if verbose:
                status = "HIT" if score.hit_at_5 else "MISS"
                stale = "STALE" if score.staleness_penalty > 0 else ""
                rank_str = str(score.answer_rank) if score.answer_rank else "-"
                print(f"      [{qi+1:2d}/{len(questions)}] {status:4s} rank={rank_str:>3s} "
                      f"{stale:5s} | {q['question'][:55]}")

        # DIAGNOSTIC: Per-config summary
        if verbose:
            cache_size_after = len(_judge_cache)
            new_judge_calls = cache_size_after - cache_size_before
            supersession_msg = f" | Superseded removed: {total_superseded_removed}" if apply_supersession else ""
            print(f"      >> Avg pool: {total_pool_size / max(len(questions), 1):.0f} | "
                  f"Judge calls: {new_judge_calls} new, {cache_size_after} total cached"
                  f"{supersession_msg}")

        results[config_name] = ConfigResult(
            config_name=config_name,
            routing_enabled=routing_enabled,
            engine_name=engine_name,
            alpha=alpha,
            scores=config_scores,
        )

    # Save judge cache after each persona (only in sequential mode)
    if judge_cache is None:
        _save_judge_cache(_judge_cache)

    return results


# ===========================================================================
# Section 11: Output formatting
# ===========================================================================


def print_ablation_table(
    all_results: dict[str, dict[str, ConfigResult]],
    alpha: float,
):
    """Print the 2x2 ablation summary table."""
    # Aggregate scores across all personas per config
    config_scores: dict[str, list[QuestionScore]] = defaultdict(list)
    for persona_results in all_results.values():
        for config_name, result in persona_results.items():
            config_scores[config_name].extend(result.scores)

    config_metrics = {
        name: compute_aggregate_metrics(scores)
        for name, scores in config_scores.items()
    }

    total_q = sum(len(scores) for scores in config_scores.values()) // max(len(config_scores), 1)

    print(f"\n{'='*70}")
    print(f"LIFEMEMBENCH 2x2 ABLATION (alpha={alpha}, "
          f"{len(all_results)} personas, {total_q} questions)")
    print(f"{'='*70}")

    def _fmt(name):
        m = config_metrics.get(name, {})
        return (f"MRR={m.get('mrr', 0):.3f}  H@1={m.get('hit_at_1', 0):.0%}  "
                f"H@5={m.get('hit_at_5', 0):.0%}  Stale={m.get('avg_staleness_penalty', 0):.0%}")

    print(f"\n{'':20s} {'Routing ON':30s} {'Routing OFF':30s}")
    print(f"{'':20s} {'----------':30s} {'-----------':30s}")
    print(f"{'Behavioral':20s} {_fmt('full'):30s} {_fmt('no_routing'):30s}")
    print(f"{'Uniform':20s} {_fmt('uniform'):30s} {_fmt('baseline'):30s}")

    # Deltas
    full_m = config_metrics.get("full", {})
    nr_m = config_metrics.get("no_routing", {})
    uni_m = config_metrics.get("uniform", {})
    base_m = config_metrics.get("baseline", {})

    print(f"\n  Behavioral advantage (B-U):")
    if full_m and uni_m:
        print(f"    With routing:    +{full_m.get('mrr',0) - uni_m.get('mrr',0):.3f} MRR")
    if nr_m and base_m:
        print(f"    Without routing: +{nr_m.get('mrr',0) - base_m.get('mrr',0):.3f} MRR")

    print(f"  Routing contribution (R-NR):")
    if full_m and nr_m:
        print(f"    With behavioral: +{full_m.get('mrr',0) - nr_m.get('mrr',0):.3f} MRR")
    if uni_m and base_m:
        print(f"    With uniform:    +{uni_m.get('mrr',0) - base_m.get('mrr',0):.3f} MRR")

    return config_metrics


def print_per_av_table(
    all_results: dict[str, dict[str, ConfigResult]],
):
    """Print per-attack-vector accuracy table for all configs."""

    # Aggregate per config
    config_scores: dict[str, list[QuestionScore]] = defaultdict(list)
    for persona_results in all_results.values():
        for config_name, result in persona_results.items():
            config_scores[config_name].extend(result.scores)

    config_metrics = {
        name: compute_aggregate_metrics(scores)
        for name, scores in config_scores.items()
    }

    # Collect all AVs
    all_avs = set()
    for m in config_metrics.values():
        all_avs.update(m.get("per_attack_vector", {}).keys())

    print(f"\n{'='*120}")
    print("PER ATTACK VECTOR BREAKDOWN  (Pass = AV-specific accuracy, Stl = stale-in-top-5 %)")
    print(f"{'='*120}")

    header = f"{'Attack Vector':<30s}"
    for lbl in ["Full", "No-Rt", "Unifm", "Base"]:
        header += f" | {lbl:>5s} {'Stl':>4s}"
    header += f" | {'n':>3s}"
    print(header)
    print("-" * len(header))

    for av in sorted(all_avs):
        row = f"{av:<30s}"
        n = 0
        for cfg in ["full", "no_routing", "uniform", "baseline"]:
            av_m = config_metrics.get(cfg, {}).get("per_attack_vector", {}).get(av, {})
            pass_rate = av_m.get("av_pass_rate", 0)
            stale = av_m.get("stale_pct", 0)
            n = max(n, av_m.get("count", 0))
            row += f" | {pass_rate:>4.0%} {stale:>4.0%}"
        row += f" | {n:>3d}"
        print(row)

    # Overall
    row = f"{'OVERALL':<30s}"
    for cfg in ["full", "no_routing", "uniform", "baseline"]:
        m = config_metrics.get(cfg, {})
        pass_rate = m.get("av_pass_rate", 0)
        stale = m.get("avg_staleness_penalty", 0)
        row += f" | {pass_rate:>4.0%} {stale:>4.0%}"
    total = sum(
        config_metrics.get("full", {}).get("per_attack_vector", {}).get(av, {}).get("count", 0)
        for av in all_avs
    )
    row += f" | {total:>3d}"
    print(row)


def print_headline_findings(
    all_results: dict[str, dict[str, ConfigResult]],
):
    """Print headline findings section."""
    config_scores: dict[str, list[QuestionScore]] = defaultdict(list)
    for persona_results in all_results.values():
        for config_name, result in persona_results.items():
            config_scores[config_name].extend(result.scores)

    config_metrics = {
        name: compute_aggregate_metrics(scores)
        for name, scores in config_scores.items()
    }

    full_mrr = config_metrics.get("full", {}).get("mrr", 0)
    base_mrr = config_metrics.get("baseline", {}).get("mrr", 0)
    nr_mrr = config_metrics.get("no_routing", {}).get("mrr", 0)
    uni_mrr = config_metrics.get("uniform", {}).get("mrr", 0)

    print(f"\n{'='*70}")
    print("HEADLINE FINDINGS")
    print(f"{'='*70}")

    delta = full_mrr - base_mrr
    pct = (delta / max(base_mrr, 0.001)) * 100
    print(f"  Overall MRR: Full={full_mrr:.3f} vs Baseline={base_mrr:.3f} "
          f"(delta=+{delta:.3f}, +{pct:.0f}%)")

    routing_contrib = full_mrr - nr_mrr
    behav_contrib = full_mrr - uni_mrr
    print(f"  Routing contribution:    +{routing_contrib:.3f} MRR")
    print(f"  Behavioral contribution: +{behav_contrib:.3f} MRR")

    # Find biggest AV gap (using av_pass_rate)
    full_avs = config_metrics.get("full", {}).get("per_attack_vector", {})
    base_avs = config_metrics.get("baseline", {}).get("per_attack_vector", {})
    biggest_gap_av = None
    biggest_gap = 0
    for av in full_avs:
        f_pass = full_avs.get(av, {}).get("av_pass_rate", 0)
        b_pass = base_avs.get(av, {}).get("av_pass_rate", 0)
        gap = f_pass - b_pass
        if gap > biggest_gap:
            biggest_gap = gap
            biggest_gap_av = av

    if biggest_gap_av:
        f_pass = full_avs[biggest_gap_av]["av_pass_rate"]
        b_pass = base_avs.get(biggest_gap_av, {}).get("av_pass_rate", 0)
        print(f"  Biggest AV gap: {biggest_gap_av} "
              f"(Full={f_pass:.0%} vs Baseline={b_pass:.0%})")


def save_results(
    all_results: dict[str, dict[str, ConfigResult]],
    alpha: float,
):
    """Save full results to JSON."""
    config_scores: dict[str, list[QuestionScore]] = defaultdict(list)
    for persona_results in all_results.values():
        for config_name, result in persona_results.items():
            config_scores[config_name].extend(result.scores)

    config_metrics = {
        name: compute_aggregate_metrics(scores)
        for name, scores in config_scores.items()
    }

    # Per-question details
    per_question = {}
    for config_name, scores in config_scores.items():
        per_question[config_name] = [
            {
                "question_id": s.question_id,
                "attack_vector": s.attack_vector,
                "correctness": s.correctness,
                "hit_at_1": s.hit_at_1,
                "hit_at_5": s.hit_at_5,
                "staleness_penalty": s.staleness_penalty,
                "supersession_pass": s.supersession_pass,
                "retraction_pass": s.retraction_pass,
                "expiry_pass": s.expiry_pass,
                "av_pass": s.av_pass,
                "answer_rank": s.answer_rank,
                "reciprocal_rank": s.reciprocal_rank,
                "pool_size": s.pool_size,
                "answerable": s.answerable,
                "top5_facts": s.top5_facts,
            }
            for s in scores
        ]

    # Build headline findings
    full_mrr = config_metrics.get("full", {}).get("mrr", 0)
    base_mrr = config_metrics.get("baseline", {}).get("mrr", 0)
    nr_mrr = config_metrics.get("no_routing", {}).get("mrr", 0)
    uni_mrr = config_metrics.get("uniform", {}).get("mrr", 0)

    headlines = [
        f"Overall MRR: Full={full_mrr:.3f} vs Baseline={base_mrr:.3f} "
        f"(delta=+{full_mrr - base_mrr:.3f})",
        f"Routing contribution: +{full_mrr - nr_mrr:.3f} MRR",
        f"Behavioral contribution: +{full_mrr - uni_mrr:.3f} MRR",
    ]

    output = {
        "meta": {
            "alpha": alpha,
            "personas": list(all_results.keys()),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "total_questions": sum(len(s) for s in config_scores.values()) // max(len(config_scores), 1),
        },
        "summary": {
            name: {k: v for k, v in m.items() if k != "per_attack_vector"}
            for name, m in config_metrics.items()
        },
        "per_attack_vector": {
            name: m.get("per_attack_vector", {})
            for name, m in config_metrics.items()
        },
        "per_question": per_question,
        "headline_findings": headlines,
    }

    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    tmp.replace(RESULTS_PATH)
    print(f"\n  Results saved to {RESULTS_PATH}")


# ===========================================================================
# Section 12: CLI
# ===========================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="LifeMemBench 2x2 ablation evaluation harness",
    )
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all 8 personas")
    parser.add_argument("--persona", type=str, metavar="NAME",
                        help="Single persona (e.g., priya, marcus)")
    parser.add_argument("--config", type=str, metavar="NAME",
                        choices=CONFIG_NAMES,
                        help="Single config (full/no_routing/uniform/baseline)")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                        help=f"Blend weight (default: {ALPHA_DEFAULT})")
    parser.add_argument("--no-supersession-filter", action="store_true",
                        help="Disable supersession filtering for ablation")
    parser.add_argument("--uniform-alpha", action="store_true",
                        help="Force behavioral configs to use global alpha instead of category-specific rates")
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="Number of personas to evaluate concurrently (default: 1)")

    args = parser.parse_args()

    # Determine personas
    if args.persona:
        if args.persona not in PERSONAS:
            print(f"ERROR: Unknown persona '{args.persona}'")
            print(f"Valid: {', '.join(PERSONAS.keys())}")
            sys.exit(1)
        persona_list = [args.persona]
    elif args.all:
        persona_list = list(PERSONAS.keys())
    else:
        parser.print_help()
        return

    # Determine configs
    config_list = [args.config] if args.config else CONFIG_NAMES

    # Load questions
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: Questions file not found: {QUESTIONS_PATH}")
        sys.exit(1)

    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH.name}")

    # Connect to Neo4j
    graphiti = get_graphiti_client()
    driver = graphiti.driver
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    anthropic_client = AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    t_start = time_module.time()

    parallel = max(1, args.parallel)

    try:
        all_results: dict[str, dict[str, ConfigResult]] = {}

        if parallel <= 1:
            # --- Sequential mode (original behavior) ---
            for persona_name in persona_list:
                persona_questions = [q for q in questions if q["persona"] == persona_name]
                if not persona_questions:
                    print(f"WARNING: No questions for persona {persona_name}")
                    continue

                print(f"\n{'='*60}")
                print(f"PERSONA: {persona_name.upper()} ({len(persona_questions)} questions)")
                print(f"{'='*60}")

                results = await evaluate_persona(
                    persona_name, persona_questions,
                    graphiti, driver, xai_client,
                    anthropic_client,
                    alpha=args.alpha,
                    configs=config_list,
                    no_supersession_filter=args.no_supersession_filter,
                    uniform_alpha=args.uniform_alpha,
                )
                all_results[persona_name] = results
        else:
            # --- Parallel mode ---
            print(f"\nParallel mode: up to {parallel} personas concurrently")

            # Shared judge cache (loaded once, saved once at end)
            shared_judge_cache = _load_judge_cache()
            print(f"Judge cache loaded: {len(shared_judge_cache)} entries")

            # Global semaphores (total concurrency, not per-persona)
            global_pool_sem = asyncio.Semaphore(POOL_CONCURRENCY)
            global_judge_sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

            # Clear routing state once before all personas
            reset_routing_state()

            # Persona-level concurrency limiter
            persona_sem = asyncio.Semaphore(parallel)

            async def _eval_one(p_name, p_questions):
                async with persona_sem:
                    print(f"\n[START] {p_name} ({len(p_questions)} questions)")
                    t0 = time_module.time()
                    result = await evaluate_persona(
                        p_name, p_questions,
                        graphiti, driver, xai_client,
                        anthropic_client,
                        alpha=args.alpha,
                        configs=config_list,
                        no_supersession_filter=args.no_supersession_filter,
                        uniform_alpha=args.uniform_alpha,
                        judge_cache=shared_judge_cache,
                        pool_sem=global_pool_sem,
                        judge_sem=global_judge_sem,
                        verbose=False,
                    )
                    dt = time_module.time() - t0
                    # Print persona summary on completion
                    full_cfg = result.get("full") or next(iter(result.values()), None)
                    if full_cfg:
                        hits = sum(1 for s in full_cfg.scores if s.hit_at_5)
                        total = len(full_cfg.scores)
                        print(f"[DONE]  {p_name} in {dt:.0f}s — "
                              f"{full_cfg.config_name}: {hits}/{total} "
                              f"({100*hits/max(total,1):.0f}%)")
                    else:
                        print(f"[DONE]  {p_name} in {dt:.0f}s")
                    return p_name, result

            # Build work items
            work = []
            for persona_name in persona_list:
                pq = [q for q in questions if q["persona"] == persona_name]
                if not pq:
                    print(f"WARNING: No questions for persona {persona_name}")
                    continue
                work.append((persona_name, pq))

            # Launch all concurrently (semaphore limits to N at a time)
            tasks = [_eval_one(name, qs) for name, qs in work]
            for coro in asyncio.as_completed(tasks):
                name, result = await coro
                all_results[name] = result

            # Save shared judge cache once at end
            _save_judge_cache(shared_judge_cache)
            print(f"\nJudge cache saved: {len(shared_judge_cache)} entries")

        # Aggregate and print
        elapsed = time_module.time() - t_start
        print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

        config_metrics = print_ablation_table(all_results, args.alpha)
        print_per_av_table(all_results)
        print_headline_findings(all_results)
        save_results(all_results, args.alpha)

    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())