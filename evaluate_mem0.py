r"""
evaluate_mem0.py - Evaluate Mem0 (open-source, default config) on LifeMemBench.

Runs the same 516 questions across 40 personas through Mem0's search, judges
with the same Claude Sonnet LLM judge, and produces results in the identical
JSON format as evaluate_lifemembench.py for direct comparison.

Mem0 replaces the entire 5-strategy candidate pool + decay engine + reranking
with a single m.search() call. No routing, no decay, no supersession/expiry
filtering. This is the point — Mem0 is a baseline that lacks temporal reasoning.

NOTE: Mem0's default Qdrant vector store is in-memory and does NOT persist
between processes. The --ingest flag ingests sessions in the same process
before evaluation. This is required for the benchmark to work with 100%
default config.

Prerequisites:
    - pip install mem0ai
    - .env with OPENAI_API_KEY and ANTHROPIC_API_KEY

Usage:
    python evaluate_mem0.py --all --ingest              # Full run: ingest + evaluate all 40
    python evaluate_mem0.py --persona priya --ingest    # Single persona
    python evaluate_mem0.py --canary                    # Priya only (auto-ingests)
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time as time_module
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
ARTIFACTS_DIR = LIFEMEMEVAL_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS_PATH = LIFEMEMEVAL_DIR / "lifemembench_questions.json"
MEM0_RESULTS_PATH = ARTIFACTS_DIR / "mem0_results.json"
MEM0_JUDGE_CACHE_PATH = ARTIFACTS_DIR / "mem0_judge_cache.json"

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

for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

from anthropic import AsyncAnthropic
from mem0 import Memory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_CONCURRENCY = 30
MEM0_SEARCH_LIMIT = 10  # top-K results from Mem0 search

PERSONAS = {
    "priya": "1_priya", "marcus": "2_marcus", "elena": "3_elena",
    "david": "4_david", "amara": "5_amara", "jake": "6_jake",
    "fatima": "7_fatima", "tom": "8_tom", "kenji": "9_kenji",
    "rosa": "10_rosa", "callum": "11_callum", "diane": "12_diane",
    "raj": "13_raj", "nadia": "14_nadia", "samuel": "15_samuel",
    "lily": "16_lily", "omar": "17_omar", "bruna": "18_bruna",
    "patrick": "19_patrick", "aisha": "20_aisha",
    "thanh": "21_thanh", "alex": "22_alex", "mirri": "23_mirri",
    "jerome": "24_jerome", "ingrid": "25_ingrid", "dmitri": "26_dmitri",
    "yoli": "27_yoli", "dariush": "28_dariush", "aroha": "29_aroha",
    "mehmet": "30_mehmet", "saga": "31_saga", "kofi": "32_kofi",
    "valentina": "33_valentina", "billy": "34_billy", "pan": "35_pan",
    "marley": "36_marley", "leila": "37_leila", "chenoa": "38_chenoa",
    "joonho": "39_joonho", "zara": "40_zara",
}


# ===========================================================================
# Copied from evaluate_lifemembench.py — Data structures
# ===========================================================================

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


# ===========================================================================
# Copied from evaluate_lifemembench.py — Utility functions
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
    compounds = re.findall(r"[a-zA-Z]+-[a-zA-Z]+", text.lower())
    keywords.extend(compounds)
    return list(dict.fromkeys(keywords))


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text.replace(",", ""))


# ===========================================================================
# Copied from evaluate_lifemembench.py — LLM Judge
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
        if indicator_lower in fact_lower:
            return True
        indicator_kws = extract_keywords(indicator)
        if not indicator_kws:
            continue
        hits = sum(1 for kw in indicator_kws if kw in fact_lower)
        if hits >= max(1, len(indicator_kws) // 2):
            return True
    return False


# ===========================================================================
# Copied from evaluate_lifemembench.py — Judge cache
# ===========================================================================

def _load_judge_cache() -> dict:
    if MEM0_JUDGE_CACHE_PATH.exists():
        try:
            return json.load(open(MEM0_JUDGE_CACHE_PATH, encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_judge_cache(cache: dict):
    tmp = MEM0_JUDGE_CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(MEM0_JUDGE_CACHE_PATH)


# ===========================================================================
# Copied from evaluate_lifemembench.py — judge_fact
# ===========================================================================

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
    """Single LLM call to judge a fact for both correctness and staleness."""
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
            raw_text = ""
            for block in resp.content:
                if block.type == "text":
                    raw_text = block.text.strip()
                    break
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
# Copied from evaluate_lifemembench.py — AV-specific pass/fail
# ===========================================================================

def av_specific_pass(
    attack_vector: str,
    topK_verdicts: list[JudgeVerdict],
    top5_verdicts: list[JudgeVerdict] | None = None,
) -> bool:
    """Unified AV-specific pass/fail for a single question."""
    if top5_verdicts is None:
        top5_verdicts = topK_verdicts[:5]

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

    has_wrong_in_top5 = any(v.contains_wrong_indicator for v in top5_verdicts)

    av_prefix = attack_vector.split("_")[0]

    if av_prefix in ("AV1", "AV4", "AV6", "AV9"):
        if not has_correct:
            return False
        if has_wrong and first_wrong_rank < first_correct_rank:
            return False
        return True

    elif av_prefix == "AV2":
        if not has_wrong_in_top5:
            return True
        if has_correct and first_correct_rank < first_wrong_rank:
            return True
        return False

    elif av_prefix == "AV7":
        return has_correct and not has_wrong_in_top5

    else:
        # AV3, AV5, AV8
        return has_correct


# ===========================================================================
# Copied from evaluate_lifemembench.py — Metrics aggregation
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
# Mem0-specific functions
# ===========================================================================

# ===========================================================================
# Ingestion (runs in same process since Mem0 default Qdrant is in-memory)
# ===========================================================================

# Persona short_name -> directory name
PERSONA_DIRS = {v: k for k, v in PERSONAS.items()}  # e.g. "priya" -> "1_priya"

LIFEMEMEVAL_DIR_FOR_SESSIONS = PROJECT_ROOT / "LifeMemEval"


def load_persona_sessions(persona_dir: str) -> list[dict]:
    """Load all session JSON files for a persona, sorted by session_id."""
    sessions_dir = LIFEMEMEVAL_DIR_FOR_SESSIONS / persona_dir / "sessions"
    if not sessions_dir.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")

    sessions = []
    for session_file in sorted(sessions_dir.glob("session_*.json")):
        with open(session_file, encoding="utf-8") as f:
            sessions.append(json.load(f))

    sessions.sort(key=lambda s: s["session_id"])

    if not sessions:
        raise ValueError(f"No session files found in {sessions_dir}")
    return sessions


def ingest_persona(persona_name: str, m: Memory, verbose: bool = True) -> dict:
    """Ingest all sessions for one persona into Mem0. Returns stats dict."""
    persona_dir = PERSONAS[persona_name]  # e.g. "1_priya"
    sessions = load_persona_sessions(persona_dir)
    t0 = time_module.time()
    sessions_ingested = 0
    consecutive_failures = 0

    for session in sessions:
        sid = session["session_id"]
        turns = session.get("turns", [])
        if not turns:
            continue

        for attempt in range(3):
            try:
                m.add(
                    messages=turns,
                    user_id=persona_name,
                    metadata={"session_id": sid, "date": session.get("date", "")},
                )
                sessions_ingested += 1
                consecutive_failures = 0
                if verbose:
                    print(f"    Session {sid:2d} ({session.get('type', '?'):8s}) — "
                          f"{len(turns)} turns — OK")
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    if verbose:
                        print(f"    Session {sid}: attempt {attempt+1} failed ({e}), "
                              f"retrying in {wait}s...")
                    time_module.sleep(wait)
                else:
                    if verbose:
                        print(f"    Session {sid}: FAILED after 3 attempts: {e}")
                    consecutive_failures += 1
        else:
            if consecutive_failures >= 5:
                if verbose:
                    print(f"  CIRCUIT BREAKER: 5 consecutive failures for {persona_name}")
                break

    elapsed = round(time_module.time() - t0, 1)

    # Count memories
    try:
        all_memories = m.get_all(user_id=persona_name)
        if isinstance(all_memories, dict):
            mem_count = len(all_memories.get("results", []))
        elif isinstance(all_memories, list):
            mem_count = len(all_memories)
        else:
            mem_count = len(all_memories) if all_memories else 0
    except Exception:
        mem_count = -1

    return {"sessions": sessions_ingested, "memories": mem_count, "time_s": elapsed}


def ingest_all_personas(m: Memory, persona_list: list[str],
                        verbose: bool = True, parallel: int = 1) -> dict:
    """Ingest sessions for all specified personas. Returns summary stats."""
    total_t0 = time_module.time()
    total_sessions = 0
    total_memories = 0

    if parallel <= 1:
        # Sequential mode
        for persona_name in persona_list:
            if verbose:
                print(f"\n  INGESTING: {persona_name}")
            stats = ingest_persona(persona_name, m, verbose=verbose)
            total_sessions += stats["sessions"]
            total_memories += max(0, stats["memories"])
            if verbose:
                print(f"  {persona_name}: {stats['sessions']} sessions, "
                      f"{stats['memories']} memories, {stats['time_s']}s")
    else:
        # Parallel mode — each persona has a unique user_id, no write conflicts
        print(f"  Parallel ingestion: {parallel} workers")
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(ingest_persona, name, m, verbose=False): name
                for name in persona_list
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    stats = future.result()
                    total_sessions += stats["sessions"]
                    total_memories += max(0, stats["memories"])
                    print(f"  [DONE] {name}: {stats['sessions']} sessions, "
                          f"{stats['memories']} memories, {stats['time_s']}s")
                except Exception as e:
                    print(f"  [FAIL] {name}: {e}")

    elapsed = time_module.time() - total_t0
    summary = {
        "personas": len(persona_list),
        "total_sessions": total_sessions,
        "total_memories": total_memories,
        "time_s": round(elapsed, 1),
    }
    if verbose:
        print(f"\n  Ingestion complete: {len(persona_list)} personas, "
              f"{total_sessions} sessions, {total_memories} memories, {elapsed:.0f}s")
    return summary


# ===========================================================================
# Mem0 search
# ===========================================================================

def mem0_search(m: Memory, query: str, user_id: str, limit: int = MEM0_SEARCH_LIMIT) -> list[Candidate]:
    """Search Mem0 and convert results to Candidate objects."""
    results = m.search(query=query, user_id=user_id, limit=limit)

    # Handle both dict and list return formats
    if isinstance(results, dict):
        entries = results.get("results", [])
    elif isinstance(results, list):
        entries = results
    else:
        entries = list(results) if results else []

    candidates = []
    for r in entries:
        if isinstance(r, dict):
            uuid = r.get("id", r.get("memory_id", ""))
            fact = r.get("memory", "")
            score = r.get("score", 0.0)
        else:
            uuid = getattr(r, "id", getattr(r, "memory_id", ""))
            fact = getattr(r, "memory", "")
            score = getattr(r, "score", 0.0)

        if not fact:
            continue

        candidates.append(Candidate(
            uuid=str(uuid),
            fact=str(fact),
            source="mem0",
            graphiti_score=float(score) if score is not None else 0.0,
        ))

    return candidates


async def score_question_mem0(
    q: dict,
    candidates: list[Candidate],
    anthropic_client: AsyncAnthropic,
    judge_cache: dict,
    sem: asyncio.Semaphore,
) -> QuestionScore:
    """Score a single question against Mem0 candidates."""

    question_id = q["id"]
    attack_vector = q["attack_vector"]
    correct_answer = q["correct_answer"]
    wrong_indicators = q.get("wrong_answer_indicators", [])

    # Build reranked-format tuples: (cand, activation, semantic, blended)
    # Mem0 has no decay — activation=1.0, semantic=blended=mem0_score
    reranked = [
        (cand, 1.0, cand.graphiti_score, cand.graphiti_score)
        for cand in candidates
    ]

    topK = reranked[:10]
    top5_for_wrong = reranked[:5]

    # Judge ALL topK candidates (parallel)
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

    # Diagnostic
    cached_count = sum(1 for v in verdicts.values() if v.cached)
    new_count = sum(1 for v in verdicts.values() if not v.cached)
    if not any(v.supports_correct for v in verdicts.values()):
        print(f"          NO correct verdict | topK={len(topK)} "
              f"judged={len(verdicts)} (cached={cached_count} new={new_count})")

    # answer_rank: rank of first correct fact
    answer_rank = None
    for rank, (cand, _, _, _) in enumerate(reranked):
        v = verdicts.get(cand.uuid)
        if v and v.supports_correct:
            answer_rank = rank + 1
            break

    hit_at_1 = answer_rank == 1
    hit_at_5 = answer_rank is not None and answer_rank <= 5
    rr = 1.0 / answer_rank if answer_rank else 0.0

    # Staleness penalty
    staleness = 0.0
    for rank, (cand, _, _, _) in enumerate(top5_for_wrong):
        v = verdicts.get(cand.uuid)
        if v and v.contains_wrong_indicator:
            if answer_rank is None or (rank + 1) < answer_rank:
                staleness = 1.0
                break

    # AV-specific diagnostic fields
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

    # Correctness: top-1 supports correct answer
    correctness = False
    if topK:
        v = verdicts.get(topK[0][0].uuid)
        if v and v.supports_correct:
            correctness = True

    # Debug info
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

    answerable = any(v.supports_correct for v in verdicts.values())

    # Unified AV-specific pass (split-window)
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
        pool_size=len(candidates),
        answerable=answerable,
        top5_facts=topK_debug,
    )


async def evaluate_persona_mem0(
    persona_name: str,
    questions: list[dict],
    m: Memory,
    anthropic_client: AsyncAnthropic,
    judge_cache: dict,
    judge_sem: asyncio.Semaphore,
    verbose: bool = True,
) -> list[QuestionScore]:
    """Evaluate all questions for one persona through Mem0."""
    scores = []

    for qi, q in enumerate(questions):
        if verbose:
            print(f"  Q{qi+1}/{len(questions)}: {q['id']} [{q['attack_vector'][:20]}]")

        # Search Mem0 (sequential — Qdrant local is not thread-safe)
        candidates = mem0_search(m, q["question"], user_id=persona_name)

        if verbose:
            print(f"    Mem0 returned {len(candidates)} candidates")

        # Score
        score = await score_question_mem0(
            q, candidates, anthropic_client, judge_cache, judge_sem,
        )

        if verbose:
            status = "PASS" if score.av_pass else "FAIL"
            rank_str = f"rank={score.answer_rank}" if score.answer_rank else "not found"
            print(f"    -> {status} | {rank_str} | staleness={score.staleness_penalty}")

        scores.append(score)

    return scores


# ===========================================================================
# Results saving
# ===========================================================================

def get_mem0_version() -> str:
    """Get installed mem0ai package version."""
    try:
        import importlib.metadata
        return importlib.metadata.version("mem0ai")
    except Exception:
        return "unknown"


def save_results(all_scores: list[QuestionScore], personas_evaluated: list[str]):
    """Save results in same JSON format as evaluate_lifemembench.py."""
    metrics = compute_aggregate_metrics(all_scores)

    per_question = {
        "mem0": [
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
            for s in all_scores
        ]
    }

    output = {
        "meta": {
            "system": "mem0",
            "mem0_version": get_mem0_version(),
            "mem0_defaults": {
                "llm": "gpt-4.1-nano-2025-04-14 (OpenAI default)",
                "embedder": "text-embedding-3-small (OpenAI default)",
                "vector_store": "qdrant (local, default)",
                "note": "100% default config — Memory() with no customization",
            },
            "alpha": None,
            "personas": personas_evaluated,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "total_questions": len(all_scores),
        },
        "summary": {
            "mem0": {k: v for k, v in metrics.items() if k != "per_attack_vector"},
        },
        "per_attack_vector": {
            "mem0": metrics.get("per_attack_vector", {}),
        },
        "per_question": per_question,
        "headline_findings": [
            f"Mem0 (default config): AV-pass={metrics.get('av_pass_rate', 0):.1%}, "
            f"MRR={metrics.get('mrr', 0):.3f}, H@5={metrics.get('hit_at_5', 0):.1%}, "
            f"staleness={metrics.get('avg_staleness_penalty', 0):.3f}",
        ],
    }

    tmp = MEM0_RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    tmp.replace(MEM0_RESULTS_PATH)
    print(f"\n  Results saved to {MEM0_RESULTS_PATH}")


def print_summary(all_scores: list[QuestionScore]):
    """Print summary table."""
    metrics = compute_aggregate_metrics(all_scores)

    print(f"\n{'='*60}")
    print(f"  MEM0 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total questions:   {metrics['total_questions']}")
    print(f"  Answerable:        {metrics['answerable']}")
    print(f"  AV pass rate:      {metrics['av_pass_rate']:.1%}")
    print(f"  MRR:               {metrics['mrr']:.3f}")
    print(f"  Hit@1:             {metrics['hit_at_1']:.1%}")
    print(f"  Hit@5:             {metrics['hit_at_5']:.1%}")
    print(f"  Avg staleness:     {metrics['avg_staleness_penalty']:.3f}")

    print(f"\n  Per attack vector:")
    print(f"  {'AV':<35} {'Count':>5} {'Pass':>6} {'Stale':>6} {'MRR':>6}")
    print(f"  {'-'*58}")
    for av, m in sorted(metrics.get("per_attack_vector", {}).items()):
        print(f"  {av:<35} {m['count']:>5} {m['av_pass_rate']:>5.0%} "
              f"{m['staleness']:>5.2f} {m['mrr']:>5.3f}")


# ===========================================================================
# CLI
# ===========================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Mem0 (default config) on LifeMemBench",
    )
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all 40 personas")
    parser.add_argument("--persona", type=str, metavar="NAME",
                        help="Single persona (e.g., priya)")
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="Parallel ingestion workers (default: 1). Eval is always "
                             "sequential — Qdrant local is not thread-safe.")
    parser.add_argument("--canary", action="store_true",
                        help="Canary: evaluate priya only with detailed output")
    parser.add_argument("--ingest", action="store_true",
                        help="Ingest sessions before evaluating (required: Mem0 default "
                             "Qdrant is in-memory and doesn't persist between processes)")
    parser.add_argument("--ingest-verbose", action="store_true",
                        help="Print per-session details during ingestion")

    args = parser.parse_args()

    # Determine personas
    if args.canary:
        persona_list = ["priya"]
        args.ingest = True  # canary always ingests
    elif args.persona:
        if args.persona not in PERSONAS:
            print(f"ERROR: Unknown persona '{args.persona}'")
            print(f"Valid: {', '.join(sorted(PERSONAS.keys()))}")
            sys.exit(1)
        persona_list = [args.persona]
    elif args.all:
        persona_list = list(PERSONAS.keys())
    else:
        parser.print_help()
        return

    if not args.ingest:
        print("WARNING: Mem0 default Qdrant is in-memory. Data doesn't persist between")
        print("  processes. Use --ingest to ingest sessions first in the same process.")
        print("  Without --ingest, search will return 0 results.\n")

    # Load questions
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: Questions file not found: {QUESTIONS_PATH}")
        sys.exit(1)

    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH.name}")

    # Initialize Mem0 (default config)
    print("Initializing Mem0 (default config)...")
    m = Memory()
    print("  Mem0 ready.")

    # Ingest if requested (must happen in same process as evaluation)
    if args.ingest:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: INGESTION ({len(persona_list)} personas)")
        print(f"{'='*60}")
        ingest_summary = ingest_all_personas(
            m, persona_list, verbose=args.ingest_verbose or args.canary,
            parallel=max(1, args.parallel),
        )
        print(f"\n{'='*60}")
        print(f"  PHASE 2: EVALUATION")
        print(f"{'='*60}")

    # Initialize Anthropic client (for judge)
    anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Load judge cache
    judge_cache = _load_judge_cache()
    print(f"Judge cache loaded: {len(judge_cache)} entries")

    judge_sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    t_start = time_module.time()
    all_scores: list[QuestionScore] = []

    # Evaluation is always sequential for Mem0 search — Qdrant local is not
    # thread-safe and Mem0 internally uses ThreadPoolExecutor for search.
    # Judge calls still run concurrently via asyncio semaphore within each persona.
    for persona_name in persona_list:
        persona_questions = [q for q in questions if q["persona"] == persona_name]
        if not persona_questions:
            print(f"WARNING: No questions for persona {persona_name}")
            continue

        print(f"\n{'='*60}")
        print(f"PERSONA: {persona_name.upper()} ({len(persona_questions)} questions)")
        print(f"{'='*60}")

        scores = await evaluate_persona_mem0(
            persona_name, persona_questions, m,
            anthropic_client, judge_cache, judge_sem,
            verbose=True,
        )
        all_scores.extend(scores)

        # Save cache after each persona
        _save_judge_cache(judge_cache)

        # Persona summary
        p_pass = sum(1 for s in scores if s.av_pass)
        print(f"  -> {persona_name}: {p_pass}/{len(scores)} pass "
              f"({100*p_pass/max(len(scores),1):.0f}%)")

    elapsed = time_module.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Print summary and save
    print_summary(all_scores)
    save_results(all_scores, persona_list)


if __name__ == "__main__":
    asyncio.run(main())
