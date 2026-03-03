r"""
llm_answer_judge.py — LLM-verified answer matching for kill gate evaluation.

Replaces fragile substring matching with a two-stage pipeline:
    Stage 1: Keyword pre-filter (free, kills ~90% of garbage)
    Stage 2: LLM judge via Grok (strict YES/NO on survivors)

Outputs answer_oracle.json — the file evaluate_v4.py looks for at:
    LongMemEval/data/full_artifacts/answer_oracle.json

Schema: { "question_id": ["edge_uuid_1", "edge_uuid_2", ...], ... }

Why this exists:
    Substring matching counts "GPT-4" as answer "4", "Sixpenny Corner" as
    answer "six", "90.62%" as answer "7". This inflated answerable count
    from 3 to 8 and made uniform look like it was winning. The LLM judge
    killed 5 false positives and revealed behavioral wins 6/7 alpha values.

Usage:
    python llm_answer_judge.py                # Build oracle for all 25 questions
    python llm_answer_judge.py --force        # Rebuild from scratch (ignore cache)
    python llm_answer_judge.py --dry-run      # Show pre-filter stats, no LLM calls

Cost: ~$0.06 for 25 questions, ~$2-3 for 234 questions.

Prerequisites:
    - Neo4j running with ingested full_234 data
    - .env with OPENAI_API_KEY, XAI_API_KEY
    - Graphiti installed
"""

import asyncio
import json
import hashlib
import os
import re
import sys
import time as time_module
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "LongMemEval" / "data"
ARTIFACTS_DIR = DATA_DIR / "full_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ORACLE_PATH = ARTIFACTS_DIR / "answer_oracle.json"
CACHE_PATH = ARTIFACTS_DIR / "judge_cache.json"
JUDGE_LOG_PATH = ARTIFACTS_DIR / "judge_log.json"

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

for var in ["OPENAI_API_KEY", "XAI_API_KEY"]:
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

# ---------------------------------------------------------------------------
# Graphiti client (same setup as evaluate_v4)
# ---------------------------------------------------------------------------

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


# ===========================================================================
# Candidate pool building (mirrored from evaluate_v4 — must stay in sync)
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
    "looking", "trying", "decide", "whether", "know", "tell", "remember",
    "conversation", "chat", "mentioned", "talked", "discussed",
}


def extract_keywords(text: str, min_len: int = 3) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOPWORDS]


def extract_numbers(text: str) -> list[str]:
    """Extract standalone numbers from text."""
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


@dataclass
class Candidate:
    uuid: str
    fact: str
    source: str
    graphiti_score: float


CYPHER_SOURCE_BASELINES = {
    "cypher_kw": 0.4,
    "cypher_intersect": 0.5,
    "cypher_neighbor": 0.3,
}


async def build_candidate_pool(
    question: str, driver, graphiti, group_id: str
) -> list[Candidate]:
    """Build fat candidate pool — identical to evaluate_v4's version."""
    seen_uuids = set()
    candidates = []

    def add(c: Candidate):
        if c.uuid not in seen_uuids:
            seen_uuids.add(c.uuid)
            candidates.append(c)

    keywords = extract_keywords(question)

    # Strategy 1: Graphiti semantic search (top 50)
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

    # Strategy 2: Cypher keyword search
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
                add(Candidate(d["uuid"], d["fact"], "cypher_kw", CYPHER_SOURCE_BASELINES["cypher_kw"]))
        except Exception:
            pass

    # Strategy 3: Multi-keyword intersection
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
                        add(Candidate(d["uuid"], d["fact"], "cypher_intersect", CYPHER_SOURCE_BASELINES["cypher_intersect"]))
                except Exception:
                    pass

    # Strategy 4: Entity name match -> neighborhood
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
                add(Candidate(d["uuid"], d["fact"], "cypher_neighbor", CYPHER_SOURCE_BASELINES["cypher_neighbor"]))
        except Exception:
            pass

    return candidates


# ===========================================================================
# Stage 1: Keyword pre-filter
# ===========================================================================

def prefilter_candidate(
    question: str,
    answer: str,
    fact: str,
) -> tuple[bool, str]:
    """Fast keyword check. Returns (pass, reason).

    Logic:
        - Extract keywords from question, answer, and fact
        - Extract numbers from answer and fact
        - Pass if: (>=1 question keyword in fact) AND (>=1 answer keyword in fact)
        - Also pass if: >=2 question keywords in fact (catches topical relevance
          even when answer is numeric and not literally present)
        - Also pass if: answer contains a number AND that exact number appears
          as a standalone number in the fact (not as substring of another number)

    This kills ~90% of garbage while keeping real matches.
    """
    fact_lower = fact.lower()
    q_kws = extract_keywords(question)
    a_kws = extract_keywords(answer)

    q_hits = [kw for kw in q_kws if kw in fact_lower]
    a_hits = [kw for kw in a_kws if kw in fact_lower]

    # Number matching: check if answer numbers appear as standalone in fact
    a_numbers = extract_numbers(answer)
    f_numbers = extract_numbers(fact)
    number_match = bool(a_numbers and set(a_numbers) & set(f_numbers))

    # Pass conditions
    if len(q_hits) >= 1 and len(a_hits) >= 1:
        return True, f"q_kw={q_hits[:3]} a_kw={a_hits[:3]}"

    if len(q_hits) >= 2:
        return True, f"q_kw_strong={q_hits[:3]}"

    if len(q_hits) >= 1 and number_match:
        matching_nums = list(set(a_numbers) & set(f_numbers))
        return True, f"q_kw={q_hits[:3]} num={matching_nums}"

    # Special case: answer is very short (1-2 words) and appears as whole word
    a_words = answer.lower().split()
    if len(a_words) <= 2:
        for w in a_words:
            if len(w) >= 3 and re.search(r"\b" + re.escape(w) + r"\b", fact_lower):
                if q_hits:
                    return True, f"q_kw={q_hits[:3]} whole_word={w}"

    return False, f"q_kw={q_hits[:2]} a_kw={a_hits[:2]} nums={a_numbers[:2]}"


# ===========================================================================
# Stage 2: LLM judge
# ===========================================================================

JUDGE_SYSTEM = """You verify whether a knowledge graph fact contains answer-relevant information for a question about a user's personal history.

Rules:
- Say YES if the fact contains ANY specific piece of information that is part of the expected answer. It does NOT need to contain the entire answer.
- For list-type answers (multiple items), a fact containing even ONE item from the list counts as YES.
- The information must be SPECIFIC, not just topically related. "User upgraded RAM" is NOT a YES for "How much RAM?" because the amount (16GB) is missing.
- Numbers must match. "User owns a road bike" is NOT a YES for "How many bikes?" because no count is given.
- Future plans are NOT the same as past events. "User plans to visit Paris" is NOT a YES for "Where did user GO?" because planning ≠ going.
- Respond with exactly one word: YES or NO."""

JUDGE_USER = """Question: {question}
Expected answer: {answer}

Fact from knowledge graph: {fact}

Does this fact contain any specific information that is part of the expected answer? Even one matching item from a list answer counts. Respond YES or NO."""


def _cache_key(question_id: str, edge_uuid: str) -> str:
    """Deterministic cache key for a (question, edge) pair."""
    raw = f"{question_id}||{edge_uuid}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def judge_one(
    question: str,
    answer: str,
    fact: str,
    question_id: str,
    edge_uuid: str,
    xai_client: AsyncOpenAI,
    cache: dict,
    sem: asyncio.Semaphore,
) -> tuple[bool, str]:
    """Ask the LLM: does this fact answer this question? Returns (is_answer, raw_response)."""

    key = _cache_key(question_id, edge_uuid)
    if key in cache:
        cached = cache[key]
        return cached["verdict"], f"CACHED:{cached['raw']}"

    async with sem:
        try:
            resp = await xai_client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_USER.format(
                        question=question, answer=answer, fact=fact[:500],
                    )},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip().upper()
            verdict = raw.startswith("YES")
            cache[key] = {"verdict": verdict, "raw": raw, "question_id": question_id, "edge_uuid": edge_uuid}
            return verdict, raw
        except Exception as e:
            return False, f"ERROR:{e}"


# ===========================================================================
# Main pipeline
# ===========================================================================

async def run_judge(force: bool = False, dry_run: bool = False):
    """Build answer_oracle.json by judging all candidates for all questions."""

    # Load questions
    q_path = ARTIFACTS_DIR / "full_questions.json"
    if not q_path.exists():
        print("ERROR: full_questions.json not found. Run ingest_full.py first.")
        sys.exit(1)

    questions = json.load(open(q_path, encoding="utf-8"))
    print(f"Loaded {len(questions)} questions")

    # Load cache
    cache = {}
    if not force and CACHE_PATH.exists():
        try:
            cache = json.load(open(CACHE_PATH, encoding="utf-8"))
            print(f"Loaded {len(cache)} cached judgments")
        except Exception:
            cache = {}

    # Connect
    graphiti = get_graphiti_client()
    group_id = "full_234"

    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    sem = asyncio.Semaphore(30)

    # Results
    oracle: dict[str, list[str]] = {}
    log: list[dict] = []
    total_candidates = 0
    total_prefilter_pass = 0
    total_llm_calls = 0
    total_verified = 0

    t0 = time_module.time()

    try:
        for qi, q in enumerate(questions):
            question_id = q["question_id"]
            question_text = q["question"]
            answer = str(q["answer"])

            print(f"\n[{qi+1}/{len(questions)}] {question_text[:70]}...")
            print(f"  Answer: {answer[:60]}")

            # Build candidate pool (same as evaluate_v4)
            pool = await build_candidate_pool(
                question_text, graphiti.driver, graphiti, group_id,
            )
            total_candidates += len(pool)
            sources = Counter(c.source for c in pool)
            print(f"  Pool: {len(pool)} candidates ({dict(sources)})")

            # Stage 1: Pre-filter
            survivors = []
            for c in pool:
                passed, reason = prefilter_candidate(question_text, answer, c.fact)
                if passed:
                    survivors.append((c, reason))

            total_prefilter_pass += len(survivors)
            print(f"  Pre-filter: {len(survivors)}/{len(pool)} passed")

            if dry_run:
                # Show what would be judged
                for c, reason in survivors[:5]:
                    print(f"    [{reason}] {c.fact[:80]}")
                if len(survivors) > 5:
                    print(f"    ... and {len(survivors) - 5} more")
                continue

            # Stage 2: LLM judge on survivors (parallel)
            verified_uuids = []
            question_log = []

            judge_coros = [
                judge_one(
                    question_text, str(answer), c.fact,
                    question_id, c.uuid,
                    xai_client, cache, sem,
                )
                for c, _ in survivors
            ]
            judge_results = await asyncio.gather(*judge_coros, return_exceptions=True)

            for (c, prefilter_reason), result in zip(survivors, judge_results):
                if isinstance(result, Exception):
                    verdict, raw = False, f"ERROR:{result}"
                else:
                    verdict, raw = result
                total_llm_calls += 1 if not raw.startswith("CACHED:") else 0

                entry = {
                    "edge_uuid": c.uuid,
                    "fact": c.fact[:200],
                    "source": c.source,
                    "prefilter": prefilter_reason,
                    "verdict": verdict,
                    "raw": raw,
                }
                question_log.append(entry)

                if verdict:
                    verified_uuids.append(c.uuid)
                    total_verified += 1
                    print(f"  ✅ {c.fact[:80]}")

            oracle[question_id] = verified_uuids
            log.append({
                "question_id": question_id,
                "question": question_text[:100],
                "answer": answer[:100],
                "pool_size": len(pool),
                "prefilter_survivors": len(survivors),
                "verified_count": len(verified_uuids),
                "judgments": question_log,
            })

            if not verified_uuids:
                print(f"  ❌ No verified answer edges (not answerable)")

    finally:
        await graphiti.close()

        # Always save oracle + cache on exit (even on Ctrl+C)
        if not dry_run and oracle:
            json.dump(oracle, open(ORACLE_PATH, "w"), indent=2)
            print(f"\nOracle saved to {ORACLE_PATH} ({len(oracle)} questions)")
        if cache:
            json.dump(cache, open(CACHE_PATH, "w"), indent=2)
            print(f"Cache saved to {CACHE_PATH} ({len(cache)} entries)")

    elapsed = time_module.time() - t0

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN COMPLETE")
        print(f"Total candidates: {total_candidates}")
        print(f"Pre-filter survivors: {total_prefilter_pass} ({100*total_prefilter_pass/max(1,total_candidates):.0f}%)")
        print(f"{'='*60}")
        return

    # Save detailed log
    json.dump(log, open(JUDGE_LOG_PATH, "w"), indent=2)
    print(f"Detailed log saved to {JUDGE_LOG_PATH}")

    # Summary
    answerable = sum(1 for uuids in oracle.values() if uuids)
    print(f"\n{'='*60}")
    print(f"LLM ANSWER JUDGE SUMMARY")
    print(f"{'='*60}")
    print(f"Questions:          {len(questions)}")
    print(f"Answerable:         {answerable}/{len(questions)} ({100*answerable/len(questions):.0f}%)")
    print(f"Total candidates:   {total_candidates}")
    print(f"Pre-filter pass:    {total_prefilter_pass} ({100*total_prefilter_pass/max(1,total_candidates):.0f}%)")
    print(f"LLM calls (new):    {total_llm_calls}")
    print(f"Verified edges:     {total_verified}")
    print(f"Time:               {elapsed:.0f}s")

    # Cost estimate (Grok pricing: $0.20/M input, $0.50/M output)
    # ~300 tokens in, ~3 tokens out per call
    est_cost = total_prefilter_pass * (300 * 0.20 + 3 * 0.50) / 1_000_000
    print(f"Est. cost:          ${est_cost:.4f}")

    # Per-question breakdown
    print(f"\nPer-question results:")
    for entry in log:
        qid = entry["question_id"]
        v = entry["verified_count"]
        p = entry["pool_size"]
        s = entry["prefilter_survivors"]
        status = "✅" if v > 0 else "❌"
        print(f"  {status} {qid}: {v} verified / {s} survived / {p} pool | {entry['question'][:50]}")

    print(f"\n{'='*60}")
    print(f"Now run: python evaluate_v4.py --evaluate --sweep")
    print(f"{'='*60}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    dry_run = "--dry-run" in args

    if "--help" in args:
        print("Usage:")
        print("  python llm_answer_judge.py              # Build answer oracle")
        print("  python llm_answer_judge.py --force       # Rebuild (ignore cache)")
        print("  python llm_answer_judge.py --dry-run     # Pre-filter stats only")
        sys.exit(0)

    asyncio.run(run_judge(force=force, dry_run=dry_run))