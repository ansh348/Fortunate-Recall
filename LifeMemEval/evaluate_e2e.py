#!/usr/bin/env python3
r"""
evaluate_e2e.py  -  End-to-end response quality evaluation pipeline.

Measures whether FR's cleaner retrieval context produces better DOWNSTREAM
LLM responses than Mem0's stale context.  Hypothesis: FR's 8% staleness
produces fewer confabulations than Mem0's 27%.

Design principle: the ONLY variable between FR and Mem0 E2E results is the
retrieved context.  Same model, same temperature (0), same system prompt,
same questions.  Any difference in response quality is CAUSED by the
retrieval context.

Pipeline:
    Phase 1 - Generate GPT-5.4 responses using FR / Mem0 retrieved facts
    Phase 2 - Judge every response with Claude Sonnet
    Phase 3 - Compute six analysis tables (A-F)

Usage:
    python evaluate_e2e.py --dry-run                       # 5 questions, inspect
    python evaluate_e2e.py --system both --parallel 25     # full run
    python evaluate_e2e.py --judge-only --parallel 25      # re-judge existing
    python evaluate_e2e.py --model gpt-5.4 --response-tag gpt54_nr
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time as time_module
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS_PATH = SCRIPT_DIR / "lifemembench_questions.json"
FR_RESULTS_PATH = ARTIFACTS_DIR / "lifemembench_results.json"
MEM0_RESULTS_PATH = ARTIFACTS_DIR / "mem0_results.json"

E2E_RESPONSES_PATH = ARTIFACTS_DIR / "e2e_responses.json"
E2E_JUDGMENTS_PATH = ARTIFACTS_DIR / "e2e_judgments.json"
E2E_SUMMARY_PATH = ARTIFACTS_DIR / "e2e_summary.json"

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


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = (
    "You are a personal AI assistant with memory about the user. "
    "Answer the user's question based ONLY on the following facts about them. "
    "Do not make up information beyond what the facts state. "
    "If the facts don't contain enough information to answer confidently, "
    'say "I don\'t have enough information about that." '
    "Keep your response concise \u2014 1-3 sentences."
)

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether an AI assistant's response correctly answers "
    "a question about a user. You will be given the question, what the correct "
    "answer should contain, indicators of wrong/outdated information, and the "
    "assistant's actual response. Evaluate strictly based on the criteria provided."
)

JUDGE_USER_TEMPLATE = """\
Question: {question}

The correct answer should reference: {correct_answer}

The following would indicate an OUTDATED or WRONG answer: {wrong_indicators}

AI Assistant's response:
\"{response}\"

Evaluate this response on two dimensions:

1. CORRECTNESS \u2014 How well does the response answer the question?
   - CORRECT: Response contains the correct answer without stating outdated information as current
   - PARTIAL: Response contains correct information BUT also mentions outdated/wrong facts
   - WRONG: Response confidently states outdated or incorrect information as if it were current
   - ABSTAIN: Response says it doesn't know or doesn't have enough information

2. CONFABULATION \u2014 Does the response confidently present outdated or superseded information as current fact?
   - true: The response states something outdated/wrong as if it were currently true
   - false: The response either gives correct info, abstains, or appropriately hedges

Respond ONLY with this JSON, no other text:
{{"correctness": "CORRECT|PARTIAL|WRONG|ABSTAIN", "confabulation": true|false, "reasoning": "one sentence explanation"}}"""

# Priority AVs for Table E case study selection
TABLE_E_PRIORITY_AVS = {"AV1", "AV2", "AV4", "AV7"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json(path: Path):
    """Load JSON with utf-8 encoding. Exit on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Cannot load {path}: {e}")
        sys.exit(1)


def load_questions() -> list[dict]:
    return load_json(QUESTIONS_PATH)


def load_retrieval_results() -> tuple[dict, dict]:
    """Return (fr_by_qid, mem0_by_qid) dicts keyed by question_id."""
    fr_data = load_json(FR_RESULTS_PATH)
    fr_by_qid = {q["question_id"]: q for q in fr_data["per_question"]["full"]}

    mem0_data = load_json(MEM0_RESULTS_PATH)
    mem0_by_qid = {q["question_id"]: q for q in mem0_data["per_question"]["mem0"]}

    return fr_by_qid, mem0_by_qid


def extract_facts(per_q_result: dict) -> list[str]:
    """Extract fact text strings from top5_facts (which actually holds up to 10)."""
    return [f["fact"] for f in per_q_result.get("top5_facts", [])]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_atomic(path: Path, data: dict):
    """Atomic write: .tmp then .replace()."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# GPT response generation
# ---------------------------------------------------------------------------


def build_generation_messages(facts: list[str], question: str) -> list[dict]:
    facts_block = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(facts))
    user_msg = (
        f"Here are the facts I know about you:\n\n{facts_block}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


async def generate_response(
    openai_client,
    model: str,
    facts: list[str],
    question: str,
    sem: asyncio.Semaphore,
) -> tuple[str, str | None]:
    """Call GPT to generate a response. Returns (text, error_or_none)."""
    messages = build_generation_messages(facts, question)

    for attempt in range(3):
        try:
            async with sem:
                resp = await openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=500,
                )

                text = resp.choices[0].message.content
                if text is None:
                    return "", "GENERATION_ERROR: empty response content"
                return text.strip(), None
        except Exception as e:
            print(f"  [attempt {attempt+1}/3] Generation error: {e}", file=sys.stderr)
            if attempt < 2:
                await asyncio.sleep(2 ** (attempt + 1))
            else:
                return "", f"GENERATION_ERROR: {e}"

    return "", "GENERATION_ERROR: exhausted retries"


async def generate_all_for_persona(
    persona_questions: list[dict],
    fr_by_qid: dict,
    mem0_by_qid: dict,
    openai_client,
    model: str,
    sem: asyncio.Semaphore,
    systems: list[str],
) -> dict[str, dict]:
    """Generate responses for all questions of one persona. Returns {qid: {...}}."""

    async def _gen_one(q: dict) -> tuple[str, dict]:
        qid = q["id"]
        entry: dict = {}

        if "fr" in systems:
            fr_result = fr_by_qid.get(qid)
            if fr_result:
                fr_facts = extract_facts(fr_result)
                text, err = await generate_response(
                    openai_client, model, fr_facts, q["question"], sem
                )
                entry["fr_response"] = text
                entry["fr_facts"] = fr_facts
                entry["fr_error"] = err
            else:
                entry["fr_response"] = ""
                entry["fr_facts"] = []
                entry["fr_error"] = f"No FR result for {qid}"

        if "mem0" in systems:
            m0_result = mem0_by_qid.get(qid)
            if m0_result:
                m0_facts = extract_facts(m0_result)
                text, err = await generate_response(
                    openai_client, model, m0_facts, q["question"], sem
                )
                entry["mem0_response"] = text
                entry["mem0_facts"] = m0_facts
                entry["mem0_error"] = err
            else:
                entry["mem0_response"] = ""
                entry["mem0_facts"] = []
                entry["mem0_error"] = f"No Mem0 result for {qid}"

        return qid, entry

    tasks = [_gen_one(q) for q in persona_questions]
    results = await asyncio.gather(*tasks)
    return {qid: entry for qid, entry in results}


# ---------------------------------------------------------------------------
# Claude Sonnet judge
# ---------------------------------------------------------------------------


async def judge_response(
    question: str,
    correct_answer: str,
    wrong_indicators: list[str],
    response_text: str,
    anthropic_client,
    sem: asyncio.Semaphore,
) -> dict:
    """Judge a single (question, response) pair. Returns {correctness, confabulation, reasoning}."""
    wrong_str = "; ".join(wrong_indicators) if wrong_indicators else "(none)"
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        wrong_indicators=wrong_str,
        response=response_text,
    )

    for attempt in range(3):
        try:
            async with sem:
                resp = await anthropic_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=3000,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 2000,
                    },
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
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

            parsed = json.loads(raw_text)
            correctness = parsed.get("correctness", "JUDGE_ERROR")
            if correctness not in ("CORRECT", "PARTIAL", "WRONG", "ABSTAIN"):
                correctness = "JUDGE_ERROR"
            return {
                "correctness": correctness,
                "confabulation": bool(parsed.get("confabulation", False)),
                "reasoning": parsed.get("reasoning", ""),
            }

        except json.JSONDecodeError:
            if attempt == 0:
                # Retry once on JSON parse failure
                continue
            return {
                "correctness": "JUDGE_ERROR",
                "confabulation": False,
                "reasoning": f"JSON parse failure: {raw_text[:100]}",
            }
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** (attempt + 1))
            else:
                return {
                    "correctness": "JUDGE_ERROR",
                    "confabulation": False,
                    "reasoning": f"API error: {e}",
                }

    return {"correctness": "JUDGE_ERROR", "confabulation": False, "reasoning": "exhausted retries"}


async def judge_all_for_persona(
    persona_questions: list[dict],
    responses_for_tag: dict,
    q_lookup: dict,
    anthropic_client,
    sem: asyncio.Semaphore,
    systems: list[str],
) -> dict[str, dict]:
    """Judge all responses for one persona. Returns {qid: {fr_judgment, mem0_judgment}}."""

    async def _judge_one(q: dict) -> tuple[str, dict]:
        qid = q["id"]
        resp_data = responses_for_tag.get(qid, {})
        entry: dict = {}

        if "fr" in systems and resp_data.get("fr_response"):
            entry["fr_judgment"] = await judge_response(
                q["question"],
                q["correct_answer"],
                q.get("wrong_answer_indicators", []),
                resp_data["fr_response"],
                anthropic_client,
                sem,
            )
        elif "fr" in systems:
            entry["fr_judgment"] = {
                "correctness": "JUDGE_ERROR",
                "confabulation": False,
                "reasoning": "No response to judge",
            }

        if "mem0" in systems and resp_data.get("mem0_response"):
            entry["mem0_judgment"] = await judge_response(
                q["question"],
                q["correct_answer"],
                q.get("wrong_answer_indicators", []),
                resp_data["mem0_response"],
                anthropic_client,
                sem,
            )
        elif "mem0" in systems:
            entry["mem0_judgment"] = {
                "correctness": "JUDGE_ERROR",
                "confabulation": False,
                "reasoning": "No response to judge",
            }

        return qid, entry

    tasks = [_judge_one(q) for q in persona_questions]
    results = await asyncio.gather(*tasks)
    return {qid: entry for qid, entry in results}


# ---------------------------------------------------------------------------
# Orchestration: generation
# ---------------------------------------------------------------------------


async def run_generation(
    questions: list[dict],
    fr_by_qid: dict,
    mem0_by_qid: dict,
    openai_client,
    model: str,
    parallel: int,
    systems: list[str],
    response_tag: str,
    existing: dict,
    force: bool,
    dry_run: bool,
) -> dict:
    """Phase 1: Generate all responses. Returns full responses dict."""
    # Group questions by persona
    persona_qs: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        persona_qs[q["persona"]].append(q)

    if response_tag not in existing:
        existing[response_tag] = {}
    tag_data = existing[response_tag]

    sem = asyncio.Semaphore(parallel)
    persona_list = sorted(persona_qs.keys())
    total = len(persona_list)

    for idx, persona in enumerate(persona_list, 1):
        pqs = persona_qs[persona]
        expected_qids = {q["id"] for q in pqs}

        # Check checkpoint
        if not force and expected_qids.issubset(set(tag_data.keys())):
            print(f"  [{idx}/{total}] {persona}: skipped (checkpoint)")
            continue

        t0 = time_module.time()
        print(f"  [{idx}/{total}] {persona} ({len(pqs)} questions)...", end="", flush=True)

        results = await generate_all_for_persona(
            pqs, fr_by_qid, mem0_by_qid, openai_client, model, sem, systems
        )
        tag_data.update(results)

        # Checkpoint after persona
        if not dry_run:
            save_atomic(E2E_RESPONSES_PATH, existing)

        elapsed = time_module.time() - t0
        fr_ok = sum(1 for r in results.values() if r.get("fr_response") and not r.get("fr_error"))
        m0_ok = sum(1 for r in results.values() if r.get("mem0_response") and not r.get("mem0_error"))
        fr_err = sum(1 for r in results.values() if r.get("fr_error"))
        m0_err = sum(1 for r in results.values() if r.get("mem0_error"))
        parts = []
        if "fr" in systems:
            parts.append(f"FR: {fr_ok}/{len(pqs)}")
        if "mem0" in systems:
            parts.append(f"Mem0: {m0_ok}/{len(pqs)}")
        err_total = fr_err + m0_err
        err_str = f" ({err_total} errors)" if err_total else ""
        print(f" {', '.join(parts)}{err_str} ({elapsed:.1f}s)")

        # Print first error for debugging
        if err_total:
            for r in results.values():
                for ek in ("fr_error", "mem0_error"):
                    if r.get(ek):
                        print(f"    Error sample: {r[ek][:200]}", file=sys.stderr)
                        break
                else:
                    continue
                break

    return existing


# ---------------------------------------------------------------------------
# Orchestration: judging
# ---------------------------------------------------------------------------


async def run_judging(
    questions: list[dict],
    responses: dict,
    anthropic_client,
    parallel: int,
    response_tag: str,
    systems: list[str],
    existing: dict,
    force: bool,
    dry_run: bool,
) -> dict:
    """Phase 2: Judge all responses. Returns full judgments dict."""
    persona_qs: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        persona_qs[q["persona"]].append(q)

    if response_tag not in existing:
        existing[response_tag] = {}
    tag_data = existing[response_tag]
    resp_tag_data = responses.get(response_tag, {})
    q_lookup = {q["id"]: q for q in questions}

    sem = asyncio.Semaphore(parallel)
    persona_list = sorted(persona_qs.keys())
    total = len(persona_list)

    for idx, persona in enumerate(persona_list, 1):
        pqs = persona_qs[persona]
        expected_qids = {q["id"] for q in pqs}

        if not force and expected_qids.issubset(set(tag_data.keys())):
            print(f"  [{idx}/{total}] {persona}: skipped (checkpoint)")
            continue

        t0 = time_module.time()
        print(f"  [{idx}/{total}] {persona} ({len(pqs)} questions)...", end="", flush=True)

        results = await judge_all_for_persona(
            pqs, resp_tag_data, q_lookup, anthropic_client, sem, systems
        )
        tag_data.update(results)

        if not dry_run:
            save_atomic(E2E_JUDGMENTS_PATH, existing)

        elapsed = time_module.time() - t0
        print(f" done ({elapsed:.1f}s)")

    return existing


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def trunc(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def pct(num: int, den: int) -> str:
    if den == 0:
        return "  - "
    return f"{num / den * 100:5.1f}%"


def av_prefix(attack_vector: str) -> str:
    """'AV1_superseded_preference' -> 'AV1'."""
    return attack_vector.split("_")[0]


# ---------------------------------------------------------------------------
# Table A: Overall E2E comparison
# ---------------------------------------------------------------------------


def compute_table_a(questions: list[dict], judgments: dict, systems: list[str]) -> dict:
    table: dict = {}
    for sys_name in systems:
        key = f"{sys_name}_judgment"
        counts = {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0, "ABSTAIN": 0, "JUDGE_ERROR": 0}
        confab_count = 0
        non_abstain = 0
        total = 0

        for q in questions:
            j = judgments.get(q["id"], {}).get(key)
            if not j:
                continue
            total += 1
            c = j["correctness"]
            counts[c] = counts.get(c, 0) + 1
            if c != "ABSTAIN":
                non_abstain += 1
                if j.get("confabulation"):
                    confab_count += 1

        table[sys_name] = {
            "total": total,
            **{k: v for k, v in counts.items()},
            "correct_pct": counts["CORRECT"] / total * 100 if total else 0,
            "partial_pct": counts["PARTIAL"] / total * 100 if total else 0,
            "wrong_pct": counts["WRONG"] / total * 100 if total else 0,
            "abstain_pct": counts["ABSTAIN"] / total * 100 if total else 0,
            "confab_count": confab_count,
            "confab_pct": confab_count / non_abstain * 100 if non_abstain else 0,
        }
    return table


def print_table_a(table: dict, systems: list[str]):
    print("\n  TABLE A: Overall E2E Comparison")
    print("  " + "-" * 58)
    header_parts = ["  Metric              "]
    for s in systems:
        label = "FR context" if s == "fr" else "Mem0 context"
        header_parts.append(f" {label:>12}")
    if len(systems) == 2:
        header_parts.append("     Delta")
    print(" | ".join(header_parts))
    print("  " + "-" * 58)

    metrics = [
        ("Correct rate", "correct_pct"),
        ("Confabulation rate", "confab_pct"),
        ("Abstain rate", "abstain_pct"),
        ("Partial rate", "partial_pct"),
        ("Wrong rate", "wrong_pct"),
    ]
    for label, key in metrics:
        parts = [f"  {label:<20}"]
        vals = []
        for s in systems:
            v = table[s][key]
            vals.append(v)
            parts.append(f" {v:>11.1f}%")
        if len(systems) == 2:
            delta = vals[0] - vals[1]
            sign = "+" if delta >= 0 else ""
            parts.append(f"  {sign}{delta:>5.1f}pp")
        print(" | ".join(parts))

    # Raw counts
    print()
    for s in systems:
        t = table[s]
        label = "FR" if s == "fr" else "Mem0"
        print(f"  {label}: {t['total']} total | "
              f"{t['CORRECT']} correct, {t['PARTIAL']} partial, "
              f"{t['WRONG']} wrong, {t['ABSTAIN']} abstain, "
              f"{t['JUDGE_ERROR']} errors | "
              f"{t['confab_count']} confabulations")


# ---------------------------------------------------------------------------
# Table B: Per-AV E2E comparison
# ---------------------------------------------------------------------------


def compute_table_b(questions: list[dict], judgments: dict, systems: list[str]) -> dict:
    av_groups: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        av_groups[q["attack_vector"]].append(q)

    table: dict = {}
    for av in sorted(av_groups.keys()):
        av_qs = av_groups[av]
        row: dict = {"n": len(av_qs)}
        for sys_name in systems:
            key = f"{sys_name}_judgment"
            correct = confab = non_abstain = 0
            for q in av_qs:
                j = judgments.get(q["id"], {}).get(key)
                if not j:
                    continue
                if j["correctness"] == "CORRECT":
                    correct += 1
                if j["correctness"] != "ABSTAIN":
                    non_abstain += 1
                    if j.get("confabulation"):
                        confab += 1
            n = len(av_qs)
            row[f"{sys_name}_correct"] = correct
            row[f"{sys_name}_correct_pct"] = correct / n * 100 if n else 0
            row[f"{sys_name}_confab"] = confab
            row[f"{sys_name}_confab_pct"] = confab / non_abstain * 100 if non_abstain else 0
        table[av] = row
    return table


def print_table_b(table: dict, systems: list[str]):
    print("\n  TABLE B: Per-AV E2E Comparison (THE MONEY TABLE)")
    print("  " + "-" * 100)
    if len(systems) == 2:
        print(f"  {'AV':<30} {'n':>3} | {'FR corr':>8} {'FR conf':>8} | "
              f"{'M0 corr':>8} {'M0 conf':>8} | {'d corr':>7} {'d conf':>7}")
    else:
        s = systems[0]
        label = "FR" if s == "fr" else "M0"
        print(f"  {'AV':<30} {'n':>3} | {label+' corr':>8} {label+' conf':>8}")
    print("  " + "-" * 100)

    for av in sorted(table.keys()):
        r = table[av]
        av_short = av[:30]
        if len(systems) == 2:
            d_corr = r["fr_correct_pct"] - r["mem0_correct_pct"]
            d_conf = r["fr_confab_pct"] - r["mem0_confab_pct"]
            print(f"  {av_short:<30} {r['n']:>3} | "
                  f"{r['fr_correct_pct']:>7.1f}% {r['fr_confab_pct']:>7.1f}% | "
                  f"{r['mem0_correct_pct']:>7.1f}% {r['mem0_confab_pct']:>7.1f}% | "
                  f"{d_corr:>+6.1f}p {d_conf:>+6.1f}p")
        else:
            s = systems[0]
            print(f"  {av_short:<30} {r['n']:>3} | "
                  f"{r[f'{s}_correct_pct']:>7.1f}% {r[f'{s}_confab_pct']:>7.1f}%")


# ---------------------------------------------------------------------------
# Table C: Staleness -> Confabulation validation
# ---------------------------------------------------------------------------


def compute_table_c(
    questions: list[dict],
    judgments: dict,
    fr_by_qid: dict,
    mem0_by_qid: dict,
    systems: list[str],
) -> dict:
    """For each system, split by retrieval staleness and compute E2E confab rate."""
    table: dict = {}
    retrieval_maps = {"fr": fr_by_qid, "mem0": mem0_by_qid}

    for sys_name in systems:
        by_qid = retrieval_maps.get(sys_name, {})
        key = f"{sys_name}_judgment"
        stale_confab = stale_total = stale_non_abstain = 0
        clean_confab = clean_total = clean_non_abstain = 0
        stale_correct = clean_correct = 0

        for q in questions:
            qid = q["id"]
            ret = by_qid.get(qid)
            j = judgments.get(qid, {}).get(key)
            if not ret or not j:
                continue

            is_stale = ret.get("staleness_penalty", 0) > 0
            is_correct = j["correctness"] == "CORRECT"
            is_non_abstain = j["correctness"] != "ABSTAIN"
            is_confab = j.get("confabulation", False)

            if is_stale:
                stale_total += 1
                if is_correct:
                    stale_correct += 1
                if is_non_abstain:
                    stale_non_abstain += 1
                    if is_confab:
                        stale_confab += 1
            else:
                clean_total += 1
                if is_correct:
                    clean_correct += 1
                if is_non_abstain:
                    clean_non_abstain += 1
                    if is_confab:
                        clean_confab += 1

        table[sys_name] = {
            "stale_count": stale_total,
            "stale_correct": stale_correct,
            "stale_correct_pct": stale_correct / stale_total * 100 if stale_total else 0,
            "stale_confab": stale_confab,
            "stale_confab_pct": stale_confab / stale_non_abstain * 100 if stale_non_abstain else 0,
            "clean_count": clean_total,
            "clean_correct": clean_correct,
            "clean_correct_pct": clean_correct / clean_total * 100 if clean_total else 0,
            "clean_confab": clean_confab,
            "clean_confab_pct": clean_confab / clean_non_abstain * 100 if clean_non_abstain else 0,
        }
    return table


def print_table_c(table: dict, systems: list[str]):
    print("\n  TABLE C: Staleness -> Confabulation Validation (Proposition 3)")
    print("  " + "-" * 70)
    for sys_name in systems:
        label = "FR" if sys_name == "fr" else "Mem0"
        t = table[sys_name]
        print(f"\n  {label}:")
        print(f"    Stale retrieval (staleness > 0):  n={t['stale_count']:>3}  "
              f"correct={t['stale_correct_pct']:>5.1f}%  confab={t['stale_confab_pct']:>5.1f}%")
        print(f"    Clean retrieval (staleness = 0):  n={t['clean_count']:>3}  "
              f"correct={t['clean_correct_pct']:>5.1f}%  confab={t['clean_confab_pct']:>5.1f}%")
        if t["stale_count"] > 0 and t["clean_count"] > 0:
            delta_confab = t["stale_confab_pct"] - t["clean_confab_pct"]
            print(f"    --> Stale context adds {delta_confab:+.1f}pp confabulation risk")


# ---------------------------------------------------------------------------
# Table D: Per-AV staleness -> confabulation
# ---------------------------------------------------------------------------


def compute_table_d(
    questions: list[dict],
    judgments: dict,
    fr_by_qid: dict,
    mem0_by_qid: dict,
    systems: list[str],
) -> dict:
    av_groups: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        av_groups[q["attack_vector"]].append(q)

    retrieval_maps = {"fr": fr_by_qid, "mem0": mem0_by_qid}
    table: dict = {}

    for av in sorted(av_groups.keys()):
        av_qs = av_groups[av]
        row: dict = {"n": len(av_qs)}

        for sys_name in systems:
            by_qid = retrieval_maps.get(sys_name, {})
            key = f"{sys_name}_judgment"
            stale_count = non_abstain = confab_count = 0
            total = 0

            for q in av_qs:
                qid = q["id"]
                ret = by_qid.get(qid)
                j = judgments.get(qid, {}).get(key)
                if not ret or not j:
                    continue
                total += 1
                if ret.get("staleness_penalty", 0) > 0:
                    stale_count += 1
                if j["correctness"] != "ABSTAIN":
                    non_abstain += 1
                    if j.get("confabulation"):
                        confab_count += 1

            row[f"{sys_name}_stale_pct"] = stale_count / total * 100 if total else 0
            row[f"{sys_name}_confab_pct"] = confab_count / non_abstain * 100 if non_abstain else 0

        table[av] = row
    return table


def print_table_d(table: dict, systems: list[str]):
    print("\n  TABLE D: Per-AV Staleness -> Confabulation (retrieval quality -> response quality)")
    print("  " + "-" * 90)
    if len(systems) == 2:
        print(f"  {'AV':<30} {'n':>3} | {'FR stale%':>9} {'FR confab%':>10} | "
              f"{'M0 stale%':>9} {'M0 confab%':>10}")
    else:
        s = systems[0]
        label = "FR" if s == "fr" else "M0"
        print(f"  {'AV':<30} {'n':>3} | {label+' stale%':>9} {label+' confab%':>10}")
    print("  " + "-" * 90)

    for av in sorted(table.keys()):
        r = table[av]
        av_short = av[:30]
        if len(systems) == 2:
            print(f"  {av_short:<30} {r['n']:>3} | "
                  f"{r['fr_stale_pct']:>8.1f}% {r['fr_confab_pct']:>9.1f}% | "
                  f"{r['mem0_stale_pct']:>8.1f}% {r['mem0_confab_pct']:>9.1f}%")
        else:
            s = systems[0]
            print(f"  {av_short:<30} {r['n']:>3} | "
                  f"{r[f'{s}_stale_pct']:>8.1f}% {r[f'{s}_confab_pct']:>9.1f}%")


# ---------------------------------------------------------------------------
# Table E: Case studies
# ---------------------------------------------------------------------------


def compute_table_e(
    questions: list[dict],
    judgments: dict,
    responses: dict,
    systems: list[str],
) -> list[dict]:
    """Find up to 15 cases where FR=CORRECT, Mem0=WRONG+confabulation."""
    if "fr" not in systems or "mem0" not in systems:
        return []

    q_lookup = {q["id"]: q for q in questions}
    priority = []
    other = []

    for q in questions:
        qid = q["id"]
        j = judgments.get(qid, {})
        fr_j = j.get("fr_judgment", {})
        m0_j = j.get("mem0_judgment", {})

        if (fr_j.get("correctness") == "CORRECT"
                and m0_j.get("correctness") == "WRONG"
                and m0_j.get("confabulation")):
            resp = responses.get(qid, {})
            case = {
                "question_id": qid,
                "persona": q["persona"],
                "attack_vector": q["attack_vector"],
                "question": q["question"],
                "correct_answer": q["correct_answer"],
                "wrong_indicators": q.get("wrong_answer_indicators", []),
                "fr_response": resp.get("fr_response", ""),
                "mem0_response": resp.get("mem0_response", ""),
                "fr_reasoning": fr_j.get("reasoning", ""),
                "mem0_reasoning": m0_j.get("reasoning", ""),
            }
            av_pre = av_prefix(q["attack_vector"])
            if av_pre in TABLE_E_PRIORITY_AVS:
                priority.append(case)
            else:
                other.append(case)

    # Sort each bucket by question_id for deterministic selection
    priority.sort(key=lambda c: c["question_id"])
    other.sort(key=lambda c: c["question_id"])

    selected = priority[:12]
    remaining = 15 - len(selected)
    if remaining > 0:
        selected.extend(other[:remaining])

    return selected


def print_table_e(cases: list[dict]):
    print(f"\n  TABLE E: Case Studies \u2014 FR Correct vs Mem0 Wrong+Confabulation ({len(cases)} cases)")
    print("  " + "-" * 100)
    if not cases:
        print("  (requires --system both to compute)")
        return

    for i, c in enumerate(cases, 1):
        print(f"\n  [{i}] {c['question_id']} | {c['attack_vector']}")
        print(f"      Q: {trunc(c['question'], 70)}")
        print(f"      Correct: {trunc(c['correct_answer'], 70)}")
        print(f"      FR resp: {trunc(c['fr_response'], 120)}")
        print(f"      M0 resp: {trunc(c['mem0_response'], 120)}")
        wrong = "; ".join(c.get("wrong_indicators", []))
        print(f"      M0 got wrong: {trunc(wrong, 50)}")


# ---------------------------------------------------------------------------
# Table F: Abstain analysis
# ---------------------------------------------------------------------------


def compute_table_f(questions: list[dict], judgments: dict, systems: list[str]) -> dict:
    table: dict = {}
    for sys_name in systems:
        key = f"{sys_name}_judgment"
        abstain_count = 0
        abstain_avs: dict[str, int] = defaultdict(int)
        total = 0

        for q in questions:
            j = judgments.get(q["id"], {}).get(key)
            if not j:
                continue
            total += 1
            if j["correctness"] == "ABSTAIN":
                abstain_count += 1
                abstain_avs[q["attack_vector"]] += 1

        table[sys_name] = {
            "total": total,
            "abstain_count": abstain_count,
            "abstain_pct": abstain_count / total * 100 if total else 0,
            "by_av": dict(sorted(abstain_avs.items())),
        }
    return table


def print_table_f(table: dict, systems: list[str]):
    print("\n  TABLE F: Abstain Analysis")
    print("  " + "-" * 70)
    for sys_name in systems:
        label = "FR" if sys_name == "fr" else "Mem0"
        t = table[sys_name]
        print(f"\n  {label}: {t['abstain_count']}/{t['total']} abstains ({t['abstain_pct']:.1f}%)")
        if t["by_av"]:
            for av, cnt in sorted(t["by_av"].items()):
                print(f"    {av}: {cnt}")

    if len(systems) == 2:
        fr_abs = table["fr"]["abstain_count"]
        m0_abs = table["mem0"]["abstain_count"]
        delta = fr_abs - m0_abs
        print(f"\n  FR abstains {abs(delta)} {'more' if delta > 0 else 'fewer'} than Mem0.")
        if delta > 0:
            print("  --> This is a GOOD outcome: \"I don't know\" is better than confidently wrong.")


# ---------------------------------------------------------------------------
# Summary orchestration
# ---------------------------------------------------------------------------


def compute_summary(
    questions: list[dict],
    judgments: dict,
    responses: dict,
    fr_by_qid: dict,
    mem0_by_qid: dict,
    systems: list[str],
) -> dict:
    return {
        "table_a": compute_table_a(questions, judgments, systems),
        "table_b": compute_table_b(questions, judgments, systems),
        "table_c": compute_table_c(questions, judgments, fr_by_qid, mem0_by_qid, systems),
        "table_d": compute_table_d(questions, judgments, fr_by_qid, mem0_by_qid, systems),
        "table_e": compute_table_e(questions, judgments, responses, systems),
        "table_f": compute_table_f(questions, judgments, systems),
    }


def print_summary(summary: dict, systems: list[str]):
    print_table_a(summary["table_a"], systems)
    print_table_b(summary["table_b"], systems)
    print_table_c(summary["table_c"], systems)
    print_table_d(summary["table_d"], systems)
    print_table_e(summary["table_e"])
    print_table_f(summary["table_f"], systems)


# ---------------------------------------------------------------------------
# Dry-run printer
# ---------------------------------------------------------------------------


def print_dry_run(responses: dict, judgments: dict, questions: list[dict], systems: list[str]):
    """Print responses and judgments for manual inspection during --dry-run."""
    print("\n" + "=" * 70)
    print("  DRY-RUN INSPECTION")
    print("=" * 70)
    for q in questions:
        qid = q["id"]
        resp = responses.get(qid, {})
        judg = judgments.get(qid, {})
        print(f"\n  --- {qid} [{q['attack_vector']}] ---")
        print(f"  Q: {q['question']}")
        print(f"  Correct: {q['correct_answer']}")
        wrong = q.get("wrong_answer_indicators", [])
        if wrong:
            print(f"  Wrong indicators: {'; '.join(wrong)}")

        for sys_name in systems:
            label = "FR" if sys_name == "fr" else "Mem0"
            r_text = resp.get(f"{sys_name}_response", "(none)")
            facts = resp.get(f"{sys_name}_facts", [])
            j = judg.get(f"{sys_name}_judgment", {})
            print(f"\n  [{label}] Facts ({len(facts)}):")
            for i, fact in enumerate(facts, 1):
                print(f"    {i}. {trunc(fact, 80)}")
            print(f"  [{label}] Response: {r_text}")
            print(f"  [{label}] Judgment: {j.get('correctness', '?')} "
                  f"confab={j.get('confabulation', '?')} "
                  f"| {j.get('reasoning', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="LifeMemBench end-to-end response quality evaluation"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="5 questions (first persona), print to stdout")
    parser.add_argument("--system", type=str, default="both",
                        choices=["fr", "mem0", "both"],
                        help="Which retrieval system(s) to evaluate (default: both)")
    parser.add_argument("--parallel", type=int, default=25, metavar="N",
                        help="Concurrency limit for API calls (default: 25)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore checkpoints, regenerate everything")
    parser.add_argument("--judge-only", action="store_true",
                        help="Skip generation, judge existing responses")
    parser.add_argument("--model", type=str, default="gpt-5.4",
                        help="Generation model (default: gpt-5.4)")
    parser.add_argument("--response-tag", type=str, default="gpt54_nr",
                        help="Tag for this response set (default: gpt54_nr)")

    args = parser.parse_args()

    systems = ["fr", "mem0"] if args.system == "both" else [args.system]
    tag = args.response_tag

    # Load data
    print("Loading data...")
    questions = load_questions()
    fr_by_qid, mem0_by_qid = load_retrieval_results()
    print(f"  {len(questions)} questions, "
          f"{len(fr_by_qid)} FR results, {len(mem0_by_qid)} Mem0 results")

    # Dry-run: first persona, 5 questions
    if args.dry_run:
        first_persona = questions[0]["persona"]
        questions = [q for q in questions if q["persona"] == first_persona][:5]
        print(f"\n  DRY RUN: {len(questions)} questions for persona '{first_persona}'")

    # Load checkpoints
    existing_responses = {} if args.force else load_checkpoint(E2E_RESPONSES_PATH)
    existing_judgments = {} if args.force else load_checkpoint(E2E_JUDGMENTS_PATH)

    # Initialize clients
    from openai import AsyncOpenAI
    from anthropic import AsyncAnthropic

    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Phase 1: Generate responses
    if not args.judge_only:
        n_calls = len(questions) * len(systems)
        print(f"\n{'=' * 60}")
        print(f"  PHASE 1: RESPONSE GENERATION")
        print(f"  {n_calls} calls | model={args.model} | tag={tag} | parallel={args.parallel}")
        print(f"{'=' * 60}")

        existing_responses = await run_generation(
            questions, fr_by_qid, mem0_by_qid,
            openai_client, args.model, args.parallel,
            systems, tag, existing_responses, args.force, args.dry_run,
        )
    else:
        if tag not in existing_responses:
            print(f"ERROR: No existing responses for tag '{tag}'. Run generation first.")
            sys.exit(1)
        print(f"  Loaded {len(existing_responses.get(tag, {}))} existing responses for tag '{tag}'")

    # Phase 2: Judge responses
    n_judge = len(questions) * len(systems)
    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: JUDGING")
    print(f"  {n_judge} calls | judge=claude-sonnet-4-6 | parallel={args.parallel}")
    print(f"{'=' * 60}")

    existing_judgments = await run_judging(
        questions, existing_responses, anthropic_client,
        args.parallel, tag, systems, existing_judgments,
        args.force, args.dry_run,
    )

    # Dry-run: print everything for inspection
    if args.dry_run:
        print_dry_run(
            existing_responses.get(tag, {}),
            existing_judgments.get(tag, {}),
            questions,
            systems,
        )

    # Phase 3: Analysis
    print(f"\n{'=' * 60}")
    print(f"  PHASE 3: ANALYSIS")
    print(f"{'=' * 60}")

    tag_judgments = existing_judgments.get(tag, {})
    tag_responses = existing_responses.get(tag, {})

    summary = compute_summary(
        questions, tag_judgments, tag_responses,
        fr_by_qid, mem0_by_qid, systems,
    )

    if not args.dry_run:
        save_atomic(E2E_SUMMARY_PATH, {tag: summary})
        print(f"  Summary saved to {E2E_SUMMARY_PATH}")

    print_summary(summary, systems)

    print(f"\n{'=' * 60}")
    print(f"  E2E evaluation complete.")
    if not args.dry_run:
        print(f"  Responses: {E2E_RESPONSES_PATH}")
        print(f"  Judgments: {E2E_JUDGMENTS_PATH}")
        print(f"  Summary:   {E2E_SUMMARY_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
