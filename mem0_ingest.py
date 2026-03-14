r"""
mem0_ingest.py - Ingest LifeMemBench personas into Mem0 (open-source, default config).

Feeds 40 personas (35 sessions each, 1,400 total) through Mem0's memory
extraction pipeline using 100% default settings. Mem0 handles extraction,
deduplication, and storage internally.

NOTE: Mem0's default Qdrant vector store is in-memory (on_disk=False) and does
NOT persist between processes. This script is useful for canary tests and
standalone ingestion experiments. For the full benchmark (ingest + evaluate),
use evaluate_mem0.py --ingest which runs both in one process.

Prerequisites:
    - pip install mem0ai
    - .env with OPENAI_API_KEY (Mem0 default LLM + embedder use OpenAI)

Usage:
    python mem0_ingest.py --canary priya 3      # Canary: ingest 1 session, search, verify
    python mem0_ingest.py --persona 1_priya     # Single persona (in-memory only)
    python mem0_ingest.py --all                 # All 40 personas (in-memory only)
    python mem0_ingest.py --status              # Show checkpoint progress
    python mem0_ingest.py --reset               # Clear checkpoint + Mem0 data
"""

import argparse
import json
import os
import shutil
import sys
import time as time_module
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent


def load_env():
    """Load .env from project root."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env()

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set. Mem0 defaults require OpenAI.")
    sys.exit(1)

from mem0 import Memory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIFEMEMEVAL_DIR = PROJECT_ROOT / "LifeMemEval"
ARTIFACTS_DIR = LIFEMEMEVAL_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = ARTIFACTS_DIR / "mem0_ingest_checkpoint.json"

# Persona directory name -> short name
PERSONAS = {
    "1_priya": "priya", "2_marcus": "marcus", "3_elena": "elena",
    "4_david": "david", "5_amara": "amara", "6_jake": "jake",
    "7_fatima": "fatima", "8_tom": "tom", "9_kenji": "kenji",
    "10_rosa": "rosa", "11_callum": "callum", "12_diane": "diane",
    "13_raj": "raj", "14_nadia": "nadia", "15_samuel": "samuel",
    "16_lily": "lily", "17_omar": "omar", "18_bruna": "bruna",
    "19_patrick": "patrick", "20_aisha": "aisha",
    "21_thanh": "thanh", "22_alex": "alex", "23_mirri": "mirri",
    "24_jerome": "jerome", "25_ingrid": "ingrid", "26_dmitri": "dmitri",
    "27_yoli": "yoli", "28_dariush": "dariush", "29_aroha": "aroha",
    "30_mehmet": "mehmet", "31_saga": "saga", "32_kofi": "kofi",
    "33_valentina": "valentina", "34_billy": "billy", "35_pan": "pan",
    "36_marley": "marley", "37_leila": "leila", "38_chenoa": "chenoa",
    "39_joonho": "joonho", "40_zara": "zara",
}


# ---------------------------------------------------------------------------
# Session loading (from ingest_lifemembench.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            return json.load(open(CHECKPOINT_PATH, encoding="utf-8"))
        except Exception:
            pass
    return {"started_at": None, "personas": {}}


def save_checkpoint(checkpoint: dict):
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    tmp.replace(CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_persona(persona_dir: str, short_name: str, m: Memory) -> dict:
    """Ingest all sessions for one persona. Returns stats dict."""
    sessions = load_persona_sessions(persona_dir)
    t0 = time_module.time()
    consecutive_failures = 0
    sessions_ingested = 0

    for session in sessions:
        sid = session["session_id"]
        turns = session.get("turns", [])
        if not turns:
            print(f"    Session {sid}: no turns, skipping")
            continue

        for attempt in range(3):
            try:
                m.add(
                    messages=turns,
                    user_id=short_name,
                    metadata={"session_id": sid, "date": session.get("date", "")},
                )
                sessions_ingested += 1
                consecutive_failures = 0
                print(f"    Session {sid:2d} ({session.get('type', '?'):8s}) — "
                      f"{len(turns)} turns — OK")
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    Session {sid}: attempt {attempt+1} failed ({e}), "
                          f"retrying in {wait}s...")
                    time_module.sleep(wait)
                else:
                    print(f"    Session {sid}: FAILED after 3 attempts: {e}")
                    consecutive_failures += 1
        else:
            if consecutive_failures >= 5:
                print(f"  CIRCUIT BREAKER: 5 consecutive failures for {short_name}")
                return {
                    "status": "error",
                    "sessions": sessions_ingested,
                    "memories": 0,
                    "time_s": round(time_module.time() - t0, 1),
                    "error": "circuit breaker: 5 consecutive failures",
                }

    # Count memories
    try:
        all_memories = m.get_all(user_id=short_name)
        if isinstance(all_memories, dict):
            mem_count = len(all_memories.get("results", []))
        elif isinstance(all_memories, list):
            mem_count = len(all_memories)
        else:
            mem_count = len(all_memories) if all_memories else 0
    except Exception as e:
        print(f"  WARNING: Could not count memories: {e}")
        mem_count = -1

    elapsed = round(time_module.time() - t0, 1)
    return {
        "status": "done",
        "sessions": sessions_ingested,
        "memories": mem_count,
        "time_s": elapsed,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Canary test
# ---------------------------------------------------------------------------

def run_canary(short_name: str, session_num: int, m: Memory):
    """Ingest one session, search, print results, clean up."""
    # Find persona dir
    persona_dir = None
    for d, s in PERSONAS.items():
        if s == short_name:
            persona_dir = d
            break
    if not persona_dir:
        print(f"ERROR: Unknown persona '{short_name}'")
        sys.exit(1)

    sessions = load_persona_sessions(persona_dir)
    target = None
    for s in sessions:
        if s["session_id"] == session_num:
            target = s
            break
    if not target:
        print(f"ERROR: Session {session_num} not found for {short_name}")
        sys.exit(1)

    canary_uid = f"{short_name}_canary"

    print(f"\n=== MEM0 CANARY: {short_name} session {session_num} ===")
    print(f"  Date:  {target.get('date', '?')}")
    print(f"  Topic: {target.get('topic', '?')}")
    print(f"  Type:  {target.get('type', '?')}")
    print(f"  Turns: {len(target.get('turns', []))}")
    print(f"  Facts: {target.get('facts_disclosed', [])}")

    print(f"\n  Ingesting via Mem0 (default config)...")
    t0 = time_module.time()
    m.add(messages=target["turns"], user_id=canary_uid)
    dt = time_module.time() - t0
    print(f"  Done in {dt:.1f}s")

    # Show all extracted memories
    print(f"\n  Extracted memories (user_id={canary_uid}):")
    try:
        all_mem = m.get_all(user_id=canary_uid)
        entries = all_mem.get("results", all_mem) if isinstance(all_mem, dict) else all_mem
        for i, entry in enumerate(entries):
            text = entry["memory"] if isinstance(entry, dict) else entry.memory
            print(f"    {i+1}. \"{text}\"")
        print(f"  Total: {len(entries)} memories")
    except Exception as e:
        print(f"  ERROR listing memories: {e}")

    # Search test with a question from the benchmark (if available)
    questions_path = LIFEMEMEVAL_DIR / "lifemembench_questions.json"
    if questions_path.exists():
        questions = json.load(open(questions_path, encoding="utf-8"))
        persona_qs = [q for q in questions if q["persona"] == short_name]
        if persona_qs:
            test_q = persona_qs[0]
            print(f"\n  Search test: \"{test_q['question']}\"")
            print(f"  Expected: {test_q['correct_answer']}")
            try:
                results = m.search(query=test_q["question"], user_id=canary_uid, limit=5)
                entries = results.get("results", results) if isinstance(results, dict) else results
                for i, r in enumerate(entries):
                    text = r["memory"] if isinstance(r, dict) else r.memory
                    score = r.get("score", "?") if isinstance(r, dict) else getattr(r, "score", "?")
                    print(f"    {i+1}. [{score}] \"{text}\"")
            except Exception as e:
                print(f"  ERROR searching: {e}")

    # Clean up canary data
    print(f"\n  Cleaning up canary user ({canary_uid})...")
    try:
        m.delete_all(user_id=canary_uid)
        print("  Done.")
    except Exception as e:
        print(f"  WARNING: cleanup failed: {e}")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status():
    checkpoint = load_checkpoint()
    personas = checkpoint.get("personas", {})

    if not personas:
        print("No ingestion data yet. Run --all or --persona to start.")
        return

    print(f"\n{'Persona':<12} {'Status':<8} {'Sessions':>8} {'Memories':>9} {'Time':>7}")
    print("-" * 50)

    total_sessions = 0
    total_memories = 0
    done_count = 0

    for name in sorted(personas.keys()):
        p = personas[name]
        status = p.get("status", "?")
        sessions = p.get("sessions", 0)
        memories = p.get("memories", 0)
        time_s = p.get("time_s", 0)

        print(f"  {name:<10} {status:<8} {sessions:>8} {memories:>9} {time_s:>6.0f}s")

        total_sessions += sessions
        total_memories += max(0, memories)
        if status == "done":
            done_count += 1

    print("-" * 50)
    print(f"  {'TOTAL':<10} {done_count}/{len(PERSONAS):<6} {total_sessions:>8} "
          f"{total_memories:>9}")


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def do_reset():
    """Clear checkpoint and Mem0 default data directories."""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"  Removed {CHECKPOINT_PATH}")

    # Qdrant default location
    for qdrant_dir in [Path("/tmp/qdrant"), Path("C:/tmp/qdrant")]:
        if qdrant_dir.exists():
            shutil.rmtree(qdrant_dir, ignore_errors=True)
            print(f"  Removed {qdrant_dir}")

    # Mem0 history DB
    history_db = Path.home() / ".mem0" / "history.db"
    if history_db.exists():
        history_db.unlink()
        print(f"  Removed {history_db}")

    print("  Reset complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest LifeMemBench personas into Mem0 (default config)",
    )
    parser.add_argument("--all", action="store_true",
                        help="Ingest all 40 personas")
    parser.add_argument("--persona", type=str, metavar="NAME",
                        help="Single persona dir (e.g., 1_priya)")
    parser.add_argument("--status", action="store_true",
                        help="Show ingestion progress table")
    parser.add_argument("--canary", nargs=2, metavar=("NAME", "SESSION"),
                        help="Canary test: persona short name + session number")
    parser.add_argument("--reset", action="store_true",
                        help="Clear checkpoint and Mem0 data")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.reset:
        do_reset()
        return

    # Initialize Mem0 with 100% default config
    print("Initializing Mem0 (default config)...")
    m = Memory()
    print("  Mem0 ready.")

    if args.canary:
        name, sess = args.canary
        run_canary(name, int(sess), m)
        return

    # Determine which personas to ingest
    if args.persona:
        if args.persona not in PERSONAS:
            print(f"ERROR: Unknown persona '{args.persona}'")
            print(f"Valid: {', '.join(sorted(PERSONAS.keys()))}")
            sys.exit(1)
        work = [(args.persona, PERSONAS[args.persona])]
    elif args.all:
        work = list(PERSONAS.items())
    else:
        parser.print_help()
        return

    checkpoint = load_checkpoint()
    if not checkpoint["started_at"]:
        checkpoint["started_at"] = time_module.strftime("%Y-%m-%dT%H:%M:%S")

    total_t0 = time_module.time()
    completed = 0
    skipped = 0

    for persona_dir, short_name in work:
        existing = checkpoint["personas"].get(short_name, {})
        if existing.get("status") == "done":
            print(f"\n  {short_name}: already done ({existing.get('memories', '?')} memories), skipping")
            skipped += 1
            continue

        # If previously errored, clean up before re-ingesting
        if existing.get("status") == "error":
            print(f"\n  {short_name}: clearing previous error state...")
            try:
                m.delete_all(user_id=short_name)
            except Exception:
                pass

        print(f"\n{'='*50}")
        print(f"  INGESTING: {short_name} ({persona_dir})")
        print(f"{'='*50}")

        result = ingest_persona(persona_dir, short_name, m)
        checkpoint["personas"][short_name] = result
        save_checkpoint(checkpoint)

        status_icon = "OK" if result["status"] == "done" else "FAIL"
        print(f"  {short_name}: {status_icon} — {result['sessions']} sessions, "
              f"{result['memories']} memories, {result['time_s']}s")

        if result["status"] == "done":
            completed += 1

    elapsed = time_module.time() - total_t0
    print(f"\n{'='*50}")
    print(f"  COMPLETE: {completed} ingested, {skipped} skipped, "
          f"{len(work) - completed - skipped} failed")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
