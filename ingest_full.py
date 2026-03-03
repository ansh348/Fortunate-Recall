"""
ingest_full.py — Full LongMemEval ingestion: all 234 questions.

Placement: C:\\Users\\anshu\\PycharmProjects\\hugeleapforward\\ingest_full.py

Key improvements over ingest_poc.py:
    - Ingests ALL 234 questions (no category filter)
    - Deduplicates sessions across questions (saves ~60% cost)
    - Checkpoint/resume: survives crashes, pick up where you left off
    - Live progress: ETA, cost estimate, sessions/hour
    - Separate phases: ingest -> enrich (can run independently)

Prerequisites:
    - Neo4j running: docker start neo4j-graphiti
    - .env with OPENAI_API_KEY, XAI_API_KEY, NEO4J_* vars
    - Graphiti installed in editable mode
    - decay_engine.py and graphiti_bridge.py at project root

Cost estimate: ~10,000 unique sessions × ~$0.014 = ~$140 for ingestion
              + ~5000 entities × ~$0.001 = ~$5 for classification
              Total: ~$140-150

Usage:
    python ingest_full.py --phase ingest       # Ingest all sessions (resumable)
    python ingest_full.py --phase enrich       # Classify + enrich entities
    python ingest_full.py --phase both         # Do both sequentially
    python ingest_full.py --phase status       # Show progress without running
    python ingest_full.py --phase reset        # Clear checkpoints (start over)
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time as time_module
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'LongMemEval' / 'data'
ARTIFACTS_DIR = DATA_DIR / 'full_artifacts'

# Cap Graphiti's internal concurrency to stay under xAI's 480 RPM limit.
# helpers.py reads this at import time, so it must be set before importing graphiti_core.
os.environ.setdefault('SEMAPHORE_LIMIT', '8')

# Checkpoint files
CHECKPOINT_FILE = ARTIFACTS_DIR / 'ingest_checkpoint.json'
ENRICH_CHECKPOINT_FILE = ARTIFACTS_DIR / 'enrich_checkpoint.json'
SESSION_MAP_FILE = ARTIFACTS_DIR / 'session_map.json'
QUESTIONS_FILE = ARTIFACTS_DIR / 'full_questions.json'

GROUP_ID = "full_234"
BULK_CHUNK_SIZE = 30   # episodes per add_episode_bulk() call (tunable via --chunk-size)
ENRICH_CONCURRENCY = 6  # concurrent classify calls (stay under 480 RPM)

CUSTOM_EXTRACTION_INSTRUCTIONS = """
CRITICAL EXTRACTION PRIORITY:
This is a PERSONAL MEMORY system. Your #1 job is extracting facts about the USER — their
experiences, preferences, relationships, plans, possessions, health, schedule, and opinions.

PRIORITIZE extracting:
- Facts where the user is the subject (e.g., "user bought a Trek Emonda", "user runs 5K in 25:50")
- User's relationships with people (e.g., "user's mom uses the same grocery app")
- User's preferences, habits, and routines (e.g., "user attends yoga 3x/week")
- Specific numbers tied to the user (e.g., "$400,000 pre-approval", "4 bikes", "25 titles on watchlist")
- User's schedule, plans, and obligations (e.g., "cocktail class on Fridays")

DEPRIORITIZE (extract ONLY if directly relevant to user's life):
- General knowledge about places, landmarks, historical facts
- Restaurant descriptions, tourist information, product specs
- Assistant's explanations or recommendations that aren't about the user
- Facts between two non-user entities (e.g., "Pantheon is in Rome")

NUMERIC PRESERVATION (CRITICAL):
When the user mentions ANY number, quantity, amount, count, time, or measurement, you MUST
preserve it exactly in the fact text. Examples:
- "25 minutes and 50 seconds" -> keep "25:50" or "25 minutes 50 seconds"
- "$400,000" -> keep "$400,000"
- "4 bikes" -> keep "4"
- "three times a week" -> keep "three times a week" or "3x/week"
"""


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
# Graphiti client setup
# ---------------------------------------------------------------------------

from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.utils.bulk_utils import RawEpisode

sys.path.insert(0, str(PROJECT_ROOT))
from decay_engine import DecayEngine, TemporalContext
from graphiti_bridge import (
    enrich_entity_node, entity_to_fact_node, is_enriched,
    build_temporal_context, rerank_by_activation, inspect_node,
)


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
# Session deduplication
# ---------------------------------------------------------------------------

def hash_session(session: list[dict]) -> str:
    """Create a deterministic hash for a session (list of turns).

    This lets us deduplicate identical sessions that appear across
    multiple questions' haystacks.
    """
    content = json.dumps(session, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def build_session_map(questions: list[dict]) -> dict:
    """Deduplicate sessions across all questions.

    Returns:
        {
            session_hash: {
                "turns": [...],
                "question_ids": [list of questions that use this session],
                "index": sequential index for naming
            }
        }
    """
    session_map = {}
    total_raw = 0

    for q in questions:
        sessions = q.get('haystack_sessions', [])
        total_raw += len(sessions)

        for session in sessions:
            h = hash_session(session)
            if h not in session_map:
                session_map[h] = {
                    'turns': session,
                    'question_ids': [],
                    'index': len(session_map),
                }
            session_map[h]['question_ids'].append(q['question_id'])

    print(f"Session deduplication: {total_raw} raw -> {len(session_map)} unique "
          f"({100 * (1 - len(session_map) / max(total_raw, 1)):.0f}% saved)")

    return session_map


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    """Load ingestion checkpoint. Returns set of completed session hashes."""
    if CHECKPOINT_FILE.exists():
        return json.load(open(CHECKPOINT_FILE, encoding='utf-8'))
    return {
        'completed_hashes': [],
        'total_sessions': 0,
        'total_time_s': 0,
        'errors': [],
        'started_at': None,
    }


def save_checkpoint(checkpoint: dict):
    """Save ingestion checkpoint atomically."""
    tmp = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)
    tmp.replace(CHECKPOINT_FILE)


def load_enrich_checkpoint() -> dict:
    """Load enrichment checkpoint."""
    if ENRICH_CHECKPOINT_FILE.exists():
        return json.load(open(ENRICH_CHECKPOINT_FILE, encoding='utf-8'))
    return {
        'completed_uuids': [],
        'results': [],
    }


def save_enrich_checkpoint(checkpoint: dict):
    """Save enrichment checkpoint atomically."""
    tmp = ENRICH_CHECKPOINT_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)
    tmp.replace(ENRICH_CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def print_progress(done: int, total: int, elapsed_s: float, errors: int = 0):
    """Print progress bar with ETA and cost estimate."""
    pct = 100 * done / max(total, 1)
    rate = done / max(elapsed_s, 1)  # sessions per second
    remaining = (total - done) / max(rate, 0.001)

    # Cost estimate: ~$0.014 per session (from PoC actual spend)
    est_cost = done * 0.014
    est_total_cost = total * 0.014

    bar_len = 30
    filled = int(bar_len * done / max(total, 1))
    bar = '#' * filled + '-' * (bar_len - filled)

    print(f"\r  [{bar}] {done}/{total} ({pct:.1f}%) "
          f"| {format_time(elapsed_s)} elapsed "
          f"| ETA {format_time(remaining)} "
          f"| ~${est_cost:.1f}/${est_total_cost:.1f} "
          f"| {rate * 3600:.0f}/hr"
          f"{f' | {errors} errors' if errors else ''}", end='', flush=True)


# ---------------------------------------------------------------------------
# Load all questions
# ---------------------------------------------------------------------------

MUST_WIN_TYPES = {
    'single-session-user',
    'single-session-assistant',
    'single-session-preference',
    'knowledge-update',
}


def load_all_questions() -> list[dict]:
    """Load the 234 must-win questions with their haystack sessions."""
    oracle_path = DATA_DIR / 'longmemeval_oracle.json'
    s_cleaned_path = DATA_DIR / 'longmemeval_s_cleaned.json'

    if not oracle_path.exists():
        print(f"ERROR: {oracle_path} not found")
        sys.exit(1)
    if not s_cleaned_path.exists():
        print(f"ERROR: {s_cleaned_path} not found")
        sys.exit(1)

    oracle = json.load(open(oracle_path, encoding='utf-8'))
    s_data = json.load(open(s_cleaned_path, encoding='utf-8'))
    s_lookup = {q['question_id']: q for q in s_data}

    # Filter to must-win types only
    filtered = [q for q in oracle if q['question_type'] in MUST_WIN_TYPES]

    # Attach haystack sessions from s_cleaned (richer data)
    for q in filtered:
        s_entry = s_lookup.get(q['question_id'])
        if s_entry and 'haystack_sessions' in s_entry:
            q['haystack_sessions'] = s_entry['haystack_sessions']

    print(f"Loaded {len(filtered)}/{len(oracle)} questions (must-win types only)")

    # Stats
    by_type = Counter(q['question_type'] for q in filtered)
    for t, c in by_type.most_common():
        print(f"  {t}: {c}")

    has_sessions = sum(1 for q in filtered if q.get('haystack_sessions'))
    print(f"  Questions with haystack sessions: {has_sessions}/{len(filtered)}")

    return filtered


# ---------------------------------------------------------------------------
# Classifier (same as ingest_poc.py)
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """You are a memory classification system. Given an entity from a knowledge graph, classify it into behavioral categories.

## Categories

1. OBLIGATIONS — Tasks, deadlines, appointments, promises.
2. RELATIONAL_BONDS — Relationships: family, partners, friends, colleagues.
3. HEALTH_WELLBEING — Physical/mental health, medications, diagnoses, fitness.
4. IDENTITY_SELF_CONCEPT — Core stable traits: ethnicity, name, occupation, heritage, values.
5. HOBBIES_RECREATION — Active leisure with skill/equipment investment.
6. PREFERENCES_HABITS — Current tastes, media, food choices, subscriptions, routines.
7. INTELLECTUAL_INTERESTS — Curiosities, learning goals without a deliverable.
8. LOGISTICAL_CONTEXT — Transient scheduling, one-time locations, errands.
9. PROJECTS_ENDEAVORS — Ongoing works with milestones: startups, research, creative projects.
10. FINANCIAL_MATERIAL — Budget, income, expenses, purchases, devices, assets.
11. OTHER — Genuinely does not fit 1-10.

## Rules
- Classify by what the entity IS, not conversational context.
- Soft membership weights summing to 1.0. Primary gets highest weight (>= 0.3).
- Detect emotional loading if present.

## Output (strict JSON)
{"primary_category": "CATEGORY", "weights": {"OBLIGATIONS": 0.0, "RELATIONAL_BONDS": 0.0, "HEALTH_WELLBEING": 0.0, "IDENTITY_SELF_CONCEPT": 0.0, "HOBBIES_RECREATION": 0.0, "PREFERENCES_HABITS": 0.0, "INTELLECTUAL_INTERESTS": 0.0, "LOGISTICAL_CONTEXT": 0.0, "PROJECTS_ENDEAVORS": 0.0, "FINANCIAL_MATERIAL": 0.0, "OTHER": 0.0}, "emotional_loading": {"detected": false, "type": null, "intensity": null}, "confidence": "high"}"""


async def classify_entity(entity_name: str, entity_summary: str,
                          xai_client: AsyncOpenAI) -> dict:
    """Classify a Graphiti entity node using v7 behavioral ontology."""
    prompt = f"""Classify this entity/fact into behavioral categories.

**Entity name:** {entity_name}
**Entity summary:** {entity_summary}

What behavioral category does this entity primarily belong to?
Consider what kind of life-domain information this entity represents.

Respond with JSON only."""

    try:
        resp = await xai_client.chat.completions.create(
            model="grok-4-1-fast-non-reasoning",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"\n  Classification error for {entity_name}: {e}")
        return {
            'primary_category': 'OTHER',
            'weights': {'OTHER': 1.0},
            'emotional_loading': {'detected': False},
            'confidence': 'low',
        }


# ---------------------------------------------------------------------------
# Phase 1: Ingest (resumable)
# ---------------------------------------------------------------------------

async def run_ingestion():
    """Ingest all unique sessions through Graphiti with checkpoint/resume."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions = load_all_questions()

    # Save question metadata
    with open(QUESTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump([{
            'question_id': q['question_id'],
            'question_type': q['question_type'],
            'question': q['question'],
            'answer': q['answer'],
            'num_sessions': len(q.get('haystack_sessions', [])),
        } for q in questions], f, indent=2)

    # Build deduplicated session map
    session_map = build_session_map(questions)

    # Save session map for reference
    session_meta = {h: {'index': v['index'], 'question_ids': v['question_ids'],
                        'num_turns': len(v['turns'])}
                    for h, v in session_map.items()}
    with open(SESSION_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(session_meta, f, indent=2)

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint['completed_hashes'])

    total = len(session_map)
    already_done = len(completed)
    remaining = total - already_done

    if already_done > 0:
        print(f"\nResuming: {already_done}/{total} already done, {remaining} remaining")
    else:
        checkpoint['started_at'] = datetime.now(timezone.utc).isoformat()

    if remaining == 0:
        print(f"\nAll {total} sessions already ingested. Nothing to do.")
        print(f"   Run --phase enrich next.")
        return

    # Sort sessions by index for deterministic order
    ordered = sorted(session_map.items(), key=lambda x: x[1]['index'])

    # Initialize Graphiti
    graphiti = get_graphiti_client()
    await graphiti.build_indices_and_constraints()

    chunk_size = getattr(run_ingestion, '_chunk_size', BULK_CHUNK_SIZE)

    print(f"\n{'=' * 70}")
    print(f"  FULL INGESTION: {total} unique sessions -> Neo4j (group: {GROUP_ID})")
    print(f"  Estimated cost: ~${total * 0.014:.0f}  |  Estimated time: ~{format_time(total / max(chunk_size, 1) * 14 * 60)}")
    print(f"{'=' * 70}\n")

    run_start = time_module.time()
    run_done = 0
    run_errors = 0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 10  # circuit breaker: JSON truncation is common, don't trip early

    # Collect remaining sessions
    remaining_sessions = [(h, d) for h, d in ordered if h not in completed]

    # Process in chunks via add_episode_bulk()
    for chunk_start in range(0, len(remaining_sessions), chunk_size):
        chunk = remaining_sessions[chunk_start:chunk_start + chunk_size]
        chunk_num = chunk_start // chunk_size + 1
        total_chunks = (len(remaining_sessions) + chunk_size - 1) // chunk_size

        print(f"\n  Chunk {chunk_num}/{total_chunks} ({len(chunk)} episodes)...", flush=True)

        # Build RawEpisode objects for the chunk
        bulk_episodes = [
            RawEpisode(
                name=f"full_s{d['index']:04d}",
                content="\n".join(
                    f"{t['role']}: {t['content']}" for t in d['turns']
                ),
                source_description=f"LongMemEval session {d['index']} (hash: {h[:8]})",
                source=EpisodeType.message,
                reference_time=datetime.now(timezone.utc),
            )
            for h, d in chunk
        ]

        try:
            await graphiti.add_episode_bulk(
                bulk_episodes=bulk_episodes,
                group_id=GROUP_ID,
                custom_extraction_instructions=CUSTOM_EXTRACTION_INSTRUCTIONS,
            )

            # Checkpoint the entire chunk on success
            for h, d in chunk:
                checkpoint['completed_hashes'].append(h)
            checkpoint['total_sessions'] = already_done + run_done + len(chunk)
            run_done += len(chunk)
            consecutive_errors = 0

        except Exception as e:
            print(f"\n  ERROR chunk {chunk_num} ({len(chunk)} episodes): {e}")
            checkpoint['errors'].append({
                'chunk_hashes': [h for h, _ in chunk],
                'chunk_num': chunk_num,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })
            run_errors += len(chunk)
            consecutive_errors += 1

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n\n  {MAX_CONSECUTIVE_ERRORS} consecutive chunk failures — likely out of API credits.")
                print(f"     Saving checkpoint and exiting cleanly.")
                print(f"     Top up your account, then rerun the same command to resume.")
                save_checkpoint(checkpoint)
                await graphiti.close()
                return

        # Save checkpoint after each chunk
        elapsed = time_module.time() - run_start
        checkpoint['total_time_s'] = checkpoint.get('total_time_s', 0) + elapsed
        save_checkpoint(checkpoint)

        # Progress display
        global_pos = already_done + run_done
        print_progress(global_pos, total, elapsed, run_errors)

        # Detailed log every few chunks
        if chunk_num % 5 == 0:
            rate = run_done / max(elapsed, 1)
            print(f"\n  -- Milestone: {run_done} sessions this run "
                  f"| {rate * 3600:.0f}/hr "
                  f"| ~${global_pos * 0.014:.1f} spent so far")

    elapsed = time_module.time() - run_start

    print(f"\n\n{'=' * 70}")
    print(f"  INGESTION COMPLETE")
    print(f"  Sessions: {already_done + run_done}/{total}")
    print(f"  This run: {run_done} sessions in {format_time(elapsed)}")
    print(f"  Errors: {run_errors}")
    print(f"  Est. cost: ~${total * 0.014:.0f}")
    print(f"{'=' * 70}")
    print(f"\n  Next: python ingest_full.py --phase enrich")

    await graphiti.close()


# ---------------------------------------------------------------------------
# Phase 2: Enrich (resumable)
# ---------------------------------------------------------------------------

async def run_enrichment():
    """Classify and enrich all entity nodes with behavioral ontology."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check ingestion completed
    checkpoint = load_checkpoint()
    if not checkpoint.get('completed_hashes'):
        print("ERROR: No ingestion checkpoint found. Run --phase ingest first.")
        sys.exit(1)

    # Initialize
    graphiti = get_graphiti_client()
    xai_client = AsyncOpenAI(
        api_key=os.environ['XAI_API_KEY'],
        base_url="https://api.x.ai/v1",
    )

    # Fetch all entity nodes
    print(f"Fetching entity nodes for group '{GROUP_ID}'...")
    entity_nodes = await EntityNode.get_by_group_ids(graphiti.driver, [GROUP_ID])
    print(f"Found {len(entity_nodes)} entity nodes")

    # Load enrichment checkpoint
    enrich_ckpt = load_enrich_checkpoint()
    completed_uuids = set(enrich_ckpt['completed_uuids'])

    to_process = [n for n in entity_nodes if n.uuid not in completed_uuids and not is_enriched(n)]
    already_done = len(entity_nodes) - len(to_process)

    if already_done > 0:
        print(f"Resuming: {already_done} already enriched, {len(to_process)} remaining")

    if not to_process:
        print(f"All {len(entity_nodes)} entities already enriched. Nothing to do.")
        # Print distribution
        _print_category_distribution(enrich_ckpt['results'])
        return

    print(f"\n{'=' * 70}")
    print(f"  ENRICHMENT: {len(to_process)} entities to classify")
    print(f"  Est. cost: ~${len(to_process) * 0.001:.2f}")
    print(f"{'=' * 70}\n")

    start = time_module.time()

    async def _classify_and_save(node):
        classification = await classify_entity(
            node.name, node.summary or '', xai_client
        )
        enrich_entity_node(node, classification)
        await node.save(graphiti.driver)
        return node.uuid, node.name, classification

    # Process in concurrent batches (stay under 480 RPM xAI limit)
    for batch_start in range(0, len(to_process), ENRICH_CONCURRENCY):
        batch = to_process[batch_start:batch_start + ENRICH_CONCURRENCY]

        results = await asyncio.gather(
            *[_classify_and_save(node) for node in batch],
            return_exceptions=True,
        )

        for j, result in enumerate(results):
            node = batch[j]
            if isinstance(result, Exception):
                print(f"\n  Classification error for {node.name}: {result}")
                enrich_ckpt['completed_uuids'].append(node.uuid)
                enrich_ckpt['results'].append({
                    'uuid': node.uuid,
                    'name': node.name,
                    'primary_category': 'OTHER',
                    'confidence': 'low',
                })
            else:
                uuid, name, classification = result
                enrich_ckpt['completed_uuids'].append(uuid)
                enrich_ckpt['results'].append({
                    'uuid': uuid,
                    'name': name,
                    'primary_category': classification.get('primary_category', 'OTHER'),
                    'confidence': classification.get('confidence', 'unknown'),
                })

        save_enrich_checkpoint(enrich_ckpt)
        done_total = already_done + batch_start + len(batch)
        elapsed = time_module.time() - start
        remaining_count = len(to_process) - (batch_start + len(batch))
        rate = (batch_start + len(batch)) / max(elapsed, 0.01)
        print(f"  Classified {done_total}/{len(entity_nodes)} "
              f"({format_time(elapsed)} elapsed, "
              f"~{remaining_count / max(rate, 0.01):.0f}s remaining)")

    save_enrich_checkpoint(enrich_ckpt)

    elapsed = time_module.time() - start
    print(f"\n{'=' * 70}")
    print(f"  ENRICHMENT COMPLETE")
    print(f"  Entities: {len(entity_nodes)}")
    print(f"  This run: {len(to_process)} in {format_time(elapsed)}")
    print(f"{'=' * 70}")

    _print_category_distribution(enrich_ckpt['results'])

    await graphiti.close()


def _print_category_distribution(results: list[dict]):
    """Print category distribution summary."""
    cats = Counter(r['primary_category'] for r in results)
    conf = Counter(r['confidence'] for r in results)
    print(f"\nCategory distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    print(f"\nConfidence distribution:")
    for c, count in conf.most_common():
        print(f"  {c}: {count}")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status():
    """Show current progress without running anything."""
    print(f"\n{'=' * 70}")
    print(f"  INGESTION STATUS")
    print(f"{'=' * 70}\n")

    # Session map
    if SESSION_MAP_FILE.exists():
        smap = json.load(open(SESSION_MAP_FILE, encoding='utf-8'))
        print(f"  Unique sessions: {len(smap)}")
    else:
        print(f"  Session map: not built yet")

    # Ingestion checkpoint
    if CHECKPOINT_FILE.exists():
        ckpt = json.load(open(CHECKPOINT_FILE, encoding='utf-8'))
        done = len(ckpt.get('completed_hashes', []))
        total = len(smap) if SESSION_MAP_FILE.exists() else '?'
        errors = len(ckpt.get('errors', []))
        t = ckpt.get('total_time_s', 0)
        print(f"  Ingestion: {done}/{total} sessions")
        print(f"  Errors: {errors}")
        print(f"  Total time: {format_time(t)}")
        print(f"  Started: {ckpt.get('started_at', 'unknown')}")
        if done > 0 and isinstance(total, int):
            est_cost = done * 0.014
            print(f"  Est. cost so far: ~${est_cost:.1f}")
    else:
        print(f"  Ingestion: not started")

    # Enrichment checkpoint
    if ENRICH_CHECKPOINT_FILE.exists():
        eckpt = json.load(open(ENRICH_CHECKPOINT_FILE, encoding='utf-8'))
        done = len(eckpt.get('completed_uuids', []))
        print(f"  Enrichment: {done} entities classified")
        if eckpt.get('results'):
            _print_category_distribution(eckpt['results'])
    else:
        print(f"  Enrichment: not started")

    print()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset_checkpoints():
    """Clear all checkpoints (start from scratch)."""
    files = [CHECKPOINT_FILE, ENRICH_CHECKPOINT_FILE, SESSION_MAP_FILE, QUESTIONS_FILE]
    for f in files:
        if f.exists():
            f.unlink()
            print(f"  Deleted {f.name}")
    print(f"\n  Checkpoints cleared. Neo4j data is NOT deleted.")
    print(f"  To fully reset, also clear the Neo4j database.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description='Fortunate Recall — Full 234-question ingestion'
    )
    parser.add_argument(
        '--phase',
        choices=['ingest', 'enrich', 'both', 'status', 'reset'],
        default='both',
        help='Which phase to run'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=BULK_CHUNK_SIZE,
        help=f'Episodes per bulk ingestion call (default: {BULK_CHUNK_SIZE})'
    )
    args = parser.parse_args()

    if args.phase == 'status':
        show_status()
        return

    if args.phase == 'reset':
        reset_checkpoints()
        return

    if args.phase in ('ingest', 'both'):
        run_ingestion._chunk_size = args.chunk_size
        await run_ingestion()

    if args.phase in ('enrich', 'both'):
        await run_enrichment()


if __name__ == '__main__':
    asyncio.run(main())
