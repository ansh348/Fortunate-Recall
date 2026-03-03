# Fortunate Recall — Comprehensive Codebase Analysis

> **Project**: Fortunate Recall | NeurIPS 2026 Submission
> **Core Thesis**: Behavioral ontology (11 categories) > Cognitive typology (3 types) > Uniform decay
> **Generated**: 2026-02-25

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [File-by-File Analysis](#2-file-by-file-analysis)
   - [2.1 LongMemEval Data Pipeline](#21-longmemeval-data-pipeline)
   - [2.2 Core Engine: decay_engine.py](#22-core-engine-decay_enginepy)
   - [2.3 Bridge: graphiti_bridge.py](#23-bridge-graphiti_bridgepy)
   - [2.4 Ingestion Scripts](#24-ingestion-scripts)
   - [2.5 Evaluation Evolution: v2 → v3 → v4](#25-evaluation-evolution-v2--v3--v4)
   - [2.6 Supporting Tools](#26-supporting-tools)
3. [Pipeline Flow Diagram](#3-pipeline-flow-diagram)
4. [Key Constants & Configuration](#4-key-constants--configuration)
5. [Design Patterns](#5-design-patterns)
6. [Current State](#6-current-state)

---

## 1. Project Overview

**Fortunate Recall** is a research system that models how human-like memory fades differently depending on the *behavioral category* of a stored fact. The core claim is that a **behavioral ontology** — classifying memories into 11 categories like OBLIGATIONS, IDENTITY_SELF_CONCEPT, HOBBIES_RECREATION — with category-conditioned temporal decay provides measurably better memory retrieval than either:

- A **uniform decay rate** (same exponential for everything), or
- A **cognitive typology** (the classic episodic / semantic / procedural split)

The system is built on top of **Graphiti** (Zep's open-source temporal knowledge graph backed by Neo4j) and evaluated against **LongMemEval**, a benchmark of 234 long-memory questions across 7 question types.

### Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│                      Fortunate Recall System                     │
├──────────────────┬───────────────────┬───────────────────────────┤
│  decay_engine.py │ graphiti_bridge.py│  Graphiti + Neo4j          │
│  (pure math,     │ (adapter/bridge,  │  (knowledge graph,         │
│   zero I/O)      │  zero Graphiti    │   hybrid retrieval,        │
│                  │  imports)         │   episode ingestion)       │
├──────────────────┴───────────────────┴───────────────────────────┤
│  classify_facts.py → ingest → enrich → evaluate → judge          │
│  (LLM classification pipeline, checkpoint/resume, ablation sweep)│
└──────────────────────────────────────────────────────────────────┘
```

### The 11 Behavioral Categories

| # | Category | Decay Rate | Example |
|---|----------|-----------|---------|
| 1 | OBLIGATIONS | 0.050/hr (~70%/day) | "Dentist appointment Thursday" |
| 2 | RELATIONAL_BONDS | 0.001/hr (~2.4%/day) | "Mom's birthday is March 15" |
| 3 | HEALTH_WELLBEING | 0.002/hr (~4.7%/day) | "Started physical therapy" |
| 4 | IDENTITY_SELF_CONCEPT | 0.0001/hr (~0.24%/day) | "I'm Irish-Italian" |
| 5 | HOBBIES_RECREATION | 0.003/hr (~7%/day) | "Learning to play guitar" |
| 6 | PREFERENCES_HABITS | 0.010/hr (~21%/day) | "I prefer oat milk" |
| 7 | INTELLECTUAL_INTERESTS | 0.002/hr (~4.7%/day) | "Fascinated by quantum computing" |
| 8 | LOGISTICAL_CONTEXT | 0.080/hr (~86%/day) | "Meeting at 3pm in Room 204" |
| 9 | PROJECTS_ENDEAVORS | 0.008/hr (~17%/day) | "Writing Chapter 3 of my novel" |
| 10 | FINANCIAL_MATERIAL | 0.005/hr (~11%/day) | "Bought a $400K house" |
| 11 | OTHER | 0.010/hr (~21%/day) | Default / unclassifiable |

---

## 2. File-by-File Analysis

### 2.1 LongMemEval Data Pipeline

Four scripts that transform raw LongMemEval data into classified, auditable facts.

---

#### `LongMemEval/data/extract_facts.py`

**Purpose**: Stage 1 — Extracts all user utterances that contain answer-bearing information from the raw oracle dataset.

**Input**: `longmemeval_oracle.json` → **Output**: `extracted_facts.json`

| Aspect | Detail |
|--------|--------|
| Lines | ~50 |
| Dependencies | stdlib only (`json`, `random`, `Counter`) |
| External calls | None |

**Logic** (script-level, no functions):
1. Loads `longmemeval_oracle.json`
2. Groups questions by `question_type`
3. For each question (sorted by type, then ID), finds turns where `role == 'user'` AND `has_answer == True`
4. Emits flat records with: `question_id`, `question_type`, `question`, `answer`, `user_utterance` (full text), `user_utterance_preview` (300 chars), `full_utterance_len`
5. Prints summary statistics (count by type, utterance length stats)
6. Writes to `extracted_facts.json`

**Design Decisions**:
- **No truncation**: Full utterance preserved for downstream classification
- **Flat denormalization**: Each record carries its own question metadata — no re-join needed downstream
- **Deterministic ordering**: Sorted by type then ID; `random.seed(42)` set as pipeline convention
- **Script-level execution**: No `if __name__` guard — run-once pipeline script

---

#### `LongMemEval/data/estimate_ingestion_cost.py`

**Purpose**: Utility — Counts total and unique sessions in the "small" dataset for cost planning.

**Input**: `longmemeval_s_cleaned.json` → **Output**: Console only

| Aspect | Detail |
|--------|--------|
| Lines | ~10 |
| Dependencies | `json` only |
| External calls | None |

**Logic**: Serializes each session to JSON string, uses `set()` for deduplication, prints raw vs unique counts.

**Note**: Operates on `_s_cleaned` (not `_oracle`), making it a sidecar utility outside the main extraction pipeline.

---

#### `LongMemEval/data/classify_facts.py`

**Purpose**: Stage 2 — Classifies each extracted fact into the 11-category behavioral ontology using an LLM (Grok 4.1 Fast via xAI API).

**Input**: `extracted_facts.json` → **Output**: `classified_facts.json`

| Aspect | Detail |
|--------|--------|
| Lines | ~284 |
| Dependencies | `openai` (AsyncOpenAI), stdlib (`json`, `asyncio`, `os`, `sys`, `Counter`) |
| LLM | `grok-4-1-fast-non-reasoning` via `https://api.x.ai/v1` |
| Concurrency | Batches of 10 via `asyncio.gather()` |
| Cost model | $0.20/M input, $0.50/M output tokens |

**Constants**:

| Constant | Value | Purpose |
|----------|-------|---------|
| `CANONICAL_CATEGORIES` | 11-element list | Authoritative category names |
| `SYSTEM_PROMPT` | ~4,000 chars | Full classification instructions |
| `MAX_RETRIES` | 3 | Per-fact API retry limit |
| `RETRY_DELAY` | 2 | Base delay (doubles: 2s, 4s, 8s) |

**Functions**:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `get_api_key()` | `-> str` | Loads `XAI_API_KEY` from env or `.env` files (parent/child dirs) |
| `validate_and_normalize()` | `(raw: dict) -> dict` | Normalizes LLM output: fuzzy category matching, weight renormalization, OTHER suppression, tie-breaking |
| `classify_one()` | `async (fact, idx) -> dict` | Classifies one fact with retry logic; returns augmented record with `classification`, `classification_raw`, `tokens_in`, `tokens_out` |
| `main()` | `async ()` | Orchestrates batched classification, prints cost/distribution summary, saves results |

**SYSTEM_PROMPT Key Rules**:
- "CLASSIFY THE FACT, NOT THE UTTERANCE" — classification is based on the nature of the stored information
- Boundary tests: skill accumulation → HOBBIES, consumption choice → PREFERENCES, core trait → IDENTITY
- PREFERENCES_HABITS guardrails: prevents absorption of INTELLECTUAL, IDENTITY, or HEALTH facts
- FINANCIAL_MATERIAL guardrail: physical items classified by what the question asks (value → FINANCIAL, usage → PREFERENCES)
- Named person disambiguation: a person in the utterance ≠ automatic RELATIONAL_BONDS
- **Deliverable test (mandatory override)**: named artifact being created/edited → PROJECTS_ENDEAVORS
- Soft membership: all 11 weights must sum to 1.0, primary gets ≥ 0.3, no ties, OTHER ≤ 0.5

**`validate_and_normalize` Correction Steps**:
1. Fuzzy-match primary category (exact → substring → default OTHER)
2. Case-insensitive weight key matching; fill missing categories with 0.0
3. OTHER suppression: if OTHER wins weakly (< 0.4) and a runner-up is within 0.1, promote runner-up
4. Ensure primary has strictly highest weight (bump +0.05 if tied)
5. Renormalize all weights to sum to 1.0
6. Validate emotional_loading dict structure

**Gravity Well Warning**: If any single category captures >25% of all facts, prints a warning about possible over-absorption.

---

#### `LongMemEval/data/generate_audit_prompt.py`

**Purpose**: Stage 3 — Generates a human-readable audit prompt for quality assurance, designed to be pasted into a separate LLM instance.

**Input**: `classified_facts.json` → **Output**: `audit_prompt.txt`

| Aspect | Detail |
|--------|--------|
| Lines | ~132 |
| Dependencies | stdlib only (`json`, `random`, `Counter`) |
| Sample size | `SAMPLE_SIZE = 100` |
| Sampling | Stratified: ≥ 2 per category, then random fill |

**Stratified Sampling** (two-pass):
1. **Guaranteed minimums**: At least 2 representatives per category (or all if fewer than 2 available)
2. **Fill remaining budget**: Random sample from all unselected classified facts

**Audit Prompt Structure**:
- Header with condensed category definitions and boundary tests
- Per-fact block: question ID, type, utterance (500 char max), question (150 char max), answer (120 char max), classifier's primary + top 3 weights, emotional loading, blank verdict line
- Summary section requesting: agreement rate, systematic errors, category boundary issues, prompt change recommendations

**Verdict System**: AGREE / PARTIAL / DISAGREE — three-tier for nuance.

---

### 2.2 Core Engine: `decay_engine.py`

**Purpose**: The mathematical heart of Fortunate Recall — computes category-conditioned temporal memory activation scores. Pure math, zero I/O, zero LLM calls, zero database access.

| Aspect | Detail |
|--------|--------|
| Lines | 1,035 |
| Dependencies | stdlib only (`math`, `time`, `dataclasses`, `typing`, `enum`) |
| External calls | None — pure computation module |

#### Core Formula

```
activation = base_decay(f, t) × anticipatory(f, t) + emotional_boost(f, t)
```

Where:
```
base_decay   = exp(-λ_eff × Δt_eff)
λ_eff        = Σ(w_k) / Σ(w_k / λ_k)           [weighted harmonic mean]
Δt_eff       = Σ(w_k × Δt_k) / Σ(w_k)           [weighted average delta]
Δt_k         = s_abs × t_absolute + s_rel × t_relative + s_conv × (messages × 0.5)
```

#### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `BlendingMode` | `HARMONIC` (default), `ARITHMETIC`, `MAX_ONLY` | How per-cluster rates combine |
| `ClockMode` | `MULTI` (default), `ABSOLUTE_ONLY` | Which temporal signals to use |

#### Dataclasses

**`DecayConfig`** — Complete configuration with every field serving as an ablation switch:

| Field Group | Key Fields | Defaults |
|-------------|-----------|----------|
| Per-cluster decay rates | `cluster_decay_rates: dict` | 11 category rates (see §4) |
| Clock sensitivity | `clock_sensitivity: dict` | 11 × 3-tuples `(s_abs, s_rel, s_conv)` |
| Conversational time | `conv_time_scale: float` | 0.5 (1 msg = 0.5 effective hours) |
| Anticipatory activation | `anticipatory_enabled`, `_window_hours`, `_peak_multiplier`, `_post_decay_rate` | True, 168h (7 days), 3.0×, 0.15/hr |
| Reactivation | `reactivation_enabled`, `_threshold`, `_boost`, `_eligible` | True, 0.10, 0.70, {HOBBIES, INTELLECTUAL, RELATIONAL} |
| Emotional boost | `emotional_loading_enabled`, `_boost_initial`, `_boost_decay_rate` | True, 0.30, 0.10/hr |
| Blending/clock | `blending_mode`, `clock_mode` | HARMONIC, MULTI |
| Uniform override | `uniform_decay`, `uniform_decay_rate` | False, 0.010 |
| Cognitive typology | `cognitive_typology`, `cognitive_decay_rates`, `cognitive_clock_sensitivity` | False, {EPISODIC: 0.050, SEMANTIC: 0.005, PROCEDURAL: 0.003} |
| 8+1 ontology | `ontology_8plus1` | False |
| Staleness | `base_activation`, `staleness_floor` | 1.0, 0.01 |

**`TemporalContext`** — The three clocks:

| Field | Type | Description |
|-------|------|-------------|
| `absolute_hours` | float | Wall-clock hours since fact's `last_updated_ts` |
| `relative_hours` | float | Hours since user's last session start |
| `conversational_messages` | int | Messages in current session since fact last touched |
| `current_timestamp` | Optional[float] | Unix timestamp of "now" |

**`FactNode`** — Minimal fact representation:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fact_id` | str | — | Unique identifier |
| `membership_weights` | dict | — | `{category: weight}`, sums to 1.0 |
| `primary_category` | str | — | Dominant category |
| `last_updated_ts` | float | — | Unix timestamp |
| `base_activation` | float | 1.0 | Starting activation |
| `future_anchor_ts` | Optional[float] | None | Deadline for anticipatory activation |
| `emotional_loading` | bool | False | Whether emotion detected |
| `emotional_loading_ts` | Optional[float] | None | When emotion was detected |
| `last_reactivation_ts` | Optional[float] | None | Last retrieval-based reactivation |
| `access_count` | int | 0 | Times retrieved |

#### The `DecayEngine` Class

**Constructor**: `__init__(config: Optional[DecayConfig] = None)` — stores config, runs `_validate_config()` with fail-fast assertions.

**Step 1 — Resolve Weights**:

| Method | Purpose |
|--------|---------|
| `_resolve_weights(fact)` | Returns effective weights: raw (default), mapped to cognitive types (ablation 9), or collapsed 8+1 (ablation 12) |
| `_resolve_primary(fact)` | Returns effective primary category under current ontology mode |

**Step 2 — Per-Cluster Temporal Delta**:

| Method | Purpose |
|--------|---------|
| `_get_decay_rate(category)` | Returns λ_k for the category (or cognitive type) |
| `_get_clock_sensitivity(category)` | Returns `(s_abs, s_rel, s_conv)` tuple |
| `compute_effective_delta(category, ctx)` | `Δt_k = s_abs × abs_hrs + s_rel × rel_hrs + s_conv × (msgs × 0.5)` |

**Step 3 — Base Decay** (`compute_base_decay`):

| Mode | Formula |
|------|---------|
| Uniform (ablation 1) | `exp(-uniform_rate × absolute_hours)` |
| MAX_ONLY (ablation 11) | `exp(-λ_primary × Δt_primary)` |
| **HARMONIC (default)** | `exp(-λ_eff × Δt_eff)` where λ_eff = weighted harmonic mean |
| ARITHMETIC (ablation 10) | `Σ(w_k × exp(-λ_k × Δt_k)) / Σ(w_k)` |

**Why harmonic**: Biases toward slow-decaying clusters. A fact 70% HEALTH + 30% LOGISTICAL decays closer to the HEALTH rate. Identity-adjacent components dominate persistence.

**Step 4 — Anticipatory Activation** (`compute_anticipatory`):

Only for OBLIGATIONS and PROJECTS_ENDEAVORS with a `future_anchor_ts`.

| Condition | Behavior |
|-----------|----------|
| > 168h before deadline | Inactive (normal decay) |
| 0–168h before deadline | **Linear ramp**: activation = base × (0.3 + 2.7 × progress) |
| At deadline | **Peak**: activation = base × 3.0 |
| Past deadline | **Rapid exponential decay**: peak × exp(-0.15 × hours_past) |

**Critical design**: Anticipatory activation **overrides** (not multiplies) base decay. A 200-hour-old obligation has base ≈ 0; multiplying gives 0. Override gives the correct rising activation.

**Step 5 — Emotional Boost** (`compute_emotional_boost`):

```
boost = 0.30 × exp(-0.10 × hours_since_detection)
```

Fast-fading additive boost: ~50% at 7h, ~10% at 23h. Emotional loading is a **signal**, not a category.

**Step 6 — Full Activation** (`compute_activation`) — THE main entry point:

```python
if anticipatory_active:
    result = anticipatory_override + emotional_boost
else:
    result = base_decay + emotional_boost

result = clamp(result, 0, 3.30)   # max = 1.0 × 3.0 + 0.30
if result < 0.01: result = 0.0     # staleness floor
```

**Reactivation**:

| Method | Purpose |
|--------|---------|
| `is_dormant(fact, ctx)` | True if eligible category + activation < 0.10 |
| `reactivate(fact, ts)` | Sets base_activation=0.70, resets decay clock (mutates FactNode) |

Eligible categories: HOBBIES_RECREATION, INTELLECTUAL_INTERESTS, RELATIONAL_BONDS.

**Batch Operations**:

| Method | Purpose |
|--------|---------|
| `rank_facts(facts, ctx)` | Returns `[(fact, activation)]` sorted descending |
| `filter_stale(facts, ctx, threshold=0.05)` | Returns only facts above threshold |
| `top_k(facts, ctx, k=10)` | Returns top-k by activation |

**Diagnostics**:

| Method | Purpose |
|--------|---------|
| `activation_curve(fact, hours_range, ...)` | `[(hours, activation)]` for plotting decay curves |
| `anticipatory_curve(fact, hours_before, hours_after)` | `[(hours_rel_to_deadline, activation)]` for deadline plots |
| `category_report(fact)` | Per-category breakdown: weight, rate, half-life; blended rate |

#### The 3 Clock Types

| Clock | What It Measures | Highest Sensitivity |
|-------|-----------------|-------------------|
| **Absolute** (`s_abs`) | Wall-clock hours since fact last updated | LOGISTICAL (0.8), OBLIGATIONS (0.7) |
| **Relative** (`s_rel`) | Hours since user's last session start | RELATIONAL (0.6), HOBBIES (0.5) |
| **Conversational** (`s_conv`) | Messages in session × 0.5h/msg | IDENTITY (0.3), INTELLECTUAL (0.3), PROJECTS (0.3) |

**Rationale**: A meeting at 3pm (LOGISTICAL) is calendar-anchored. "Haven't talked to mom in 3 months" (RELATIONAL) is session-gap anchored. Identity facts barely decay regardless of clock.

#### Ablation Presets (Factory Methods)

| # | Method | What It Tests |
|---|--------|--------------|
| — | `default()` | Full system: behavioral ontology + multi-clock + anticipatory + emotional |
| 1 | `uniform()` | Single rate (0.010) for all categories. Tests whether per-category rates matter. |
| 2 | `single_clock()` | Absolute time only. Tests multi-clock routing value. |
| 3 | `no_anticipatory()` | No deadline activation override. Tests Goal 3 contribution. |
| 8 | `no_emotional()` | No emotional boost. Tests emotion signal value. |
| 9 | `cognitive()` | **THE critical ablation.** 3 cognitive types replace 11 behavioral categories. If it matches behavioral, the thesis is disproven. |
| 10 | `arithmetic_blend()` | Arithmetic mean of activations instead of harmonic mean of rates. |
| 11 | `max_only()` | Primary cluster only, no soft blending. |
| 12 | `ontology_8plus1()` | Collapses HOBBIES + PREFERENCES back into IDENTITY. Tests 11→8 category value. |

Ablations 4–7 are not defined in this module (likely test classifier or graph structure components).

#### Self-Test Suite

`run_tests()` — 11 test groups covering invariants:
1. Health decays slower than Logistical (24h)
2. Identity decays slowest of all (30 days)
3. Hobby reactivates after dormancy (2000h)
4. Preference decays faster than Identity (7 days)
5. Multi-cluster fact decays between component rates
6. Lazy computation consistency (memoryless)
7. Anticipatory rises before deadline, drops after
8. Emotional boost is additive and fast-decaying
9. Uniform ablation makes Health = Logistical
10. Cognitive ablation collapses Health + Preference to SEMANTIC
11. 8+1 ablation collapses Hobbies + Preferences into Identity

---

### 2.3 Bridge: `graphiti_bridge.py`

**Purpose**: Adapter layer between `decay_engine.py` (pure math) and Graphiti (knowledge graph). Wires the decay engine into Graphiti **without modifying Graphiti source**.

| Aspect | Detail |
|--------|--------|
| Lines | 590 |
| Dependencies | `time`, `datetime`, `typing` (stdlib); `decay_engine` (project) |
| Graphiti imports | **ZERO** — entirely duck-typed |

#### Architecture: Zero-Modification Integration

The bridge uses four techniques to integrate without touching Graphiti's code:

1. **Duck-typing on `EntityNode.attributes`**: Graphiti persists this dict as Neo4j node properties. The bridge writes `fr_`-prefixed keys into it.
2. **Post-processing pattern**: Ingestion → `graphiti.add_episode()` → `enrich_entity_node()` → `node.save()`. Retrieval → `graphiti.search()` → `rerank_by_activation()`. No monkey-patching.
3. **Multi-pattern result extraction**: `_compute_result_activation()` tries multiple duck-typing patterns for different Graphiti search result shapes.
4. **Private attribute for edges**: `enrich_entity_edge()` creates `_fr_metadata` on edge instances (since edges lack `attributes`).

#### Attribute Constants

All prefixed with `fr_` (Fortunate Recall) to avoid collisions:

| Constant | Neo4j Property | Type |
|----------|---------------|------|
| `ATTR_PRIMARY_CATEGORY` | `fr_primary_category` | string |
| `ATTR_MEMBERSHIP_WEIGHTS` | `fr_membership_weights` | string (JSON) |
| `ATTR_FUTURE_ANCHOR_TS` | `fr_future_anchor_ts` | float or null |
| `ATTR_EMOTIONAL_LOADING` | `fr_emotional_loading` | boolean |
| `ATTR_EMOTIONAL_LOADING_TS` | `fr_emotional_loading_ts` | float or null |
| `ATTR_LAST_UPDATED_TS` | `fr_last_updated_ts` | float |
| `ATTR_ACCESS_COUNT` | `fr_access_count` | integer |
| `ATTR_LAST_REACTIVATION_TS` | `fr_last_reactivation_ts` | float or null |
| `ATTR_CONFIDENCE` | `fr_confidence` | string |
| `ATTR_ENRICHED` | `fr_enriched` | boolean (sentinel) |

#### Public API Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `enrich_entity_node` | `(entity_node, classification, reference_time=None)` | Stamps all `fr_*` attributes on a Graphiti EntityNode before `node.save()` |
| `enrich_entity_edge` | `(entity_edge, classification, reference_time=None)` | Same but for EntityEdge (creates `_fr_metadata` dict) |
| `set_future_anchor` | `(entity_node, deadline: datetime)` | Sets `fr_future_anchor_ts` for anticipatory activation |
| `is_enriched` | `(entity_node) -> bool` | Checks `fr_enriched` sentinel |
| `entity_to_fact_node` | `(entity_node) -> Optional[FactNode]` | Converts Graphiti node → decay engine FactNode; returns None if unenriched |
| `build_temporal_context` | `(last_session_ts, session_message_count, now) -> TemporalContext` | Constructs temporal context with `absolute_hours=0.0` placeholder (computed per-fact later) |
| `rerank_by_activation` | `(search_results, ctx, engine, blend_weight=0.5) -> list` | Blends Graphiti score with decay activation: `(1-α) × graphiti + α × activation` |
| `record_access` | `(entity_node, driver=None)` | Increments access counter, updates timestamp |
| `inspect_node` | `(entity_node, engine) -> dict` | Diagnostic report: category, weights, activation, half-lives |

#### Key Design: Neutral Fallback

Unenriched nodes get **0.5** (midpoint) for both activation and Graphiti score — neither boosted nor penalized. This allows graceful degradation during partial enrichment.

#### Format Translations

The bridge handles conversions that neither system should know about:
- **Weights dict ↔ JSON string**: Python dicts for decay engine; JSON strings for Neo4j
- **datetime ↔ Unix timestamp**: Python datetimes at the API; floats in storage
- **Missing categories**: Ensures all 11 categories present in weights dict

#### Self-Test Suite

`_run_bridge_tests()` — 7 test groups using `MockEntityNode`:
1. Enrichment attributes stamped correctly
2. EntityNode → FactNode mapping (all 11 categories present)
3. Activation is positive and decayed at 24h
4. Unenriched handling returns None
5. Future anchor timestamp storage
6. Diagnostic inspect_node returns expected fields
7. Access counter increment

---

### 2.4 Ingestion Scripts

Two scripts that load LongMemEval conversations into Neo4j via Graphiti, then classify entities with the behavioral ontology.

---

#### `ingest_poc.py`

**Purpose**: Ingests **ALL** questions from the oracle dataset (no filtering). Despite its name suggesting "PoC", this is the broader, unfiltered variant.

| Aspect | Detail |
|--------|--------|
| Lines | ~680 |
| Dependencies | `openai`, `graphiti_core`, `decay_engine`, `graphiti_bridge` |
| LLM (Graphiti) | `grok-4-1-fast-reasoning` via xAI |
| LLM (classification) | `grok-4-1-fast-non-reasoning` via xAI |
| Embedder | OpenAI (default config) |
| Cost estimate | ~$0.20/session |
| Group ID | `"full_234"` |

**Functions**:

| Function | Purpose |
|----------|---------|
| `load_env()` | Reads `.env`, sets env vars |
| `get_graphiti_client()` | Creates Graphiti with xAI LLM + OpenAI embedder + Neo4j |
| `hash_session(session)` | SHA-256 first 16 hex chars for deduplication |
| `build_session_map(questions)` | Deduplicates sessions across questions by content hash |
| `load_checkpoint()` / `save_checkpoint()` | Atomic checkpoint I/O (write to `.tmp`, then `Path.replace()`) |
| `load_enrich_checkpoint()` / `save_enrich_checkpoint()` | Same for enrichment phase |
| `format_time(seconds)` | Human-readable time formatting |
| `print_progress(done, total, elapsed, errors)` | In-place progress bar with ETA and cost |
| `load_all_questions()` | Loads ALL oracle questions, attaches haystack sessions from s_cleaned |
| `classify_entity(name, summary, client)` | Async LLM classification → behavioral ontology |
| `run_ingestion()` | Phase 1: sequential `graphiti.add_episode()` per session |
| `run_enrichment()` | Phase 2: classify + `enrich_entity_node()` per entity |
| `show_status()` | Read-only checkpoint summary |
| `reset_checkpoints()` | Deletes all checkpoint/artifact files |
| `main()` | CLI: `--phase ingest\|enrich\|both\|status\|reset` |

**Checkpoint/Resume**:
- **Ingestion**: Tracks completed session hashes in a list. O(1) lookup via set conversion on resume. Saves after EVERY session.
- **Enrichment**: Tracks completed entity UUIDs. Also checks `is_enriched()` as secondary guard. Saves every 10 entities.
- **Circuit breaker**: 5 consecutive ingestion failures → clean save-and-exit.
- **Atomic writes**: Write to `.tmp` → `Path.replace()` to prevent corruption.

**Session Processing**:
- Sessions identified by SHA-256 content hash (first 16 hex chars)
- Named `full_s{index:04d}` (e.g., `full_s0000`)
- Sequential ingestion (no async batching — one `add_episode()` at a time)

---

#### `ingest_full.py`

**Purpose**: Ingests only a **filtered subset** of questions matching 4 "must-win" types. Despite its name suggesting "full", this is the narrower, focused variant.

| Aspect | Detail |
|--------|--------|
| Lines | ~680 |
| Dependencies | Identical to `ingest_poc.py` |
| Cost estimate | ~$0.014/session (calibrated from PoC actual spending) |
| Group ID | `"full_234"` |

**Key Difference — Question Filtering**:

```python
MUST_WIN_TYPES = {
    'single-session-user',
    'single-session-assistant',
    'single-session-preference',
    'knowledge-update',
}
filtered = [q for q in oracle if q['question_type'] in MUST_WIN_TYPES]
```

| Aspect | `ingest_poc.py` | `ingest_full.py` |
|--------|----------------|-----------------|
| Question filter | **None** (all) | 4 must-win types |
| Cost/session | $0.20 (initial estimate) | $0.014 (calibrated) |
| `MUST_WIN_TYPES` | Does not exist | Defined |
| `load_all_questions()` returns | `oracle` (all) | `filtered` (subset) |

All other code is byte-for-byte identical.

**Naming Paradox**: The PoC script ingests everything; the "full" script filters. The cost estimates reveal the timeline: `ingest_poc.py` used a rough $0.20/session estimate; `ingest_full.py` was written later with calibrated $0.014/session from actual PoC data.

---

### 2.5 Evaluation Evolution: v2 → v3 → v4

Three successive iterations of the kill-gate evaluation harness.

---

#### `evaluate_v2.py`

**Purpose**: Second iteration. Fixes two v1 bugs: (1) entity attribute lookup via direct Neo4j query instead of relying on search result objects; (2) improved answer matching with normalization and token overlap.

| Aspect | Detail |
|--------|--------|
| Lines | ~500 |
| Metric | hit@5 |
| Pool size | 10 (Graphiti search only) |
| Reranking | Blended: 50% semantic + 50% activation |
| Activation source | **Entity-level** (average of source/target node activations) |
| LLM | `grok-4-1-fast-non-reasoning` |

**Functions**:

| Function | Purpose |
|----------|---------|
| `_fetch_entity_fr_attrs(driver, uuid)` | Async Neo4j query for entity's `fr_*` properties; cached in `_entity_cache` |
| `_attrs_to_fact_node(attrs, id)` | Converts Neo4j attrs → `FactNode` |
| `compute_edge_activation(edge, driver, ctx, engine)` | Looks up source/target entity nodes, averages their activations |
| `rerank_with_neo4j_lookup(results, driver, ctx, engine, α=0.5)` | Blended reranking: `(1-α) × graphiti + α × activation` |
| `_normalize(s)` | Lowercase, strip punctuation, collapse whitespace |
| `_answer_match(expected, fact)` | Multi-strategy: substring → single-word boundary → token overlap ≥ 50% |
| `run_evaluation()` | Main loop: 25 questions × 3 engines, reports hit@5 per engine/type |

**Engines**: `DecayEngine.default()` (behavioral), `DecayEngine.uniform()`, `DecayEngine.cognitive()`

**Fundamental Flaw**: Entity-level activation is wrong. "User caught 7 bass" gets IDENTITY activation (from entity "User") instead of HOBBIES activation.

---

#### `evaluate_v3.py`

**Purpose**: Fixes v2's pool problem by building a fat candidate pool from 4 retrieval strategies. Adds MRR metric and `no_rerank` baseline. Still uses entity-level activation.

| Aspect | Detail |
|--------|--------|
| Lines | ~600 |
| Metrics | hit@5 + MRR + pool ceiling |
| Pool size | Variable, 50-300+ candidates |
| Reranking | **Pure activation** (no semantic blending) |
| Activation source | **Entity-level** (same flaw as v2) |
| New baseline | `no_rerank` (Graphiti's original ordering) |

**New Components**:

| Component | Purpose |
|-----------|---------|
| `STOPWORDS` | ~90 English stopwords for keyword extraction |
| `class Candidate` | `__slots__`-based: uuid, fact, source_uuid, target_uuid, source |
| `extract_keywords(text)` | Alphabetic words ≥ 3 chars, not in STOPWORDS |
| `build_candidate_pool(question, driver, graphiti, group_id)` | 4-strategy fat pool builder |

**4 Retrieval Strategies**:

| # | Strategy | Cypher/API | Limit |
|---|----------|-----------|-------|
| 1 | Graphiti semantic search | `graphiti.search()` | 50 results |
| 2 | Cypher keyword search | `WHERE toLower(e.fact) CONTAINS $kw` | 20/keyword × 8 keywords |
| 3 | Multi-keyword intersection | Two keywords AND on same edge | 10/pair × 6 pairs |
| 4 | Entity neighborhood | Entity name match → connected edges | 30/keyword × 5 keywords |

**Deduplication**: By edge UUID across all strategies.

**Why pure activation reranking**: Cypher-sourced candidates have no Graphiti semantic score. Blending would bias toward Graphiti candidates.

**Result**: Revealed that behavioral performed **WORSE** than uniform (MRR 0.030 vs 0.052) — proving entity-level classification was the wrong abstraction.

---

#### `evaluate_v4.py`

**Purpose**: The definitive fix. Three architectural changes: (1) edge-level classification, (2) blended reranking with alpha parameter, (3) per-question temporal anchors. This is the version designed to produce publishable results.

| Aspect | Detail |
|--------|--------|
| Lines | ~1,100 |
| Metrics | hit@5 (all + answerable), MRR, per-question rank comparison, win/loss tallies |
| Pool size | Variable (same 4-strategy fat pool) |
| Reranking | **Blended**: `α × semantic + (1-α) × activation` |
| Activation source | **Edge-level** (fact text classified directly) |
| Alpha sweep | [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] |
| LLM | `grok-4-1-fast-reasoning` (upgraded) |

**Phase A — Edge Classification** (`enrich_edges`):

| Function | Purpose |
|----------|---------|
| `CLASSIFY_EDGE_PROMPT` | System prompt for classifying edge fact text into 11 categories |
| `classify_edge_fact(fact, client)` | Async LLM call with 500-char truncation, 3 retries, markdown stripping |
| `enrich_edges(driver, group_id)` | Batch: 50 edges at a time, 10 concurrent (semaphore), writes `fr_*` to Neo4j edges |

**Phase B — Evaluation** (`run_evaluate`):

| Function | Purpose |
|----------|---------|
| `_edge_attrs_to_fact_node(attrs, uuid)` | Converts edge's `fr_*` attrs → `FactNode` (uses `created_at_ts`) |
| `_to_unix_ts(value)` | Neo4j datetime / Python datetime / float → Unix seconds |
| `build_question_temporal_anchors(driver, group_id, questions, now)` | Per-question temporal context from ingested session data |
| `compute_edge_activation_v4(uuid, ctx, engine)` | Edge-level activation from `_edge_cache` |
| `rerank_candidates(candidates, ctx, engine, alpha=0.5)` | `α × graphiti_score + (1-α) × activation` |
| `run_evaluate(graphiti, group_id, alpha, suffix)` | Full eval loop returning structured summary dict |
| `print_alpha_sweep_summary(summaries)` | Compact MRR comparison table across alpha values |

**Per-Question Temporal Anchors** (new in v4):
- Queries Episodic nodes to find each question's last haystack session
- Gets `max(created_at)` of edges linked to that episode
- Each question gets its own `current_timestamp` in the temporal context

**LLM Answer Oracle** (new in v4):
- Reads `answer_oracle.json` (produced by `llm_answer_judge.py`)
- Maps `question_id → set[edge_uuids]` for LLM-verified answer edges
- Falls back to substring matching if oracle unavailable

**Synthetic Semantic Scores for Cypher Candidates**:
```python
CYPHER_SOURCE_BASELINES = {'cypher_kw': 0.4, 'cypher_intersect': 0.5, 'cypher_neighbor': 0.3}
```
Graphiti results get rank-based scores: `1.0 - (i / len(results))`

**CLI**:
```
--enrich-edges    Phase A: classify all edges
--evaluate        Phase B: run evaluation
--alpha 0.3       Custom blend weight
--sweep           Run alpha sweep (0.2-0.8)
```

#### Evolution Summary Table

| Aspect | v2 | v3 | v4 |
|--------|----|----|-----|
| Pool | 10 | 50-300+ | 50-300+ |
| Activation | Entity-level | Entity-level | **Edge-level** |
| Reranking | Blended (0.5) | Pure activation | **Blended (alpha sweep)** |
| Metrics | hit@5 | hit@5 + MRR | hit@5 + MRR + per-Q rank + wins |
| Baselines | 3 engines | + no_rerank | + answerable filtering |
| Temporal | Global (now-24h) | Global (now-24h) | **Per-question anchors** |
| Answer match | Substring + token | Same | **LLM oracle** + fallback |
| LLM model | non-reasoning | non-reasoning | **reasoning** |
| Result format | Print only | Print only | **Structured dict** + per-alpha files |
| Cache | Entity UUIDs | Entity UUIDs | **Edge UUIDs** |

---

### 2.6 Supporting Tools

---

#### `fix_timestamps.py`

**Purpose**: Linearly remaps edge `created_at` timestamps across a 6-month window. Fixes the evaluation artifact where all edges were bulk-ingested within ~3.5 hours, making decay-based scoring useless (all activations ≈ 1.0).

| Aspect | Detail |
|--------|--------|
| Lines | ~230 |
| Dependencies | `neo4j` (AsyncGraphDatabase) |
| Target span | 180 days |
| Batch size | 200 edges per UNWIND |

**Steps**:
1. Fetch all RELATES_TO edges ordered by `created_at` ascending
2. Map oldest → `now - 180 days`, newest → `now - 1 hour`
3. Linear interpolation for all intermediate edges
4. Batch update via Cypher `UNWIND`
5. Verify spread with min/max query
6. Print decay differentiation table (IDENTITY: 0.646 at 6mo vs LOGISTICAL: 0.000)

---

#### `reenrich_edges_v7.py`

**Purpose**: Re-classifies all edges with the production "v7" classification prompt. Earlier versions used a simplified 10-line classifier with misclassifications. Also fixes a critical bug where a non-identity `V7_TO_DECAY` mapping caused 8/11 categories to silently fall back to uniform decay.

| Aspect | Detail |
|--------|--------|
| Lines | ~340 |
| Dependencies | `openai`, `neo4j` |
| LLM | `grok-4-1-fast-non-reasoning` |
| Concurrency | Semaphore(10) per batch of 50 |

**Key Constants**:
- `V7_CATEGORIES`: 11 canonical category names
- `V7_TO_DECAY`: Identity mapping `{cat: cat}` (previous bug: non-identity mapping caused silent fallback)
- `SYSTEM_PROMPT`: Full v7 prompt with all guardrails, boundary tests, and weighted membership rules

**Steps**:
1. Clear old `fr_*` properties on all edges
2. Classify in batches (50 edges, 10 concurrent LLM calls)
3. `validate_and_normalize()` → `map_to_decay_engine()` → write to Neo4j
4. Print distribution report
5. Sanity check: query edges with known-tricky words (bass, fish, comedy, cocktail) and verify classifications

---

#### `diagnose_retrieval.py`

**Purpose**: Diagnostic tool that isolates where the pipeline fails per question — distinguishing extraction failures from retrieval failures from ranking failures.

| Aspect | Detail |
|--------|--------|
| Lines | ~340 |
| Dependencies | `openai`, `graphiti_core` |
| Output | `retrieval_diagnosis.json` + console report |

**Three-Check Diagnosis**:

| Check | Method | Finding |
|-------|--------|---------|
| 1 — Edge search | Cypher `WHERE toLower(e.fact) CONTAINS $answer` | Is the answer in the graph at all? |
| 2 — Node search | Cypher `WHERE toLower(n.name) CONTAINS $word` | Is the entity in the graph? |
| 3 — Graphiti search | `graphiti.search(question, num_results=50)` | Does semantic search find it? At what rank? |

**Status Color Codes**:
- 🔴 RED: Not in graph
- 🔵 BLUE: In graph, search misses it
- 🟠 ORANGE: Found but ranked > 10
- 🟡 YELLOW: In top 10 but not top 5
- 🟢 GREEN: In top 5

---

#### `llm_answer_judge.py`

**Purpose**: Two-stage pipeline replacing brittle substring matching with LLM-verified answer matching. Produces `answer_oracle.json` consumed by `evaluate_v4.py`.

| Aspect | Detail |
|--------|--------|
| Lines | ~500 |
| Dependencies | `openai`, `graphiti_core` |
| LLM | `grok-4-1-fast-reasoning` (temperature=0, max_tokens=5) |
| Concurrency | Semaphore(10) |
| Cache | Persistent JSON keyed by SHA-256 of (question_id, edge_uuid) |
| Cost | ~$0.06 for 25 questions |

**Two-Stage Pipeline**:

| Stage | Method | Purpose |
|-------|--------|---------|
| 1 — Keyword pre-filter | `prefilter_candidate()` | Free check: ≥1 question keyword AND ≥1 answer keyword in fact (eliminates ~90% of candidates) |
| 2 — LLM judge | `judge_one()` | Grok YES/NO determination on survivors |

**Pre-filter Pass Conditions** (any one):
- ≥1 question keyword AND ≥1 answer keyword in fact
- ≥2 question keywords in fact
- ≥1 question keyword AND answer number appears as standalone in fact
- Answer is 1-2 words, whole-word match, with ≥1 question keyword

**LLM Judge Rules** (from `JUDGE_SYSTEM`):
- YES if fact contains ANY specific piece of expected answer
- One list item is enough
- Information must be SPECIFIC not just topical
- Numbers must match
- Future plans are not past events

**Output**: `answer_oracle.json` — `{question_id: [edge_uuid, ...]}` mapping of verified answer edges.

**Critical Sync Requirement**: `build_candidate_pool()` must match `evaluate_v4.py`'s implementation for oracle to be valid.

---

#### `check_sessions.py`

**Purpose**: Quick estimation script — counts unique sessions across PoC question types, estimates ingestion cost.

| Aspect | Detail |
|--------|--------|
| Lines | ~25 |
| Dependencies | `json`, `hashlib` |
| Filter | 4 question types: single-session-user, single-session-assistant, single-session-preference, knowledge-update |
| Cost | `unique_sessions × $0.20` |

---

#### `cost.py`

**Purpose**: Extrapolates PoC cost to full-scale run based on actual spending data.

| Aspect | Detail |
|--------|--------|
| Lines | ~15 |
| Dependencies | `json` |
| PoC cost | $17 (hardcoded from actual run) |
| Full dataset | 9,977 sessions |
| Formula | `(17 / poc_sessions) × 9977` |

---

#### `enrich_and_evaluate.py`

**Purpose**: Crash-recovery bootstrap script. Assumes sessions are in Neo4j but entity enrichment may be incomplete. Runs enrichment → kill gate evaluation in one pass.

| Aspect | Detail |
|--------|--------|
| Lines | ~490 |
| Dependencies | `openai`, `graphiti_core`, `decay_engine`, `graphiti_bridge` |
| LLM | `grok-4-1-fast-non-reasoning` |
| Enrichment batch | 10 entities, parallel classification, serial Neo4j writes |

**Key Design — Bypasses `node.save()`**: Writes `fr_*` properties via direct Cypher `SET` statements. This avoids triggering Graphiti's embedding recomputation, which has a known bug.

**Two Phases**:
1. **Enrichment**: Fetch unenriched entities → classify → `write_enrichment_to_neo4j()` via Cypher
2. **Evaluation**: Load 25 questions → search + rerank with 3 engines (behavioral, uniform, cognitive) → report hit@5 per engine/type → verdict

---

## 3. Pipeline Flow Diagram

```
                         ┌─────────────────────────────────────┐
                         │       LongMemEval Raw Data          │
                         │  longmemeval_oracle.json (15 MB)    │
                         │  longmemeval_s_cleaned.json (277 MB)│
                         │  longmemeval_m_cleaned.json (2.7 GB)│
                         └───────────┬─────────────────────────┘
                                     │
              ┌──────────────────────┼───────────────────────────┐
              │                      │                           │
              v                      v                           v
   ┌──────────────────┐  ┌──────────────────┐       ┌──────────────────┐
   │  extract_facts.py │  │estimate_ingestion│       │ check_sessions.py│
   │  (Stage 1)        │  │  _cost.py        │       │ cost.py          │
   │                   │  │  (utility)       │       │ (estimation)     │
   │  oracle.json      │  │  s_cleaned.json  │       │                  │
   │  → extracted_     │  │  → console       │       │                  │
   │    facts.json     │  │                  │       │                  │
   └────────┬──────────┘  └──────────────────┘       └──────────────────┘
            │
            v
   ┌──────────────────┐
   │ classify_facts.py │
   │ (Stage 2)         │
   │                   │
   │ LLM: Grok 4.1     │
   │ 11-cat ontology   │
   │ → classified_     │
   │   facts.json      │
   └────────┬──────────┘
            │
            v
   ┌──────────────────────┐
   │ generate_audit_       │
   │   prompt.py           │
   │ (Stage 3)             │
   │                       │
   │ Stratified sample     │
   │ → audit_prompt.txt    │
   │ → Human/LLM QA audit  │
   └───────────────────────┘


   ┌──────────────────────────────────────────────────────────────┐
   │                    INGESTION PIPELINE                        │
   │                                                              │
   │  ingest_poc.py (all questions) ─────┐                        │
   │  ingest_full.py (must-win only) ────┤                        │
   │                                     │                        │
   │  Phase 1: Sessions → graphiti.add_episode() → Neo4j          │
   │  Phase 2: Entities → classify → enrich_entity_node() → save  │
   └──────────────────────────────────────┬───────────────────────┘
                                          │
                                          v
   ┌──────────────────────────────────────────────────────────────┐
   │                    NEO4J GRAPH STATE                          │
   │                                                              │
   │  Entity nodes with fr_* attributes (behavioral ontology)     │
   │  RELATES_TO edges (facts between entities)                    │
   │  Episodic nodes (ingested sessions)                          │
   └───────────┬──────────────────────────────────────────────────┘
               │
               │  Post-ingestion Fixes
               │
               ├──── fix_timestamps.py ──── Linear remap → 6-month spread
               │
               ├──── reenrich_edges_v7.py ── Re-classify edges with v7 prompt
               │
               ├──── enrich_and_evaluate.py  Crash-recovery enrichment + eval
               │
               v
   ┌──────────────────────────────────────────────────────────────┐
   │                    EVALUATION PIPELINE                        │
   │                                                              │
   │  diagnose_retrieval.py ── Pipeline failure diagnosis          │
   │        │                                                     │
   │        v                                                     │
   │  llm_answer_judge.py ── answer_oracle.json (verified edges)  │
   │        │                                                     │
   │        v                                                     │
   │  evaluate_v2.py ── Entity-level, 10-result pool, blended     │
   │        │                                                     │
   │        v                                                     │
   │  evaluate_v3.py ── Entity-level, fat pool, pure activation   │
   │        │           (revealed entity-level bug)               │
   │        v                                                     │
   │  evaluate_v4.py ── EDGE-level, fat pool, blended + sweep     │
   │                    (the publishable version)                 │
   └──────────────────────────────────────────────────────────────┘
               │
               v
   ┌──────────────────────────────────────────────────────────────┐
   │                    CORE COMPUTATION                           │
   │                                                              │
   │  decay_engine.py ◄──── graphiti_bridge.py ◄──── evaluate_v4  │
   │  (pure math,          (adapter, zero        (edge-level      │
   │   zero I/O,            Graphiti imports)      activation,     │
   │   11 categories,                              alpha sweep)   │
   │   3 clocks,                                                  │
   │   anticipatory,                                              │
   │   emotional boost)                                           │
   └──────────────────────────────────────────────────────────────┘
```

---

## 4. Key Constants & Configuration

### 4.1 The 11 Behavioral Categories with Decay Rates

| Category | λ (per hour) | Half-Life | Clock Sensitivity (abs, rel, conv) | Design Rationale |
|----------|-------------|-----------|-------------------------------------|-----------------|
| IDENTITY_SELF_CONCEPT | 0.0001 | ~289 days | (0.4, 0.3, 0.3) | "I'm Irish-Italian" — unchanged for years |
| RELATIONAL_BONDS | 0.001 | ~29 days | (0.2, 0.6, 0.2) | Relationships persist; inter-session gap matters |
| HEALTH_WELLBEING | 0.002 | ~14 days | (0.5, 0.3, 0.2) | Conditions persist but context shifts |
| INTELLECTUAL_INTERESTS | 0.002 | ~14 days | (0.3, 0.4, 0.3) | Curiosities dormant but reactivatable |
| HOBBIES_RECREATION | 0.003 | ~10 days | (0.3, 0.5, 0.2) | Skills persist; dormancy handled by reactivation |
| FINANCIAL_MATERIAL | 0.005 | ~6 days | (0.5, 0.3, 0.2) | Variable — purchases fade, debts persist |
| PROJECTS_ENDEAVORS | 0.008 | ~3.6 days | (0.5, 0.2, 0.3) | Milestone-gated; active projects stay relevant |
| PREFERENCES_HABITS | 0.010 | ~2.9 days | (0.6, 0.2, 0.2) | Tastes shift; supersession handles updates |
| OTHER | 0.010 | ~2.9 days | (0.4, 0.3, 0.3) | Default moderate rate |
| OBLIGATIONS | 0.050 | ~14 hours | (0.7, 0.1, 0.2) | Fast decay BUT anticipatory activation overrides pre-deadline |
| LOGISTICAL_CONTEXT | 0.080 | ~8.7 hours | (0.8, 0.1, 0.1) | "Meeting at 3pm" irrelevant by tomorrow |

### 4.2 Cognitive Typology Mapping (Ablation 9)

| Behavioral Category | Cognitive Type | Cognitive λ |
|--------------------|---------------|-------------|
| OBLIGATIONS | SEMANTIC | 0.005 |
| RELATIONAL_BONDS | SEMANTIC | 0.005 |
| HEALTH_WELLBEING | SEMANTIC | 0.005 |
| IDENTITY_SELF_CONCEPT | SEMANTIC | 0.005 |
| PREFERENCES_HABITS | SEMANTIC | 0.005 |
| INTELLECTUAL_INTERESTS | SEMANTIC | 0.005 |
| FINANCIAL_MATERIAL | SEMANTIC | 0.005 |
| HOBBIES_RECREATION | PROCEDURAL | 0.003 |
| PROJECTS_ENDEAVORS | PROCEDURAL | 0.003 |
| LOGISTICAL_CONTEXT | EPISODIC | 0.050 |
| OTHER | SEMANTIC | 0.005 |

### 4.3 8+1 Collapse Mapping (Ablation 12)

```
HOBBIES_RECREATION   → IDENTITY_SELF_CONCEPT
PREFERENCES_HABITS   → IDENTITY_SELF_CONCEPT
```

### 4.4 Anticipatory Activation Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `anticipatory_window_hours` | 168 (7 days) | Activation starts rising this far before deadline |
| `anticipatory_peak_multiplier` | 3.0 | At deadline: activation = base × 3 |
| `anticipatory_post_decay_rate` | 0.15/hr | Rapid exponential decay AFTER deadline |
| Floor at window edge | 0.3 (30%) | Minimum activation when entering anticipatory window |

### 4.5 Emotional Loading Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `emotional_boost_initial` | 0.30 | Additive boost when emotion detected |
| `emotional_boost_decay_rate` | 0.10/hr | Emotional boost half-life ≈ 7 hours |

### 4.6 Reactivation Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `reactivation_threshold` | 0.10 | Below this → dormant |
| `reactivation_boost` | 0.70 | On signal, activation reset to this |
| Eligible categories | HOBBIES, INTELLECTUAL, RELATIONAL | Only these categories can reactivate |

### 4.7 Classifier Prompt Evolution

| Version | Used In | Key Changes |
|---------|---------|------------|
| v1 (simple) | Early `ingest_poc.py` | 10-line system prompt, 11 categories with 1-line descriptions |
| v7 (production) | `classify_facts.py`, `reenrich_edges_v7.py` | Full boundary tests, PREFERENCES guardrails, FINANCIAL guardrail, deliverable test, named person disambiguation, emotional loading, weighted membership |
| Edge variant | `evaluate_v4.py` | Adapted v7 for edge fact text instead of entity name/summary |
| Ingest variant | `ingest_poc.py`, `ingest_full.py` (`CLASSIFY_SYSTEM`) | Intermediate version for entity classification during ingestion |

### 4.8 LLM Models Used

| Model | Where Used | Purpose |
|-------|-----------|---------|
| `grok-4-1-fast-reasoning` | `evaluate_v4.py`, `llm_answer_judge.py`, `ingest_poc.py` (Graphiti LLM) | Reasoning model for complex classification and judgment |
| `grok-4-1-fast-non-reasoning` | `classify_facts.py`, `reenrich_edges_v7.py`, `evaluate_v2/v3.py`, `enrich_and_evaluate.py` | Cheaper non-reasoning variant for classification |
| OpenAI embedder (default) | All Graphiti clients | Embedding vectors for semantic search |

### 4.9 Alpha Sweep Values

```python
ALPHA_SWEEP_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# blended = alpha × semantic + (1-alpha) × activation
# alpha=0.0 → pure activation | alpha=1.0 → pure semantic
```

### 4.10 Cypher Source Baselines

```python
CYPHER_SOURCE_BASELINES = {
    'cypher_kw': 0.4,
    'cypher_intersect': 0.5,
    'cypher_neighbor': 0.3,
}
```

---

## 5. Design Patterns

### 5.1 Pure Math Module (decay_engine.py)

The decay engine has **zero external dependencies** and **zero I/O**. All state is passed as arguments; all results are returned as values. This enables:
- Deterministic unit testing without mocks
- Import into any Python context (notebook, script, server)
- No hidden state or caching bugs on the hot path

### 5.2 Bridge / Adapter Pattern (graphiti_bridge.py)

Classic unidirectional adapter that converts between two data models without coupling them:
```
Graphiti EntityNode.attributes → [bridge] → decay_engine FactNode → activation
```
The bridge has **zero Graphiti imports** — it relies entirely on duck-typing and the `attributes` dict as the persistence seam. The `fr_` prefix provides namespace isolation.

### 5.3 Async Batched Concurrency

Used in classification scripts (`classify_facts.py`, `reenrich_edges_v7.py`, `evaluate_v4.py`):
```python
semaphore = asyncio.Semaphore(10)  # cap concurrent LLM calls
batch = edges[i:i+50]              # process in chunks
results = await asyncio.gather(*[classify_with_sem(e) for e in batch])
```
Prevents overwhelming the API while achieving parallelism. Sequential between batches.

### 5.4 Checkpoint / Resume

Used in ingestion scripts (`ingest_poc.py`, `ingest_full.py`):
- **Atomic writes**: Write to `.tmp` → `Path.replace()` (prevents corruption)
- **Session-level granularity**: Save after every session (ingestion) or every 10 entities (enrichment)
- **Dual-check**: Track UUIDs in checkpoint + query `fr_enriched` attribute
- **Circuit breaker**: 5 consecutive failures → clean save-and-exit

### 5.5 Fat Candidate Pool (evaluate_v3+)

Instead of relying solely on Graphiti's top-N semantic results, the evaluation builds a hybrid candidate pool from 4 strategies (semantic, keyword, intersection, neighborhood). This isolates the decay engine's contribution from Graphiti's retrieval quality.

### 5.6 LLM-as-Judge (llm_answer_judge.py)

Two-stage pipeline:
1. **Free keyword pre-filter** eliminates ~90% of candidates
2. **LLM judge** makes YES/NO determination on survivors
3. Persistent cache (SHA-256 keyed) makes re-runs nearly free

### 5.7 Strategy Pattern via Config (DecayConfig)

All behavioral variations are controlled by `DecayConfig` fields. The engine branches on config flags rather than using subclasses. Factory methods (`DecayEngine.uniform()`, `.cognitive()`, etc.) provide clean ablation APIs.

### 5.8 Override vs. Multiply (Anticipatory Activation)

Anticipatory activation **replaces** base decay rather than multiplying it. This is documented with clear rationale: a 200-hour-old obligation has base ≈ 0; `0 × 3.0 = 0` defeats the purpose.

### 5.9 Progressive Enrichment (LongMemEval Pipeline)

Each pipeline stage adds to records without removing prior fields:
```
extract_facts.py  → question_id, question, answer, user_utterance
classify_facts.py → + classification, classification_raw, tokens_in, tokens_out
generate_audit.py → reads all of the above for audit sampling
```

### 5.10 Reproducibility Convention

- `random.seed(42)` in extraction and audit scripts
- `temperature=0` for all LLM classification calls
- Deterministic sort order (by type, then ID) in extraction
- Content-based session deduplication via SHA-256

---

## 6. Current State

### Kill Gate: PASSED

The PoC kill gate evaluation has been completed. The behavioral ontology (11 categories, multi-clock routing, harmonic blending) demonstrates measurable improvement over both uniform decay and cognitive typology baselines.

### Phase 2: Beginning

With the kill gate passed, the project is entering Phase 2:
- Scale from 25 PoC questions to the full 234-question LongMemEval benchmark
- Run the complete 12-ablation study
- Produce paper figures from activation curves and alpha sweep data
- Target: NeurIPS 2026 submission

### File Maturity

| File | Status | Notes |
|------|--------|-------|
| `decay_engine.py` | **Stable** | Core math complete, all ablations defined, self-tests passing |
| `graphiti_bridge.py` | **Stable** | Bridge pattern complete, self-tests passing |
| `classify_facts.py` | **Stable** | v7 prompt finalized, audit complete |
| `evaluate_v4.py` | **Active** | The current evaluation harness; alpha sweep ready |
| `llm_answer_judge.py` | **Stable** | Oracle generation complete for PoC |
| `reenrich_edges_v7.py` | **Stable** | v7 re-classification complete for PoC |
| `fix_timestamps.py` | **Stable** | One-time fix applied |
| `ingest_poc.py` | **Stable** | PoC ingestion complete |
| `ingest_full.py` | **Ready** | Prepared for Phase 2 full ingestion |
| `evaluate_v2.py` | **Superseded** | Historical; kept for reference |
| `evaluate_v3.py` | **Superseded** | Historical; kept for reference |
| `diagnose_retrieval.py` | **Utility** | Available for debugging |
| `check_sessions.py` | **Utility** | One-off estimation |
| `cost.py` | **Utility** | One-off estimation |
| `enrich_and_evaluate.py` | **Recovery** | Crash-recovery fallback |
| `extract_facts.py` | **Stable** | Stage 1 complete |
| `estimate_ingestion_cost.py` | **Utility** | One-off estimation |
| `generate_audit_prompt.py` | **Stable** | Stage 3 complete |

### Known Issues / Technical Debt

1. **Naming paradox**: `ingest_poc.py` ingests all questions; `ingest_full.py` filters to must-win types. Names suggest the opposite.
2. **Code duplication**: `ingest_poc.py` and `ingest_full.py` are ~95% identical; could be refactored into a shared module with a filter parameter.
3. **Entity vs edge evolution**: `evaluate_v4.py` partially bypasses `graphiti_bridge.py`'s entity-level functions, implementing its own edge-level equivalents. The bridge could be extended to natively support edge-level operations.
4. **Stale comment**: `generate_audit_prompt.py` line 10 says "50 facts" but `SAMPLE_SIZE = 100`.
5. **`build_candidate_pool` sync requirement**: The function exists in both `llm_answer_judge.py` and `evaluate_v4.py` — must stay identical for oracle validity.

---

*Generated by comprehensive codebase analysis. All function signatures, constants, and design decisions verified against source files.*
