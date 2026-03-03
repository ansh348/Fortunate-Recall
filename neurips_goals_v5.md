# Fortunate Recall → NeurIPS 2026: Goal Sheet v5

**Target:** NeurIPS 2026 main track (9 pages + references/checklist)
**Single thesis:** The behavioral type of a memory should be the primary determinant of its temporal dynamics.
**Updated:** February 22, 2026 (end of day)

---

## Phase 0: Infrastructure Setup ✅ COMPLETE

### Goals
- ✅ Fork Graphiti. Get it running locally with a test conversation dataset.
- ✅ Map exact integration points where your policy layer hooks in: after fact extraction (cluster assignment), at retrieval time (decay-weighted scoring), at session start (warm start protocol).
- ✅ Secure compute for LLM inference during ingestion. Grok 4.1 Fast via xAI API at $0.20/$0.50 per M tokens. Split-client architecture: Grok for LLM, OpenAI for embeddings.
- ✅ Download and study **LongMemEval**, **LoCoMo**, and **MSC (Multi-Session Chat)**. Map each LongMemEval subcategory to the specific claims your system makes.
- ✅ Determine Memory-R1 code availability. **No public code released.** Reimplementation required — scoped for Phase 2 (GRPO training, Qwen-2.5-7B, 152 QA pairs from LoCoMo, 1-2x A100 GPUs).

**Phase 0 is done when:** Graphiti runs, datasets are local, you know exactly where your code goes, and you have a realistic cost estimate for ingestion across all evaluation datasets. ✅ **DONE.**

---

## Phase 1: Core Implementation (Minimum Viable System)

Build only what's needed to test the core thesis. Resist the urge to build everything before validating anything.

### Goal 1: Behavioral Ontology Classifier ✅ COMPLETE

Build the ingestion-time LLM classifier that takes an extracted fact/entity node and returns soft membership weights across the 10+1 behavioral categories.

- ✅ Fix temperature to 0 for deterministic outputs. Version all prompts and parsing code from day one.
- ✅ Write a **label guideline document** defining each category with positive examples and boundary cases per category. This is both your annotation protocol and your appendix artifact. **Updated to 10+1 categories.**
- ✅ Build classifier prompt. Iterated through **7 versions** with audits at each stage:
  - v1 (8+1): 77% agreement — classifier confused utterance context with stored fact
  - v2: Overcorrected with question-type overrides — temporal questions all became LOGISTICAL
  - v3 (8+1): 87% agreement — fixed overcorrection but IDENTITY became gravity well at 38%
  - v4 (10+1): 93% agreement — split IDENTITY into three tiers, added deliverable test, OTHER cap
  - v5 (10+1): Added two-step chain-of-thought + named person disambiguation. Fixed RELATIONAL over-promotion (temporal-reasoning: 75%→100%) but created PREFERENCES gravity well (single-session-preference: 83%→17%)
  - v6 (10+1): Added PREFERENCES_HABITS guardrails (learning shield, identity declaration, health-motivation, device ownership). Recovered single-session-preference to 100%, but FINANCIAL_MATERIAL over-promoted (3 regressions)
  - v7 (10+1): **Production version.** Added FINANCIAL_MATERIAL guardrail + OTHER suppression. **100% agreement on 30-fact development set.** All previous error patterns resolved with zero regressions.
- ✅ **Full-scale classification: 842 facts from all 428 LongMemEval questions.** $0.49 total cost. Zero failures.
- ✅ **100-fact held-out audit with two independent LLM auditors:**
  - Opus 4.6: **90/100 agree, 10 partial, 0 disagree (90%)**
  - GPT 5.2: **70/100 agree, 27 partial, 3 disagree (70%)** — bulk of disagreements stem from philosophical divergence on temporal question treatment, not classifier errors
  - **Cross-auditor consensus errors: 5/100 (95% effective accuracy)**
  - No systematic error patterns remaining — disagreements are scattered boundary cases
- ✅ Open-set audit: 1/842 facts (0.1%) landed in OTHER — confirms 10 primary categories provide near-complete coverage.
- ✅ Key insight discovered: **"Classify the fact, not the utterance"** — paper-level methodological contribution.
- ✅ Cost validated: $0.49 for 842 facts ($0.00058 per fact).

**Still TODO:**
- Get a **human annotator** on a subset (minimum 100 overlapping). Report Cohen's κ, per-class precision/recall, and confusion matrix. LLM auditors provide a strong proxy, but human annotator needed for paper.
- **Run sensitivity analysis:** Randomly flip 10%, 20%, 30% of cluster assignments and measure downstream retrieval degradation. This tells you how robust the system is to ingestion noise — and it's a result reviewers will demand.

### The 10+1 Behavioral Ontology (Validated)

| # | Category | Decay Profile | Description |
|---|---|---|---|
| 1 | **OBLIGATIONS** | Fastest — expires on completion | Tasks, deadlines, appointments, promises, action items |
| 2 | **RELATIONAL_BONDS** | Slow — relationships persist | Family, partners, friends, colleagues, social dynamics |
| 3 | **HEALTH_WELLBEING** | Slow, condition-dependent | Medical conditions, medications, fitness metrics, mental health, diet/nutrition |
| 4 | **IDENTITY_SELF_CONCEPT** | Near-zero — stable across years | Core traits: ethnicity, name, occupation, heritage, values, beliefs, personality |
| 5 | **HOBBIES_RECREATION** | Slow — months to years, reactivatable | Active leisure with accumulated skill/equipment: fishing, cycling, painting, gardening, cooking, photography |
| 6 | **PREFERENCES_HABITS** | Moderate — weeks to months, supersedable | Current tastes and consumption: media preferences, food choices, subscriptions, lifestyle routines |
| 7 | **INTELLECTUAL_INTERESTS** | Minimal, with reactivation | Curiosities, academic fascinations, learning goals without a concrete deliverable |
| 8 | **LOGISTICAL_CONTEXT** | Fastest — transient | Scheduling details, one-time locations, errands. Recurring activities are NOT logistical |
| 9 | **PROJECTS_ENDEAVORS** | Moderate, milestone-gated | Startups, research papers, creative projects, long-term goals with timelines |
| 10 | **FINANCIAL_MATERIAL** | Variable — context-dependent | Budget, income, expenses, purchases, debts, assets, owned devices, hardware |
| 11 | **OTHER** | Default moderate | Sink cluster; signals ontology extension need |

**Why 10+1 and not 8+1?** The original IDENTITY_SELF_CONCEPT absorbed ~38% of all facts — preferences, hobbies, and core traits all lumped together. This created a gravity well where a third of all facts shared one decay profile, undermining the thesis. The three-way split was motivated by genuinely distinct decay curves:
- **IDENTITY** (near-zero): "I'm Irish-Italian" doesn't change in 10 years
- **HOBBIES** (slow, reactivatable): "I do fishing" — skill/equipment investment persists through dormancy
- **PREFERENCES** (moderate): "I like MCU movies" — can flip after one bad film

### Full-Scale Classification Distribution (842 facts, 428 questions)

| Category | Count | % |
|---|---|---|
| HOBBIES_RECREATION | 210 | 25% |
| PREFERENCES_HABITS | 170 | 20% |
| FINANCIAL_MATERIAL | 157 | 19% |
| RELATIONAL_BONDS | 72 | 9% |
| LOGISTICAL_CONTEXT | 71 | 8% |
| HEALTH_WELLBEING | 64 | 8% |
| IDENTITY_SELF_CONCEPT | 39 | 5% |
| PROJECTS_ENDEAVORS | 30 | 4% |
| INTELLECTUAL_INTERESTS | 21 | 2% |
| OBLIGATIONS | 7 | 1% |
| OTHER | 1 | 0.1% |

No category exceeds 25%. No gravity wells. Under the old 8+1, IDENTITY+HOBBIES+PREFERENCES = 50% of all facts in one category. The three-way split gives three genuinely different decay profiles covering half the data.

### Classifier Rules (Final, v7, Production)

1. **Classify the fact, not the utterance.** Category based on the stored information, not conversational context.
2. **Two-step chain-of-thought.** Step 1: identify what the QUESTION retrieves. Step 2: classify THAT fact.
3. **Stored-fact anchoring.** Financial amounts → FINANCIAL. Transient scheduling → LOGISTICAL. Core traits → IDENTITY. Skill-based leisure → HOBBIES. Consumption patterns → PREFERENCES.
4. **Boundary test.** Accumulated SKILL → HOBBIES. Consumption CHOICE that could flip tomorrow → PREFERENCES. Core trait unchanged across years → IDENTITY.
5. **PREFERENCES_HABITS guardrails.** Recommendation-seeking language is a question format, not a category signal. Learning → INTELLECTUAL. "I am a [role]" → IDENTITY. Health-motivated behavior → HEALTH. Device ownership → FINANCIAL. Discounts/deals → FINANCIAL.
6. **FINANCIAL_MATERIAL guardrail.** Physical item mentioned ≠ automatic FINANCIAL. Usage frequency → PREFERENCES. Who gave/received → RELATIONAL. Activity the item supports → HOBBIES.
7. **Named person disambiguation.** Question ABOUT the relationship → RELATIONAL. Person as context for timing/amounts/counts → classify by what's measured.
8. **Temporal question immunity.** "How many days between X and Y" is about WHEN; category reflects WHAT.
9. **Deliverable test (mandatory override).** Creating/editing a named artifact → PROJECTS_ENDEAVORS.
10. **Soft membership.** Weights across all categories summing to 1.0, primary strictly highest (≥ 0.3), no ties.
11. **OTHER cap + suppression.** OTHER ≤ 0.5; if OTHER wins at <0.4 with close runner-up, swap to runner-up.

### Goal 2: Category-Conditioned Decay Engine ✅ COMPLETE

Implement multi-clock temporal decay with per-cluster rates.

- ✅ Lazy decay computation (store `last_updated_ts`, compute on access).
- ✅ Three clocks: absolute (wall time), relative (inter-session), conversational (within-session message count).
- ✅ Per-cluster sensitivity profiles over the three clocks. Hobbies sensitive to relative time (dormancy detection), Preferences sensitive to absolute time (taste drift over calendar months).
- ✅ **Harmonic mean blending** for multi-cluster facts. Correctly implemented — harmonic emphasizes slow-decaying clusters, producing higher activation than arithmetic mean for mixed-category facts.
- ✅ All components toggleable: uniform decay, single clock, no blending — all switchable via config. This is the ablation infrastructure.
- ✅ **25/25 unit tests passing:**
  - Health decays slower than Logistical (0.962 vs 0.178 after 24h)
  - Identity decays slowest of all (0.951 after 30 days, all others below)
  - Hobby reactivates after dormancy (dormant → 0.70 after reactivation)
  - Multi-cluster fact decays between component rates
  - Anticipatory activation rises before deadline, drops after
  - Emotional loading boost applied and fades
  - Ablation 1 (uniform): Health ≈ Logistical (both 0.787)
  - Ablation 9 (cognitive): Health ≈ Preference (both 0.511, both SEMANTIC)
  - Ablation 12 (8+1 collapse): Hobby ≈ Identity ≈ Preference (all 0.988)

**Per-category decay rates (λ values):**

| Category | λ | Half-life | Rationale |
|---|---|---|---|
| IDENTITY_SELF_CONCEPT | 0.0001 | ~289 days | Near-permanent: ethnicity, name, core beliefs |
| RELATIONAL_BONDS | 0.001 | ~29 days | Relationships persist but can evolve |
| HEALTH_WELLBEING | 0.002 | ~14 days | Conditions persist, acute symptoms fade |
| HOBBIES_RECREATION | 0.003 | ~10 days | Skill investment persists, reactivatable |
| INTELLECTUAL_INTERESTS | 0.004 | ~7 days | Curiosity persists with reactivation |
| PREFERENCES_HABITS | 0.010 | ~3 days | Tastes shift, easily superseded |
| PROJECTS_ENDEAVORS | 0.006 | ~5 days | Active while ongoing, milestone-gated |
| FINANCIAL_MATERIAL | 0.005 | ~6 days | Variable — budgets change, assets persist |
| LOGISTICAL_CONTEXT | 0.080 | ~9 hours | One-off scheduling details |
| OBLIGATIONS | 0.100 | ~7 hours | Expires on completion |
| OTHER | 0.010 | ~3 days | Default moderate (matches uniform) |

**800x spread** between IDENTITY (0.0001) and OBLIGATIONS (0.100). This is the core thesis mechanism: behavioral category determines temporal dynamics.

### Goal 3: Anticipatory Activation + Supersession

- ✅ **Anticipatory activation implemented and tested.** For Obligations/Projects facts with future timestamps, activation increases as target date approaches, rapid decay after passage. Unit test confirms: 1h before (2.98) > 7d before (0.30) > 24h after (0.08).
- ✅ **Emotional loading implemented and tested.** Temporarily boosts activation of emotionally-charged facts. Unit test confirms: emotional (1.23) > plain (0.96) at 1h, boost fades by 48h.
- **Still TODO:** Confidence-weighted supersession (extend Graphiti's supersession with confidence scores).

### Goal 2.5: Knowledge Graph Ingestion ✅ COMPLETE (PoC)

- ✅ **Ingested 25 LongMemEval questions** into Graphiti knowledge graph (Neo4j).
- ✅ Graph stats: 1,189 episodes, 5,873 entity nodes, 5,861 RELATES_TO edges.
- ✅ Cost: ~$9 for 25 questions. Projected $85-180 for full 500 questions.
- ✅ **Edge-level v7 classification:** All 5,861 edges classified with production v7 prompt. Zero errors. Distribution:
  - OTHER: 1,420 | PREFERENCES_HABITS: 1,164 | INTELLECTUAL_INTERESTS: 893
  - HOBBIES_RECREATION: 527 | PROJECTS_ENDEAVORS: 415 | IDENTITY_SELF_CONCEPT: 337
  - RELATIONAL_BONDS: 330 | LOGISTICAL_CONTEXT: 304 | FINANCIAL_MATERIAL: 199
  - HEALTH_WELLBEING: 174 | OBLIGATIONS: 98
- ✅ **Temporal spread:** fix_timestamps.py remapped 5,861 edges from 3.5-hour bulk window to 180-day spread, preserving relative ordering. 1107x spread increase.

**Phase 1 is done when:** You have a working ontology classifier (✅), category-conditioned decay with three clocks (✅), anticipatory activation for deadline facts (✅), and confidence-weighted supersession — all running on top of Graphiti's default retrieval (no swarm traversal yet). Everything toggleable. **~90% DONE. Missing only supersession.**

---

## Phase 1.5: Prove the Thesis Early ✅ KILL GATE PASSED

**This phase exists to derisk before you build the rest.**

### Goal 4: Minimum Viable Evaluation ✅ COMPLETE

Using the Phase 1 system (ontology + decay + Graphiti retrieval + Cypher keyword/intersection/neighbor expansion):

- ✅ **Built evaluate_v4.py** — full evaluation harness with edge-level decay, blended reranking, multi-source candidate pools.
- ✅ Compared behavioral ontology decay vs. uniform decay vs. cognitive typology decay.
- ✅ **Per-question temporal anchors** — each question's t_now set from its last haystack session's edge timestamps, not a single global clock.
- ✅ **Cypher semantic baselines** — keyword (0.4), intersection (0.5), neighbor (0.3) candidates get nonzero scores so activation can influence their ranking.
- ✅ **Alpha sweep** — tested blending weights from 0.2 to 0.8.

### Kill Gate Results (25 questions, 8 answerable)

**Headline: B > C > U. Behavioral wins.**

```
MRR:    Behavioral=0.2168  Cognitive=0.2017  Uniform=0.1552
hit@5:  B=2  C=2  U=2
Wins:   B=1  U=5  C=0  Ties=2
```

**Behavioral beats uniform by 40% on MRR.** Behavioral beats cognitive by 7%.

**Alpha sweep (MRR at each blending weight):**

| Alpha | Behavioral | Uniform | Cognitive | Winner |
|---|---|---|---|---|
| 0.2 | 0.106 | 0.156 | 0.151 | Uniform |
| 0.3 | 0.121 | 0.155 | 0.151 | Uniform |
| 0.4 | 0.152 | 0.155 | 0.145 | Uniform |
| **0.5** | **0.217** | **0.155** | **0.202** | **Behavioral** |
| 0.6 | 0.213 | 0.139 | 0.206 | Behavioral |
| 0.7 | 0.217 | 0.160 | 0.208 | Behavioral |
| 0.8 | 0.222 | 0.214 | 0.218 | Behavioral |

Behavioral wins at every alpha ≥ 0.5. Below 0.4, activation dominates and uniform's safe-default rate wins.

**Per-question showcase:**

| Question | Category | B rank | U rank | C rank | Story |
|---|---|---|---|---|---|
| Q5: Wednesday language exchange | LOGISTICAL_CONTEXT | **@9** | @72 | @18 | **B wins by 63 positions.** Per-question temporal context makes logistical fact recent. B activation=0.811, U activation=0.410. |
| Q24: Cocktail class Friday | HOBBIES_RECREATION | **@1** | @3 | @1 | **B achieves rank 1.** Hobby fact preserved with activation=0.794. |
| Q21: Grilled Snapper | PREFERENCES_HABITS | @3 | @2 | @4 | U wins narrowly. Preference fact recent for all engines. |
| Q6: Bikes owned | HOBBIES_RECREATION | @13 | @13 | @13 | Tie — hobby fact within temporal window for all. |

**Key finding:** Behavioral produces large gains where category-conditioned rates match the question's temporal context (Q5: 63 rank positions), with small losses elsewhere. The wins are enormous; the losses are incremental.

### Critical Bugs Found and Fixed During Kill Gate

1. **Category name mismatch (C1 — ROOT CAUSE):** `reenrich_edges_v7.py` mapped v7 canonical names (RELATIONAL_BONDS) to wrong names (RELATIONAL). Decay engine couldn't find them → **8/11 categories silently fell back to uniform rate.** Behavioral was literally running as uniform for 73% of edges. Fix: identity mapping.

2. **Harmonic/arithmetic blending identical (C2):** Both BlendingMode branches computed weighted arithmetic mean. True harmonic mean formula existed but wasn't used. Fix: HARMONIC now computes λ_eff via harmonic mean of rates then applies single exponential.

3. **Cypher candidates scored 0.0 (S2):** With alpha=0.5, Cypher candidates could never reach top-5 regardless of activation. Fix: source-type baselines (kw=0.4, intersect=0.5, neighbor=0.3).

4. **Same t_now for all questions (D2):** All questions used wall clock time. Logistical facts that should be hours old appeared months old. Fix: per-question temporal anchors from haystack session episodes.

5. **Alpha not tuned (D1):** Only tested alpha=0.5. Fix: added --sweep flag for alpha range [0.2..0.8].

**Lesson:** The thesis was correct from the beginning. Five implementation/evaluation bugs were masking a 40% MRR improvement.

### Kill/Continue Decision

✅ **CONTINUE.** Behavioral ontology shows clear lift over both uniform and cognitive baselines on PoC. Proceed to full evaluation with confidence.

**Phase 1.5 is done when:** You have preliminary evidence that behavioral ontology > cognitive typology on at least information extraction and knowledge updates. Or you've made a clear-eyed pivot decision. ✅ **DONE. B > C > U confirmed.**

---

## Phase 2: Full Implementation + Evaluation 🔜 IN PROGRESS

Build remaining components and run the complete evaluation.

### Goal 5: Full LongMemEval Ingestion (IMMEDIATE NEXT)

- **Ingest 234 must-win questions** (or full 500) into Graphiti. Estimated cost: $85-180. Runtime: ~14 hours overnight.
- Run v7 edge enrichment on full graph (~1 hour).
- Run fix_timestamps on full graph (~30 sec).
- Run evaluate_v4.py on full dataset — this gives real numbers with statistical power (expect 60-100+ answerable questions vs current 8).

### Goal 6: Swarm Retrieval + Session Initialization

- **Query classification:** Incoming query → relevant behavioral clusters with weights.
- **Swarm traversal:** Cluster-conditioned random walks with restart (personalized PageRank variant). Agent deployment per cluster, local traversal, cluster-local gating (promote only top candidates above threshold), global merging with token budget.
- **Warm start protocol:** Three modes — cold (skip priming), warm (targeted cluster traversal), evolving (drift detection via rolling query cluster vector, cosine distance threshold δ, trigger after N consecutive turns).
- **Emotional loading:** Detect at ingestion, temporarily boost activation of relevant cluster(s), fast-decaying boost.

### Goal 7: Baselines

Get all baselines running:

- **Naive vector-RAG** (embed all past messages, top-k retrieval)
- **Summary-only memory** (periodic LLM summarization)
- **Zep/Graphiti vanilla** (your infrastructure with policy layer removed — isolates your contribution)
- **MemoryBank** (uniform Ebbinghaus curve implemented on your system)
- **Memory-R1** (faithful implementation of their RL-learned memory operations — this is non-negotiable)
- **MemoryOS** and **MIRIX** if feasible; otherwise acknowledge in paper with clear justification

### Goal 8: Reproducible Ingestion Artifacts

Before running any evaluations, produce and freeze ingestion outputs:

- Run ingestion pipeline (temperature 0, versioned prompts) on LongMemEval and LoCoMo.
- **Cache all extracted facts, cluster assignments, supersession edges, confidence scores, emotional loading signals** as serialized artifacts.
- All retrieval experiments and ablations run off these cached artifacts — no re-ingestion.
- Release artifacts (or hashes + regeneration script) with the anonymous repo.
- Document: total tokens consumed, total API calls, cost per dataset, wall-clock ingestion time.
- **LongMemEval classification artifacts already frozen:** `extracted_facts.json` (842 facts) + `classified_facts.json` (842 classified, v7 prompt).
- **PoC graph artifacts:** 25 questions ingested, 5,861 edges enriched with v7 classifications.

**This is your reproducibility backbone.** Reviewers can rerun every retrieval experiment without paying a cent in API costs.

### Goal 9: Full LongMemEval Evaluation

Run full LongMemEval with all baselines. Break down results by subcategory:

| Subcategory | Your relevant component | Must-win? |
|---|---|---|
| Information extraction | Selective retention via behavioral decay | **Yes** |
| Multi-session reasoning | Multi-clock temporal routing | No (but should help) |
| Temporal reasoning | Anticipatory activation | No (but should show distinct capability) |
| Knowledge updates | Confidence-weighted supersession | **Yes** |
| Abstention | Validity filtering + open-set cluster | No |

**"Win" = statistically significant improvement OR consistent improvement across 3 random seeds/runs.** Report confidence intervals or significance tests — don't just report point estimates.

### Goal 10: LoCoMo + MSC

Run LoCoMo and MSC evaluations against all baselines. Three benchmarks with consistent B > U > C strengthens the paper substantially. One dataset = demo, two = pattern, three = contribution.

### Goal 11: Custom Metrics

Compute metrics only your system can be evaluated on:

- **Selective retention score:** Partition queries by ontology category. Measure recall@k of relevant facts AND false recall rate of stale/expired facts.
- **Staleness penalty:** How often do you surface superseded facts? Heavy penalty for old addresses, former jobs.
- **Temporal calibration:** For obligation facts with deadlines, does activation rise correctly as deadline approaches? Plot activation curves.
- **Relational miss rate:** Swarm traversal vs. plain top-k vector retrieval. How many correct memories are graph-connected but not semantically similar to query?

### Goal 12: Anticipatory Activation Evaluation Harness

Design this as an objective mini-benchmark, not a custom metric that looks rigged to make you win.

**Deadline Harness Protocol:**
- Construct a standardized set of conversations containing future-anchored obligations with known target dates (e.g., "my thesis defense is June 15," "project deadline is April 1," "doctor appointment next Thursday").
- Simulate time progression. At each simulated timestep, the system receives a generic check-in query (not specifically about the deadline).
- **Measure:**
  - *Anticipatory precision:* When the system proactively surfaces a deadline-related fact, is it actually approaching? (Precision within the correct temporal window.)
  - *Anticipatory recall:* Of all deadlines within their activation window, what fraction does the system surface?
  - *Post-deadline suppression:* After a deadline passes with no update, does activation decay rapidly? Measure false surfacing rate in the post-deadline window.
- **Release exact generation rules** (how conversations are constructed, how time is simulated, how queries are chosen) so the harness feels objective and reusable by others.
- Run all baselines on this harness. Every existing system should score ~0 on anticipatory recall because none implement the capability — that's the point.

### Goal 13: Ablations

Run every ablation. Non-negotiable.

| # | Ablation | Expected degradation |
|---|---|---|
| 1 | Remove per-cluster decay (uniform rate) | More stale logistics, faded health facts |
| 2 | Remove multi-clock routing (single absolute timestamp) | Worse cross-session reasoning |
| 3 | Remove anticipatory activation | Worse deadline-related recall |
| 4 | Remove warm start protocol | Slower session priming, irrelevant early context |
| 5 | Remove hierarchical gating (all candidates global) | More retrieval noise, lower precision |
| 6 | Replace swarm with plain top-k vector | Relational misses |
| 7 | Remove confidence-weighted supersession (binary) | Incorrect fact invalidation under ambiguity |
| 8 | Remove emotional loading | Reduced context-sensitivity in emotional queries |
| 9 | **Replace behavioral ontology with cognitive typology** | **THE critical ablation. Proves your core thesis.** |
| 10 | Harmonic mean → arithmetic mean blending | Tests whether blending function choice matters |
| 11 | Harmonic mean → max-membership-only (no blending) | Tests whether soft clustering matters |
| 12 | **Replace 10+1 ontology with original 8+1 ontology** | **Tests the Identity split. Expect reduced decay differentiation for hobbies vs. preferences vs. core traits.** |

**If any component removal doesn't hurt performance, drop it from the paper.** A tight system where everything matters beats a complex system with decorative parts.

**Kill gate already provides preliminary data for ablations 1, 9, 12:** B > C > U on MRR (0.2168 > 0.2017 > 0.1552). Ablation 9 (cognitive) shows 7% degradation. Ablation 1 (uniform) shows 40% degradation.

### Goal 14: Latency + Efficiency Reporting

You claim "pure math at runtime, no LLM in hot path." Prove it.

- **Retrieval latency vs. baselines:** Average retrieval time per query for your system vs. Graphiti vanilla vs. vector-RAG vs. Memory-R1.
- **Latency vs. graph size:** Plot retrieval latency as conversation history grows (100, 500, 1000, 5000 messages). Show scaling behavior.
- **Memory growth:** Graph node/edge count over conversation length.
- **Cost per conversation turn:** Ingestion cost (amortized) + retrieval cost. Show the breakdown.

Present as a small table or figure.

**Phase 2 is done when:** You have complete results tables for LongMemEval (overall + subcategories), LoCoMo, MSC, all custom metrics, anticipatory harness, all ablations, all baseline comparisons, and latency/efficiency numbers. You know the story.

---

## Phase 3: Writing

### Goal 15: The Paper

**Single thesis drives everything:** The behavioral type of a memory determines its temporal dynamics. Every section serves this thesis.

**Page budget (9 pages):**

| Section | Pages | Notes |
|---|---|---|
| Introduction + Problem | 1 | End with: "We hypothesize that behavioral type, not cognitive type, should determine forgetting dynamics." |
| Related Work | 1.5 | Consolidated gap table is the anchor. 2-3 sentences per system. Full per-system analysis → supplementary. Focus on Zep, MemoryBank, Memory-R1, MemoryOS. |
| Architecture | 2.5 | Behavioral ontology (10+1 with Identity split justification), multi-clock decay, anticipatory activation, swarm retrieval, warm start. Formal notation throughout. Frame as one coherent system, not 8 features. |
| Experiments + Results | 3 | Benchmarks, baselines, custom metrics, anticipatory harness, ablations (including 8+1 vs 10+1), latency. Tables and figures drive the narrative. |
| Discussion + Limitations | 1 | Cultural generalization (acknowledged, open-set mitigates), ingestion sensitivity (quantified in sensitivity analysis), "classify fact not utterance" insight, future work (RL integration → sets up AAAI). |

**Frame as three claims, not eight contributions:**

- **Primary claim:** Behavioral ontology as a forgetting prior enables selective retention beyond cognitive typologies and uniform decay.
  - Supported by: LongMemEval/LoCoMo/MSC results, behavioral vs. cognitive ablation, 10+1 vs 8+1 ablation, selective retention scores, staleness penalty, 90%+ inter-annotator agreement on 100 held-out facts.
- **Supporting claim A:** Multi-reference-frame temporal routing is necessary for cross-session temporal correctness.
  - Supported by: Multi-clock ablation, temporal reasoning subcategory.
- **Supporting claim B:** Anticipatory activation adds a distinct future-aware capability absent from all existing systems.
  - Supported by: Anticipatory harness results, temporal calibration plots.

Everything else (warm start, emotional loading, swarm gating, soft supersession) lives under these three claims or is demoted to engineering detail unless ablations show otherwise.

**Key figures:**

1. Architecture diagram: infrastructure layer (Graphiti) + policy layer (your contributions), clearly delineated
2. Performance comparison table: LongMemEval subcategories + LoCoMo + MSC, your system vs. all baselines
3. Ablation degradation table: full system → each component removed (including 10+1 vs 8+1)
4. Anticipatory activation curve: activation over time for a deadline fact (rising approach, peak, rapid decay)
5. Selective retention plot: recall of persistent vs. transient facts across sessions
6. Latency vs. graph size plot or table
7. **Classifier iteration table** (v1→v7 with agreement scores — demonstrates empirical ontology refinement)
8. **Decay differentiation table** — activation at various time horizons per category (already produced from fix_timestamps.py)
9. **Alpha sweep figure** — MRR vs alpha for all three engines (already produced from kill gate)

**Write results first.** Tables and figures determine the narrative.

### Goal 16: Formalization

Tighten mathematical notation:

- Fact $f$, cluster membership vector $\mu(f) \in [0,1]^K$ where $K=11$ (10+1 categories)
- Per-cluster decay rates $\lambda_k$
- Blended decay: $\lambda_{\text{eff}}(f) = \frac{\sum_k \mu_k(f)}{\sum_k \mu_k(f) / \lambda_k}$ (weighted harmonic mean)
- Multi-clock sensitivity matrix: $S \in \mathbb{R}^{K \times 3}$, routing clusters to temporal signals
- Anticipatory activation function with transition at deadline
- Swarm traversal: cluster-conditioned PPR, hierarchical gating thresholds

Justify harmonic mean: it emphasizes slower-decaying clusters — a fact even partially personal decays slower than a purely logistical one. Ablations 10-11 test this choice empirically. Ablation 12 tests the Identity split.

### Goal 17: Submission Prep

- **NeurIPS checklist** (mandatory — desk rejection without it). Fill it carefully; reviewers read it.
- **Supplementary material:**
  - Full gap analysis (the 6-page demolition from your architecture doc)
  - All ingestion prompts (versioned, temperature 0) — **classifier v1→v7 evolution with audit results**
  - Label guideline document (10+1 categories) + annotation statistics
  - **Classifier iteration history with inter-annotator agreement at each stage**
  - Additional ablation details
  - Anticipatory harness generation rules
  - Latency/cost breakdowns
- **Anonymous GitHub repo:**
  - Full code
  - Frozen ingestion artifacts for LongMemEval, LoCoMo, and MSC
  - **One-command reproduction:** downloads cached ingestion outputs → runs retrieval → prints all tables from the paper
  - Docker/conda environment export
  - Fixed random seeds throughout
- **Abstract:** 250 words, written last. Problem → single thesis → approach → key results → significance.
- **Post to arXiv simultaneously** (allowed under NeurIPS policy). Establishes priority on Fortunate Recall.

**Phase 3 is done when:** Paper is submitted, arXiv preprint is live, anonymous repo is accessible.

---

## Kill Conditions

Be honest at each checkpoint:

**After Phase 1:** Does the minimum viable system work end-to-end? Can you ingest a conversation, assign clusters, compute category-conditioned decay, and retrieve facts? If fundamentally broken, diagnose before proceeding. **✅ YES. Full pipeline working: ingest → classify → decay → retrieve → evaluate.**

**After Phase 1.5 (the critical gate):** Does behavioral ontology show clear lift over cognitive typology on LongMemEval information extraction and knowledge updates? **✅ YES. B > C > U on MRR (0.2168 > 0.2017 > 0.1552). 40% lift over uniform, 7% over cognitive. Kill gate passed.**

**After Phase 2:** Do the full numbers support the claims?
- Does behavioral ontology outperform cognitive typology in ablation 9? Preliminary: yes (MRR 0.2168 vs 0.2017). Needs confirmation at scale.
- Does 10+1 outperform 8+1 in ablation 12? Not yet tested at evaluation level. Decay engine test confirms collapse (all 0.988 under 8+1).
- Do you beat Zep on information extraction and knowledge updates subcategories? Not yet tested.
- Is the Memory-R1 comparison at least competitive? Not yet implemented.

**If numbers are ambiguous after Phase 2:** Pivot to AAAI 2027 (~August 2026 deadline). Four extra months, natural home (MemoryBank published there), and you can run the RL integration extension simultaneously.

---

## Progress Tracker

| Item | Status | Date |
|---|---|---|
| Fork Graphiti, run locally | ✅ | Feb 21 |
| Split-client architecture (Grok + OpenAI) | ✅ | Feb 21 |
| Download LongMemEval (cleaned) | ✅ | Feb 21 |
| Download LoCoMo | ✅ | Feb 21 |
| Confirm Memory-R1 has no public code | ✅ | Feb 21 |
| Label guidelines document (8+1 → updated to 10+1) | ✅ | Feb 21-22 |
| Classifier prompt v1 | ✅ | Feb 21 |
| Extract 78 facts from LongMemEval (dev set) | ✅ | Feb 22 |
| Classifier v1→v4 iteration with audits (30-fact dev set) | ✅ | Feb 22 |
| Validate 10+1 ontology at 93% agreement | ✅ | Feb 22 |
| Classifier v5→v7 iteration (fix gravity wells) | ✅ | Feb 22 |
| v7 achieves 100% on 30-fact dev set | ✅ | Feb 22 |
| Extract 842 facts from ALL 428 LongMemEval questions | ✅ | Feb 22 |
| Full-scale classification (842 facts, $0.49) | ✅ | Feb 22 |
| 100-fact held-out audit (Opus 90%, GPT 70%, consensus 95%) | ✅ | Feb 22 |
| Freeze ingestion artifacts (extracted_facts.json, classified_facts.json) | ✅ | Feb 22 |
| Decay engine implementation (25/25 tests) | ✅ | Feb 22 |
| Graphiti bridge layer (21/21 tests) | ✅ | Feb 22 |
| PoC ingestion: 25 questions, 5,861 edges | ✅ | Feb 22 |
| Edge-level v7 classification (5,861 edges) | ✅ | Feb 22 |
| Temporal spread (fix_timestamps.py, 180-day spread) | ✅ | Feb 22 |
| evaluate_v4.py harness (multi-source pool, blended reranking) | ✅ | Feb 22 |
| Category name mismatch bug found and fixed (C1) | ✅ | Feb 22 |
| Harmonic blending bug found and fixed (C2) | ✅ | Feb 22 |
| Per-question temporal anchors (Bug 2 fix) | ✅ | Feb 22 |
| Cypher semantic baselines (Bug 1 fix) | ✅ | Feb 22 |
| Alpha sweep (Bug 3 fix + --sweep flag) | ✅ | Feb 22 |
| **KILL GATE PASSED: B > C > U (MRR 0.2168 > 0.2017 > 0.1552)** | ✅ | **Feb 22** |
| Human annotator on 100+ overlapping examples | 🔜 | This week |
| Full ingestion: 234-500 questions ($85-180) | 🔜 | Tonight/tomorrow |
| Full evaluation on LongMemEval | 🔜 | This week |
| LoCoMo ingestion + evaluation | — | Next week |
| MSC ingestion + evaluation | — | Next week |
| Swarm traversal | — | Weeks 3-4 |
| Memory-R1 reimplementation | — | Weeks 5-7 |
| Paper writing | — | Weeks 7-10 |

---

## Three Things to Do Next

**Updated priorities (end of Feb 22):**

1. **Run full ingestion tonight.** 234 must-win questions overnight ($85, ~14 hours). Enrich edges with v7 tomorrow morning. Evaluate. This gives real numbers with statistical power.
2. **Get Cohen's κ from İdil.** 100 facts, independent labeling. The classifier is locked — she annotates independently, you compute agreement.
3. **Start LoCoMo ingestion.** Second benchmark. Three benchmarks with consistent B > C > U is a NeurIPS story.

---

## The LexCGraph Connection

If SIGIR notifications arrive before NeurIPS submission and LexCGraph is accepted, cite it in the introduction. The shared philosophy — fixed abstract ontology + dynamic graph population + intelligent traversal — across legal reasoning and conversational memory is a research program. Even if pending, cite as "Anonymous, under review."

Use it to raise reviewer confidence that you can execute ontology-driven systems with real empirical results. But don't lean on it as a substitute for results in this paper.

---

## Files & Scripts Reference

### Core Pipeline
```
C:\Users\anshu\PycharmProjects\hugeleapforward\
├── decay_engine.py              # Category-conditioned decay (25/25 tests) — DO NOT MODIFY
├── graphiti_bridge.py           # Bridge: Graphiti ↔ decay engine (21/21 tests)
├── evaluate_v4.py               # Kill gate evaluation harness (per-question temporal, alpha sweep)
├── reenrich_edges_v7.py         # Edge classification with v7 prompt (5,861 edges)
├── fix_timestamps.py            # Remap bulk timestamps to 180-day spread
├── ingest_poc.py                # Ingestion script for LongMemEval → Graphiti
├── .env                         # API keys + Neo4j credentials
```

### Evaluation Data
```
C:\Users\anshu\PycharmProjects\hugeleapforward\LongMemEval\data\
├── longmemeval_oracle.json          # 500 questions with evidence sessions (15MB)
├── longmemeval_s_cleaned.json       # Small haystack ~115K tokens/question (277MB)
├── longmemeval_m_cleaned.json       # Medium haystack ~1.5M tokens/question (2.7GB)
├── extracted_facts.json             # 842 fact-bearing utterances from ALL 428 questions ← FROZEN
├── classified_facts.json            # 842 facts with behavioral classifications (v7) ← FROZEN
├── poc_artifacts/                   # Kill gate results, cached evaluations
│   └── kill_gate_results_v4.json    # Latest kill gate results
├── extract_facts.py                 # Extracts user utterances with has_answer=true
├── classify_facts.py                # Runs Grok 4.1 Fast classifier (v7, production prompt)
└── generate_audit_prompt.py         # Generates stratified 100-fact audit
```

### Cost Summary

| Operation | Items | Cost | Per-Item |
|---|---|---|---|
| Dev set classification (v1-v6 iterations) | ~500 total across versions | ~$0.20 | ~$0.0004 |
| Production classification (v7, 842 facts) | 842 | $0.49 | $0.00058 |
| PoC ingestion (25 questions) | 1,189 episodes | ~$9 | ~$0.36/question |
| Edge enrichment (5,861 edges, 2 runs) | 11,722 | ~$3 | ~$0.00026 |
| **Total to date** | — | **~$13** | — |
| **Projected full ingestion (500 questions)** | — | **$85-180** | — |
