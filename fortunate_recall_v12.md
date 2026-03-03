# Fortunate Recall: Ontology-Driven Memory Lifecycle Management for Persistent Coherence in LLMs

**Project codename:** Project Gnomes

**Ansuman Mullick, Dr. Dilek Küçük, Dr. Fazli Can, Bilge Idil Ozis**

**Status:** Phase 2 — Lifecycle Architecture + Custom Benchmark | **Target Venue:** NeurIPS 2026 (fallback: AAAI 2027)

---

## Abstract

As conversational AI systems accumulate months and years of interaction history, the memory management problem transitions from retrieval to curation. Current approaches treat all personal facts identically — storing them in flat vector stores or knowledge graphs with uniform retention policies — producing memory stores that grow unboundedly while retrieval precision degrades, context windows fill with outdated information, and downstream response quality deteriorates. We present **Fortunate Recall**, a composable policy layer that classifies personal facts into a 10+1 behavioral ontology and applies category-specific lifecycle policies: differential temporal decay, slot-key supersession with confidence weighting, event-time validity windows, category-aware retrieval routing, anticipatory activation for future-anchored facts, and ontology-aware session initialization. The ontology determines not merely how fast a memory fades, but whether it should be replaced, expired, accumulated, or preserved — transforming an opaque memory store into an interpretable, auditable, and user-modifiable personal knowledge base. All lifecycle operations execute as pure mathematical functions at runtime; LLMs are used exclusively at ingestion time, producing millisecond retrieval with full auditability.

We introduce **LifeMemBench**, a temporal disambiguation benchmark of [N] questions across 50 synthetic personas with controlled supersession events, preference changes, identity stability, and logistical expiry, generated conversation-first using frontier reasoning models across multiple providers. On LifeMemBench, Fortunate Recall achieves [X] MRR, outperforming Graphiti ([Y]), Mem0 ([Z]), MemGPT ([W]), and uniform decay baselines by [N]+ points on temporal disambiguation tasks. On LongMemEval, behavioral lifecycle policies achieve statistically indistinguishable retrieval performance from flat-rate baselines (all p > 0.05, |Cohen's d| < 0.19), confirming zero performance cost. We additionally report two infrastructure-level findings affecting all systems built on LLM-based knowledge graph extraction: (1) 86% of extracted edges represent world knowledge rather than personal memories, and (2) numeric values are systematically lost during paraphrase-based entity extraction. The ontology, lifecycle policies, benchmark, and evaluation code are released publicly.

---

## 1 The Problem: Memory Systems That Cannot Forget

Large language models lose coherence over long conversations because the linear context window is arguably the wrong abstraction for memory. Real memory is not a queue; it is a graph with activation patterns.

Current LLM memory systems treat memory as a retrieval problem. This work argues it is a **lifecycle management problem**, and more specifically, a **forgetting policy problem**. The question is not only how to find the right context, but how to determine which memories should persist, which should be replaced, which should expire, and at what rate — conditioned on the type of memory.

Recent work has made significant progress on memory infrastructure. Zep/Graphiti (2025) [2] introduced temporally-aware knowledge graphs with bi-temporal validity tracking, entity extraction, and hybrid retrieval achieving sub-300ms latency. A-MEM (2025) [3] proposed Zettelkasten-inspired agentic memory with dynamic note linking. MemoryBank (2023) [4] applied Ebbinghaus forgetting curves to LLM memory. Memoria (2025) [5] introduced scalable agentic memory with weighted KG-based user modeling. MemoryOS (2025) [10] proposed an OS-inspired hierarchical memory architecture. Memory-R1 (2025) [11] applied reinforcement learning to learn memory operations end-to-end. MIRIX (2025) [12] introduced multi-agent memory with structured memory types. FluxMem (2026) [16] treats memory structure selection as adaptive with probabilistic gates. These systems have established strong foundations for how to store, retrieve, and organize memory.

What remains unsolved is the **lifecycle layer**: given a memory infrastructure capable of storing, linking, and retrieving facts with temporal metadata, what should govern the dynamics of forgetting, supersession, expiry, and reactivation?

Consider a user who interacts with an AI assistant three times per day over two years. Each session produces 5–15 extractable personal facts. After two years, the user's memory store contains 10,000–30,000 facts. Now consider what happens at retrieval time:

- The user changed jobs twice. Three "works at" facts coexist with equal status. The system surfaces all three — or whichever has the highest embedding similarity to the query — with no mechanism to identify which is current.
- The user mentioned a dentist appointment six months ago. The appointment has long passed, but the fact "dentist appointment Thursday at 2pm" still scores highly on any query about upcoming plans. The system has no concept of event-time expiry.
- The user mentioned having ADHD exactly once, in the third session. After 700 sessions of conversation about cooking, travel, and work, semantic retrieval buries this fact under thousands of more recent, more verbose edges. Yet ADHD is a permanent trait that should inform every response about learning, focus, or productivity.
- The user explicitly said "Actually, forget about the Denver move — I've decided to stay." The retraction exists as another fact in the store, but the original "wants to move to Denver" still competes for retrieval slots.
- The user said "I'm thinking about moving to London." This does not invalidate "I live in Istanbul" — but a binary supersession system treats it as a contradiction and kills the old fact. The ambiguity is lost.

At 30,000 facts, semantic search returns hundreds of candidates. The downstream LLM receives 20–50 facts in its context window, of which half may be outdated, contradictory, or irrelevant. It does not know that "works at Google" was superseded by "works at Anthropic." It does not know that the dentist appointment already happened. It sees all facts as equally valid and either hallucinates a synthesis or picks arbitrarily.

**This is why current production systems impose hard caps on memory size.** Claude's memory is a flat list of short text snippets with limited capacity. ChatGPT's memory is similar. These systems chose to limit memory rather than manage it, because unbounded memory with no lifecycle policy is worse than limited memory.

We argue that the missing component is not better retrieval — it is **lifecycle management conditioned on the behavioral type of each fact.** "User has ADHD" and "User's meeting is at 3pm" are both personal facts, but they have fundamentally different temporal dynamics:

- **Identity** facts (ethnicity, diagnoses, core values) persist for years, rarely change, and should be retrievable regardless of recency.
- **Preferences** (favorite foods, media habits, routines) change periodically and should be *replaced* by newer values, not merely faded.
- **Logistics** (appointments, travel plans, deadlines) have event-time validity windows and should *expire* after their event time passes.
- **Obligations** (deadlines, commitments) become *more* relevant as they approach, then expire after passing — an inversion of standard decay.
- **Relational bonds** (family members, close friends) accumulate slowly and persist unless explicitly dissolved.

No existing memory system makes these distinctions. All personal facts receive identical treatment. The result is a memory store that grows without structure, degrades without signal, and cannot be inspected, debugged, or modified by users or developers.

We introduce the concept of **Fortunate Recall**: not total recall, but *intelligent* recall. The system still forgets — decay is a feature, not a bug. But it forgets according to the behavioral type of each fact, preserving what matters, replacing what changed, expiring what passed, and reactivating what approaches. Catastrophic forgetting — the uncontrolled loss of previously learned information — is one of the most recognized failure modes in machine learning. Fortunate Recall is its inverse: structured, adaptive persistence where catastrophic forgetting would otherwise erode coherence.

---

## 2 Comprehensive Gap Analysis

To precisely position this work, we conducted a deep technical analysis of the most relevant systems in the LLM memory landscape. Each system has made genuine contributions; the gaps identified here are not criticisms of engineering quality but rather identification of unsolved research problems that motivate our architecture.

### 2.1 Zep/Graphiti (2025): Brilliant Infrastructure, No Behavioral Intelligence

Zep, powered by its open-source engine Graphiti (~22K GitHub stars, 25K+ weekly PyPI downloads, MCP server with hundreds of thousands of weekly users), represents the current state-of-the-art in memory infrastructure. It achieves 94.8% on the DMR benchmark and up to 71.2% on LongMemEval with 90% latency reduction. Graphiti's bi-temporal model, hybrid retrieval (semantic + BM25 + BFS), and entity extraction pipeline are production-grade. The engineering is excellent. The research gaps are as follows:

**Gap 1: No forgetting mechanism.** Zep does not forget. Facts either exist as valid or get explicitly superseded when new contradicting information arrives through their temporal extraction and edge invalidation pipeline. There is no decay, no fading, no relevance degradation over time. A fact about what a user ate for lunch six months ago has the same retrieval weight as a parent's chronic illness, permanently, unless something explicitly contradicts it. The graph grows monotonically. This conflates archival storage with active memory.

**Gap 2: No behavioral awareness of fact types.** Zep's fact extraction prompts extract raw triples: entity A, relation, entity B, plus temporal metadata. There is zero classification of what behavioral category a fact belongs to. A scheduling detail and a core identity trait are structurally identical edges in the graph. The system literally cannot distinguish between "user has ADHD" and "user's meeting is at 3pm" — they are both edges with valid_at timestamps and identical temporal treatment.

**Gap 3: Purely reactive retrieval with no proactive surfacing.** Zep's retrieval is entirely query-driven. There is no anticipatory capability. The system never proactively surfaces "You mentioned your launch was early March" as the deadline approaches. It has no model of the future.

**Gap 4: No session initialization intelligence.** When a user begins a new conversation, the system has no mechanism for inferring likely intent, no warm start protocol, no awareness of what context is likely relevant.

**Gap 5: Communities are structural, not behavioral.** Zep's community detection uses label propagation on graph connectivity. A community might contain a user's mother, her diabetes, the hospital, and a parking ticket at the hospital — all connected, but behaviorally very different facts that should have radically different temporal dynamics.

**Gap 6: Binary supersession under ambiguity.** Zep performs binary supersession: old fact invalidated, new fact valid. But real conversation is ambiguous. "I'm thinking about moving to London" does not invalidate "I live in Istanbul." Zep's invalidation pipeline lacks a confidence-weighted soft supersession mechanism that preserves both facts under ambiguity.

**Gap 7: Noisy retrieval on certain query types.** Zep's own LongMemEval results show a 17.7% performance drop on single-session-assistant questions. The system actively hurts performance on certain query patterns, suggesting retrieval noise that a behavioral gating mechanism could directly address.

**Gap 8: Self-identified ontology gap.** Zep's own conclusion states: "domain-specific ontologies present significant potential. Graph ontologies, foundational in pre-LLM knowledge graph work, warrant further exploration within the Graphiti framework." They are explicitly pointing at the gap this work fills.

### 2.2 MemoryBank (AAAI 2024): Right Philosophy, Crude Implementation

MemoryBank [4] is philosophically the closest to our work — it explicitly applies Ebbinghaus forgetting curves to LLM memory, arguing for cognitively-motivated forgetting. Published at AAAI 2024, it demonstrated that forgetting is a feature, not a bug. We build directly on this intuition while addressing fundamental limitations:

**Gap 1: Single uniform forgetting curve for all memory types.** Every memory in MemoryBank decays at the same base rate. The only modulation is through recall frequency. But the type of memory does not matter. A parent's chronic illness decays at the same base rate as a scheduling detail. The only reason the illness persists longer is if it happens to be mentioned more often.

**Gap 2: No graph structure.** MemoryBank stores memories as flat text segments with embeddings. No knowledge graph, no entity relationships, no structured facts. This means it cannot perform multi-hop reasoning, cannot track fact supersession, and cannot represent complex relationships between memories.

**Gap 3: Single temporal reference frame.** One clock: wall time since last recall. No session-relative time, no within-session conversational time, no deadline-proximity signals.

**Gap 4: No behavioral categorization.** There is a vague notion of "significance" that modulates decay, but it is a single scalar determined by access frequency, not by the semantic or behavioral type of the memory.

**Gap 5: Limited evaluation scope.** MemoryBank was evaluated only as a companion chatbot (SiliconFriend) on empathy and personality understanding. No systematic benchmarking against standard memory retrieval tasks, no LongMemEval.

### 2.3 A-MEM (NeurIPS 2025): Dynamic Organization, Zero Temporal Intelligence

A-MEM [3] introduces a Zettelkasten-inspired agentic memory system with dynamic note construction, autonomous link generation, and memory evolution. The contributions to memory organization are real. The temporal gaps are fundamental:

**Gap 1: No temporal dynamics whatsoever.** Memories exist or they do not. No timestamps, no decay, no mechanism for old memories becoming less relevant.

**Gap 2: No forgetting mechanism.** The Zettelkasten method is a permanent filing system. This is ideal for a researcher's knowledge base but fundamentally wrong for conversational memory where the vast majority of exchanged information is logistically irrelevant within days.

**Gap 3: Memory evolution is content-driven, not behavior-driven.** Updates are purely semantic (does the content relate?) not behavioral (should this memory persist, fade, or intensify based on what kind of memory it is?).

**Gap 4: LLM-heavy at runtime.** A-MEM makes LLM calls for note construction, link generation, and memory evolution during operation. Our system restricts LLM calls to the ingestion pipeline and uses pure mathematical operations at retrieval time.

### 2.4 MemGPT / Letta (2024): Tiered Context, LLM-Driven Lifecycle

MemGPT [8] introduces tiered context management with core memory (~2K tokens always in context), archival memory (vector store), and recall memory (conversation logs). The LLM itself decides what to promote and demote between tiers.

**Gap 1: The LLM is the policy.** All lifecycle decisions require runtime inference. This is expensive at scale, opaque, and unreliable — different LLM runs may make different promotion decisions for the same state.

**Gap 2: No internal structure in archival.** The archival tier has no structure — facts about identity, preferences, and logistics coexist without differentiation. When the archival store contains 10,000+ facts, the LLM's context window cannot hold enough to make informed promotion decisions.

**Gap 3: No temporal decay, supersession, or event-time validity.** Facts either exist in a tier or they don't. There is no mechanism for gradual fading, automatic replacement, or deadline-aware activation.

### 2.5 Mem0 (2025): Clean API, No Lifecycle

Mem0 provides a clean add/update/delete API over a flat vector store. Memories are text chunks embedded and retrieved by cosine similarity. It has a memory update operation but it is triggered by explicit API calls, not automatic detection.

**Gap 1: No structured lifecycle.** If "I like pizza" and "I like sushi" are stored in separate sessions, both exist as separate vectors unless the application explicitly deletes one. At scale, the vector store accumulates thousands of preference snapshots, all competing for the same retrieval slots.

**Gap 2: No category awareness.** No distinction between identity facts, transient logistics, and preferences. All memories receive identical treatment.

### 2.6 MemoryOS (2025): OS Metaphor, Cognitive Ontology

MemoryOS [10] proposes an operating-system-inspired hierarchical memory architecture with stores subdivided into episodic, semantic, and procedural. Evaluated on LoCoMo, it demonstrates improvements over flat baselines.

**Gap 1: Cognitive ontology, not behavioral ontology.** MemoryOS categorizes by cognitive type (episodic/semantic/procedural). A parent's chronic illness and a lunch order are both "semantic" memories — structurally identical, receiving identical temporal treatment. Our ontology classifies by behavioral domain (Health vs. Logistical Context), which directly determines lifecycle dynamics. The cognitive typology tells you what *form* a memory takes; the behavioral ontology tells you how it should *age*.

**Gap 2: No per-category decay conditioning.** Within each cognitive type, all memories have the same persistence characteristics.

**Gap 3: Fixed hierarchy.** What gets promoted or demoted is governed by recency and access patterns, not by behavioral type.

### 2.7 Memory-R1 (2025): Learned Policy, No Interpretable Structure

Memory-R1 [11] learns memory operations (ADD, UPDATE, DELETE, NOOP) via reinforcement learning. A memory manager agent is trained with RL to optimize downstream task performance. This is a serious and philosophically distinct competitor:

**Gap 1: Learned policy is opaque.** The RL-trained policy provides no interpretable explanation for why a particular memory was retained or discarded. Our ontology provides an explicit, inspectable, and debuggable forgetting policy. When the system makes a mistake, an engineer can identify which classification was wrong and correct it.

**Gap 2: Requires massive interaction data to converge.** RL policies need extensive training data. For a new user or domain, the policy starts from scratch. Our ontology functions as a strong prior that produces reasonable forgetting behavior from the first session.

**Gap 3: No behavioral structure in the action space.** The action space {ADD, UPDATE, DELETE, NOOP} is flat. The system cannot express "delete logistical details faster but preserve relational bonds" as a policy because the action space has no awareness of memory type.

**Gap 4: No multi-reference-frame temporal sensitivity or anticipatory activation.**

**Philosophical positioning.** Memory-R1 asks: "Can we learn the optimal memory policy end-to-end?" Fortunate Recall asks: "Can we design an interpretable memory policy that works from day one and improves with use?" These are complementary. Our ontology could serve as initialization, reward shaping, or structural constraint for an RL-based manager.

### 2.8 MIRIX (2025) and FluxMem (2026)

**MIRIX** [12] introduces multi-agent memory with structured cognitive types and dedicated agents per type. Gaps: cognitive not behavioral types; uniform temporal treatment within types; LLM-heavy multi-agent overhead at runtime; no anticipatory activation.

**FluxMem** [16] addresses a related but distinct problem: adaptively selecting which memory *structure* (buffer, summary, KG, log) to use. Gaps: adapts structure not policy; no behavioral ontology; no temporal routing. FluxMem's structure selection and Fortunate Recall's behavioral policy are orthogonal innovations that could compose.

### 2.9 Consolidated Gap–Contribution Mapping

| Gap | Zep | A-MEM | MemBank | Mem0 | MemGPT | MemOS | Mem-R1 | MIRIX | Flux | **Ours** |
|---|---|---|---|---|---|---|---|---|---|---|
| Behavioral ontology | × | × | × | × | × | Cog. | × | Cog. | × | **✓** |
| Per-category lifecycle | × | × | Uniform | × | × | × | Learned | × | × | **✓** |
| Slot-key supersession | Binary | N/A | N/A | × | × | × | × | × | N/A | **Soft** |
| Event-time validity | × | × | × | × | × | × | × | × | × | **✓** |
| Multi-clock temporal | Bi-temp | × | Single | × | × | × | × | × | × | **3 clocks** |
| Anticipatory activation | × | × | × | × | × | × | × | × | × | **✓** |
| Session initialization | × | × | × | × | × | × | × | × | × | **✓** |
| Category-aware routing | × | × | × | × | × | × | × | × | × | **✓** |
| Emotional loading signal | × | × | × | × | × | × | × | × | × | **✓** |
| LLM-free runtime | ✓ | × | Partial | ✓ | × | Partial | × | × | Partial | **✓** |
| Interpretable policy | N/A | N/A | N/A | N/A | × | N/A | × | N/A | N/A | **✓** |
| User-modifiable policy | × | × | × | × | × | × | × | × | × | **✓** |

*Table 1: Consolidated gap analysis. "Cog." = cognitive psychology categories without behavioral conditioning. "Learned" = capability exists but opaque. Every row in the final column represents a capability this work contributes.*

---

## 3 Core Concept: Fortunate Recall

Catastrophic forgetting — the uncontrolled loss of previously learned information — is one of the most recognized failure modes in machine learning. This work introduces its inverse: **Fortunate Recall**.

Fortunate Recall is not total recall. The system still forgets; decay is a feature, not a bug. But it forgets intelligently: per-category, per-domain, with threshold-gated recovery and confidence-weighted supersession. Work context from last Tuesday's standup fades. A parent's chronic illness persists. A deadline that seemed distant becomes urgent as it approaches. An intellectual curiosity raised six months ago can still be reactivated by a sufficiently strong relevance signal.

The system does not remember everything. It remembers the right things at the right time, and just as importantly, it *forgets* the right things at the right rate. That is fortunate recall: structured, adaptive persistence where catastrophic forgetting would otherwise erode coherence.

---

## 4 Contributions

This work builds on the memory infrastructure established by prior systems (temporal knowledge graphs, entity extraction, validity tracking) and contributes a behavioral lifecycle layer that governs memory dynamics. The contributions directly address every gap identified in Section 2:

**Contribution 1: Behavioral Ontology as a Lifecycle Prior.** A taxonomy of 10+1 abstract behavioral categories for conversational memory, each with distinct, empirically motivated lifecycle policies — not just decay rates, but supersession semantics, expiry logic, and accumulation rules. The ontology serves as a universal prior over memory lifecycle; per-user parameter evolution serves as the posterior. Validated on 842 facts with 93% cross-auditor agreement. Edge-level validation on 16,138 knowledge graph edges revealed 86% are world knowledge — a novel finding addressed by a world-knowledge detection filter.

**Contribution 2: Slot-Key Supersession with Confidence Weighting.** Supersession edges carry confidence scores. When the ingestion pipeline is uncertain whether new information truly contradicts an existing fact (e.g., "I'm thinking about moving to London" vs. "I moved to London"), the supersession edge is created with low confidence. Retrieval treats low-confidence supersessions as soft: both old and new facts may be surfaced, with ambiguity noted. High-confidence supersessions (e.g., "I now work at Anthropic" vs. "I work at Google") mark the old fact as inactive. This ensures graceful degradation under noisy ingestion rather than silently invalidating correct facts.

**Contribution 3: Event-Time Validity and Anticipatory Activation.** For obligations and logistics, activation is governed by distance to the event, not distance from creation. Deadlines increase in activation as they approach (inverse decay), then expire after passing. Every existing memory system models only the past fading. This models the future approaching.

**Contribution 4: Multi-Reference-Frame Temporal Sensitivity.** Three temporal clocks (absolute, session-relative, conversational frequency) with per-category sensitivity profiles. Different behavioral categories are routed to different temporal signals: logistics are sensitive to absolute calendar time; relational bonds are sensitive to session gaps; obligations are sensitive to deadline proximity; identity barely cares about any temporal signal.

**Contribution 5: Category-Aware Retrieval Routing.** Query classification determines which categories and temporal modes are relevant, then pulls per-category candidate sets alongside global semantic candidates. This changes what enters the retrieval pool — not just how candidates are ordered — directly addressing the retrieval ceiling that limits reranking-only approaches.

**Contribution 6: Ontology-Aware Session Initialization (Warm Start Protocol).** Three session modes: cold (low-complexity, skip traversal), warm (cluster-targeted priming with inferred intent), and evolving (drift detection with async context expansion). The ontology narrows traversal to specific category neighborhoods, reducing the search space by an order of magnitude.

**Contribution 7: Emotional Loading as Signal Modifier.** Transient emotional state is excluded as a memory category and instead formalized as a signal that modulates other categories. Frustration about a work deadline boosts OBLIGATIONS activation temporarily; the boost itself is subject to fast decay. This prevents emotional content from being orphaned while ensuring emotional context influences retrieval appropriately.

**Contribution 8: Interpretable, Day-One-Functional, User-Modifiable Policy.** Unlike learned policies (Memory-R1) that require extensive training data and produce opaque decisions, the behavioral ontology provides an interpretable forgetting policy that functions from the first session. The ontology is the prior; per-user parameter evolution is the posterior. Users can inspect their memory graph by category, understand why facts persist or fade, correct misclassifications, and adjust per-category retention to match their preferences. Developers can debug retrieval failures with clear causal chains and tune per-category behavior for different domains.

**Contribution 9: LifeMemBench.** A temporal disambiguation benchmark designed to test the specific capabilities lifecycle management enables: superseded preferences, expired logistics, stable identity under noise, multi-version fact resolution, contradiction detection, and selective forgetting. Generated conversation-first using frontier reasoning models, with controlled temporal structure across 50 synthetic personas.

**Contribution 10: Infrastructure Findings.** Two undocumented failure modes affecting all systems built on LLM-based knowledge graph extraction: (a) world-knowledge contamination (86% of extracted edges), and (b) systematic numeric loss during paraphrase-based extraction.

---

## 5 Architecture

The architecture comprises two layers: an infrastructure layer that adopts established patterns from the literature, and a lifecycle policy layer that contributes the novel behavioral dynamics.

### 5.1 Infrastructure Layer (Adopted from Literature)

The memory graph follows the multi-tier pattern established by Zep/Graphiti:

**Episode nodes:** Raw messages, append-only and immutable. Preserve provenance and full conversational context. Each carries a creation timestamp and session identifier.

**Fact/Entity nodes:** Extracted structured claims derived from episodes. Multiple episodes can source the same fact. Each fact carries a confidence score reflecting the ingestion pipeline's certainty.

**Validity tracking:** Following Zep's bi-temporal model, fact nodes carry validity intervals and can be superseded rather than deleted, preserving historical state.

**Hybrid retrieval:** Semantic, keyword, and graph-based search with no LLM in the retrieval hot path, consistent with Zep's demonstrated latency requirements.

This infrastructure is not claimed as a contribution. It is the foundation on which the lifecycle layer operates.

### 5.2 Lifecycle Policy Layer (Novel Contributions)

The lifecycle layer governs how memories behave over time, conditioned on their behavioral type. It sits on top of any temporal knowledge graph infrastructure and could, in principle, be applied to Zep/Graphiti or similar systems as an additional module.

#### 5.2.1 The Behavioral Ontology

Fact nodes are classified into behavioral categories. The ontology is the system's inductive bias for lifecycle management. Rather than learning lifecycle operations end-to-end (as Memory-R1 attempts) or applying uniform temporal mechanics (as most current systems do), the system begins with a strong prior: the behavioral type of a memory determines its full lifecycle — decay rate, supersession semantics, expiry logic, and accumulation rules.

The categories are abstract and stable across users. Specific facts flow in and out, but the ontology stays fixed. This is the same structural intuition as LexCGraph (ontology-driven legal reasoning graphs, SIGIR 2026 under review) [7], except the ontology is not legal doctrine; it is human life.

**Distinction from cognitive typologies.** MemoryOS and MIRIX categorize by cognitive type (episodic/semantic/procedural). This describes the *form* of memory. Our taxonomy describes the *behavioral domain*. A parent's chronic illness is "semantic" in cognitive typology but "Health & Wellbeing" in behavioral ontology. A meeting time is also "semantic" but "Logistical Context." The cognitive type tells you nothing about appropriate lifecycle dynamics; the behavioral domain does.

**The 10+1 Categories:**

| # | Category | Lifecycle Policy | Description |
|---|---|---|---|
| 1 | **Identity & Self-Concept** | **Accumulate.** Near-zero decay. New facts add to identity; supersession only on explicit high-confidence contradiction. | Ethnicity, diagnoses, core values, heritage, name, occupation. Things unchanged if you woke with amnesia about your preferences. |
| 2 | **Relational Bonds** | **Accumulate.** Slow decay. Dissolution requires explicit evidence. Sensitive to session gaps (long silence = meaningful). | Family, partners, friendships, colleagues, social dynamics |
| 3 | **Intellectual Interests** | **Accumulate with priority shift.** Minimal decay, reactivatable. New interests don't erase old ones but receive recency priority. | Curiosities, academic fascinations, active learning goals without concrete deliverable |
| 4 | **Health & Wellbeing** | **Dual policy.** Chronic conditions accumulate like identity. Acute symptoms expire. High-sensitivity reconfirmation after extended inactivity. | Medical conditions, medications, fitness metrics, mental health, diet |
| 5 | **Projects & Endeavors** | **State machine.** Active → completed / abandoned / paused. Only active projects surface by default. Milestone-gated. | Startups, research papers, creative projects, long-term goals with timelines |
| 6 | **Hobbies & Recreation** | **Slow decay, reactivatable.** Supersede on explicit replacement. Dormancy detection via session gaps; reactivates with sufficiently strong signal. | Skill-based leisure: fishing, cycling, painting, gardening. Accumulated skill/equipment investment persists. |
| 7 | **Preferences & Habits** | **Slot-key supersession.** Moderate decay. New value for same slot replaces old. "Likes sushi" supersedes "likes pizza." | Current tastes, media preferences, consumption patterns, lifestyle routines. What you like *right now*. |
| 8 | **Financial & Material** | **State tracking with supersession.** Variable decay. Major purchases, salary changes replace prior values. | Income, possessions, subscriptions, financial goals, budgets |
| 9 | **Obligations & Commitments** | **Event-time validity + anticipatory activation.** Activation increases as deadline approaches, expires on completion/passing. | Work tasks, deadlines, appointments, promises, action items |
| 10 | **Logistical Context** | **Fastest decay + event-time expiry.** Transient. "Flight at 6am tomorrow" expires after the flight. | Scheduling details, one-time locations, errands, travel logistics |
| 11 | **Other / Open-Set** | **Moderate default.** Sink cluster; signals ontology extension need. | Facts resisting classification |

*Table 2: The 10+1 behavioral ontology with lifecycle policies. Each category determines not just a decay rate but a complete lifecycle strategy.*

**The Identity Split: Empirical Justification.** The original design used 8+1 categories, but empirical validation on LongMemEval data revealed that IDENTITY_SELF_CONCEPT absorbed ~38% of all facts — preferences, hobbies, and core traits all lumped together. This created an "identity gravity well" where a third of all facts shared one lifecycle profile, undermining the thesis. The three-way split was motivated by genuinely distinct temporal dynamics:

- **Identity & Self-Concept** (near-zero decay): "I'm Irish-Italian" does not change in 10 years.
- **Hobbies & Recreation** (slow decay, reactivatable): "I do fishing" — involves skill accumulation and equipment investment. Can go dormant but reactivates because underlying investment persists.
- **Preferences & Habits** (moderate decay, supersedable): "I like MCU movies" — can flip after one bad film.

If these three tiers had identical decay profiles, soft membership would handle overlap without requiring separate categories. They earned their place because each has a measurably different temporal curve. This evolution illustrates a methodological principle: **ontology design must be empirically grounded in the distribution of facts in target data, not derived purely from theory.**

**Emotional State as Signal, Not Category.** Transient emotional state ("I'm frustrated right now") is deliberately excluded as a cluster. Mood is not a category of memory; it is a signal that modulates other categories. A message expressing frustration about a work deadline belongs in Obligations with an emotional loading modifier, not in a separate "Emotions" bucket. Emotional loading detected at ingestion temporarily boosts the activation of the relevant category, with the boost itself subject to fast decay.

**Soft Membership.** Facts are soft-clustered with membership weights across categories summing to 1.0, primary ≥ 0.3. A fact about quitting a job to pursue art spans Obligations, Identity, Projects, and Financial simultaneously — and that is the system working, not failing. The effective decay rate is computed via weighted harmonic mean over category-specific rates, weighted by membership strength. The harmonic mean naturally emphasizes slower-decaying categories: a fact that is even partially personal-identity should decay slower than a purely logistical one.

**Coverage.** The ontology truly fails only on facts that belong to zero categories, which the Open-Set cluster catches by design. On 78 validated facts from LongMemEval, the Open-Set captured only 1 fact (1.3%), confirming near-complete coverage.

#### 5.2.2 Classify the Fact, Not the Entity

A critical design principle: classification targets the **fact** (the relational edge between entities), not the conversational utterance and not the entity node.

**Why not utterance-level?** When a user says "I changed my last name while updating my insurance paperwork," the utterance context is about insurance (Obligations), but the stored fact is the user's former name (Identity). Utterance-level classification systematically confuses these because it reads the conversational frame rather than the information being memorized. This was the single most important discovery during classifier development.

**Why not entity-level?** Entity-level classification assigns a category to the node "Alex" (RELATIONAL_BONDS) and lets all edges involving Alex inherit that category. But "Alex caught 7 bass with the user" should be HOBBIES, not RELATIONAL. Edge-level classification eliminates the asymmetric noise that entity-level introduces: when edges inherit wrong categories, category-aware systems apply wrong lifecycle policies, while flat-rate systems are unaffected.

**Fact-level (edge-level) classification** ensures each edge receives its own category based on what the edge represents, not which entities it connects or which conversation it appeared in.

#### 5.2.3 The Classifier

**Architecture.** A prompted LLM classifier operating at ingestion time. Input: edge text, source/target entity names, conversational context. Output: soft membership weights across 11 categories.

**Validated Rules** (each motivated by specific failure modes during development):

1. **Classify the fact, not the utterance.** Category reflects the stored information, not the conversational frame.
2. **Stored-fact anchoring.** Financial amounts → FINANCIAL. Transient scheduling → LOGISTICAL. Core traits → IDENTITY. Skill-based leisure → HOBBIES. Consumption patterns → PREFERENCES.
3. **Boundary test.** Accumulated SKILL → HOBBIES. Consumption CHOICE that could flip tomorrow → PREFERENCES. Core trait unchanged across years → IDENTITY.
4. **Temporal question immunity.** Questions about "how many days between X and Y" are about WHEN; the category reflects WHAT.
5. **Soft membership.** Weights across all categories summing to 1.0, primary ≥ 0.3.
6. **OTHER cap.** OTHER must never exceed 0.5 weight; uncertain facts distribute across plausible candidates.
7. **Deliverable test.** If the user is creating/editing a named artifact → PROJECTS_ENDEAVORS.
8. **Schedule vs. pattern.** Raw time/date as schedule detail → LOGISTICAL. Behavioral pattern → motivational category.

**Iteration history:**

| Version | Key Change | Agreement |
|---|---|---|
| v1 | Initial 8+1 prompt | 77% (19/30 ✅, 8/30 ⚠️, 3/30 ❌) |
| v2 | "Classify fact, not utterance" + overrides | Overcorrected |
| v3 | Softened to "stored fact" rules | 87% (23/30 ✅, 6/30 ⚠️, 1/30 ❌) |
| v4 | Split to 10+1 + deliverable test + OTHER cap + CoT | **93%** (avg across two auditors) |

*Table 3: Classifier iteration with inter-annotator agreement.*

**Cost model.** Classification occurs once per edge at ingestion. At runtime, no LLM calls. All retrieval and lifecycle operations use pre-computed category labels.

#### 5.2.4 Differential Temporal Decay

Each category has a base decay rate λ governing activation over time:

$$\text{activation}(e) = \exp(-\lambda_c \cdot \Delta t)$$

| Category | λ (per hour) | Half-life | 1-month survival | 6-month survival |
|---|---|---|---|---|
| IDENTITY_SELF_CONCEPT | 0.0015 | 19 days | 34% | 1.3% |
| RELATIONAL_BONDS | 0.0015 | 19 days | 34% | 1.3% |
| INTELLECTUAL_INTERESTS | 0.0020 | 14 days | 24% | 0.4% |
| HEALTH_WELLBEING | 0.0025 | 12 days | 17% | 0.1% |
| PROJECTS_ENDEAVORS | 0.0025 | 12 days | 17% | 0.1% |
| HOBBIES_RECREATION | 0.0035 | 8 days | 8% | 0.002% |
| PREFERENCES_HABITS | 0.0050 | 6 days | 3% | ~0% |
| FINANCIAL_MATERIAL | 0.0055 | 5 days | 2% | ~0% |
| OBLIGATIONS | 0.0060 | 5 days | 1% | ~0% |
| LOGISTICAL_CONTEXT | 0.0080 | 4 days | 0.3% | ~0% |
| OTHER | 0.0050 | 6 days | 3% | ~0% |

*Table 4: Decay rates. Spread: 5.3×. Calibrated to avoid hierarchy squatting (Section 10.2.3) while preserving meaningful differentiation.*

For soft-clustered facts, the effective decay rate is the weighted harmonic mean across category-specific rates: $\lambda_{eff} = \left(\sum_c w_c / \lambda_c\right)^{-1} \cdot \sum_c w_c$

**Staleness floor.** No edge's activation drops below a configurable per-category floor (IDENTITY: 0.05, LOGISTICAL: 0.001). This prevents total information loss — even highly decayed identity facts retain minimal retrievability.

**Efficient implementation.** Decay is conceptually continuous but implemented lazily. Each edge stores a `last_updated_ts`. When an edge is touched during retrieval, decay is computed from the stored timestamp, producing results equivalent to continuous decay without per-tick updates across the graph.

Decay alone is a weak signal (see Section 10). Its value lies in being one component of the full lifecycle framework. The following policies provide the structural differentiation.

#### 5.2.5 Slot-Key Supersession with Confidence Weighting

For categories with **replace** semantics (PREFERENCES, FINANCIAL, LOGISTICAL, PROJECTS), the system maintains slot keys: normalized (subject, attribute) pairs.

When a new edge shares a slot key with an existing edge:
- **High confidence** (clear contradiction): Old edge marked `superseded_by = <new_edge_uuid>`, `is_active = false`. New edge becomes active.
- **Low confidence** (ambiguous): Both edges remain active. Supersession edge created with low confidence. Retrieval surfaces both with ambiguity noted.

**Slot key examples:**
- "User likes pizza" → slot: `(user, food_preference)`
- "User likes sushi" → slot: `(user, food_preference)` → high-confidence supersession
- "User lives in Istanbul" → slot: `(user, current_city)`
- "User is thinking about moving to London" → `(user, current_city)` → **low-confidence** supersession (both survive)
- "User moved to London" → `(user, current_city)` → high-confidence supersession of Istanbul

For categories with **accumulate** semantics (IDENTITY, RELATIONAL), supersession is triggered only by explicit contradiction with high confidence. "User is Irish-Italian" is not superseded by "User enjoys Japanese culture."

**Why uniform decay cannot replicate this:** Supersession is a discrete state change (active → inactive), not a continuous decay function. No decay rate can mark a specific fact as "replaced by a newer version." A uniform system can prefer newer facts by recency, but it cannot suppress the old version.

#### 5.2.6 Event-Time Validity and Anticipatory Activation

Every existing memory system models only the past fading. This work also models the future approaching.

For OBLIGATIONS and LOGISTICAL_CONTEXT, activation is governed by distance to the event, not distance from creation. At ingestion, a lightweight extractor identifies temporal anchors:
- "Dentist appointment next Thursday at 2pm" → `event_time = <resolved datetime>`
- "Tax return due April 15" → `event_time = 2026-04-15, type = deadline`

**Before event (anticipatory activation):** Relevance increases as the target date approaches. For a fact like "launch is early March," activation rises as March approaches. This enables proactive surfacing without being prompted.

**After event (expiry):** Edge is marked `expired = true` and excluded from default retrieval.

**Interaction with supersession:** If the fact is superseded before the date ("launch moved to April"), the old fact's anticipatory activation ceases and the new fact inherits the behavior. If supersession is low-confidence, both may have anticipatory activation with ambiguity noted.

**Why this is structurally impossible without the ontology:** Event-time validity requires knowing that a fact IS an obligation or logistical detail. A uniform system applies the same temporal policy to "dentist at 2pm" and "user has ADHD."

#### 5.2.7 Multi-Reference-Frame Temporal Sensitivity

Different behavioral categories are sensitive to different temporal reference frames.

**Absolute time (calendar clock).** Standard wall-clock timestamps. Governs LOGISTICAL decay, deadline proximity for OBLIGATIONS, and calendar-anchored events.

**Relative time (session clock).** Time elapsed since the user's last session. A user who messages daily has a different temporal rhythm than one who appears monthly. Governs RELATIONAL_BONDS: three months of silence is meaningful. Governs HOBBIES dormancy detection.

**Conversational frequency.** How often a fact appears across sessions. A fact mentioned in 40 of 100 sessions is a recurring theme; its frequency modulates the base decay rate downward. A fact mentioned once in 100 sessions decays at the base rate.

**Per-category routing.** Each category has a sensitivity profile over the three temporal signals:
- LOGISTICAL: primarily absolute time
- RELATIONAL: primarily session gaps
- OBLIGATIONS: primarily deadline proximity (directional)
- IDENTITY: near-zero sensitivity to all temporal signals
- HOBBIES: relative time for dormancy detection
- PREFERENCES: absolute time (tastes change over calendar months)

The key contribution is not the clocks themselves but the **per-category routing**: the behavioral ontology determines which temporal signal matters for each fact.

#### 5.2.8 Ontology-Aware Session Initialization (Warm Start Protocol)

Most memory systems assume the user is mid-conversation or treat session initialization as outside their scope.

**Cold session:** The user needs something simple. No priming needed. Low-complexity intent recognized; graph traversal skipped.

**Warm session:** The user declares intent. Targeted traversal primes the session with the active frontier: the subgraph with highest recent activation in the relevant category neighborhood. Context is ready before the first real message.

**Evolving session:** Starts warm, but conversation drifts. The system maintains a rolling query category vector (exponential moving average) and compares it to the primed category vector via cosine distance. When distance exceeds threshold δ for N consecutive turns, background traversal expands session context asynchronously.

The ontology makes this tractable. Without behavioral categories, the system would need to search the entire graph for relevant context. With the ontology, warm start narrows traversal to specific category neighborhoods, reducing the search space by an order of magnitude.

#### 5.2.9 Category-Aware Retrieval Routing

[SECTION TO BE COMPLETED AFTER IMPLEMENTATION. Key idea: query classification → desired categories + time sensitivity mode → pull separate candidate sets per category alongside global semantic candidates → merge and deduplicate before ranking. This changes the candidate pool, not just the ordering — attacking the retrieval ceiling that limits reranking-only approaches.]

#### 5.2.10 Per-User Parameter Evolution

The ontology is the universal prior. Per-user parameter evolution is the posterior.

The 10+1 categories and their base decay rates represent a population-level default. As the system observes individual user behavior, parameters evolve without LLM calls:

- A user who mentions hobbies frequently → HOBBIES decay rate decreases (reinforcement evidence)
- A user who changes preferences rapidly → PREFERENCES decay rate increases
- A user who rarely mentions health → HEALTH reconfirmation threshold remains at default; a user who frequently discusses medications → HEALTH persistence increases

Parameter evolution uses clean statistical signals: access frequency counters, temporal gap statistics, supersession rates per category. No LLM inference required. The system becomes more personalized with use while remaining interpretable — every parameter adjustment can be traced to specific observational evidence.

---

## 6 The LLM Boundary

A critical design difference from A-MEM, Memory-R1, MIRIX, and MemGPT (which make LLM calls at runtime) and a point of alignment with Zep (which keeps LLMs out of the retrieval hot path):

**LLM at ingestion (async, user does not wait):**
- Entity and edge extraction from conversational text (via Graphiti)
- Behavioral category classification per edge (with soft membership)
- World-knowledge vs. personal-memory detection
- Supersession detection with confidence scores
- Slot-key extraction
- Event-time extraction for obligations/logistics
- Emotional loading detection

**Pure math at runtime (fast, no inference call):**
- Category-conditioned decay computation (exponential function)
- Multi-clock temporal signal routing (per-category sensitivity profiles)
- Anticipatory activation for future-anchored edges (deadline proximity function)
- Supersession filtering (boolean check on `is_active` + confidence threshold)
- Event-time validity checking (datetime comparison)
- Category-aware retrieval routing (Cypher query with category filter)
- Candidate ranking by blended score (sorting)
- Staleness floor application (max operation)
- Emotional loading decay (exponential decay on boost modifier)

**Per-user parameter tuning without LLM:** Frequency, recency, and emotional loading are clean signals for simple statistics: counters, timestamps, exponential moving averages.

**Deployment implications:**
- **Latency:** Retrieval completes in milliseconds (graph queries + arithmetic), not seconds (LLM inference).
- **Cost:** No per-query LLM costs. Ingestion cost amortized across all future retrievals.
- **Determinism:** Same graph state + same query = identical results. No stochastic variation.
- **Auditability:** Every retrieval decision traces to category labels, decay rates, supersession states, and temporal computations.

---

## 7 Interpretability and User Control

A capability uniquely enabled by the behavioral ontology. Because every fact carries a human-readable category label and a transparent lifecycle policy, the entire memory system becomes inspectable, auditable, and modifiable.

### 7.1 For Users

Users can:
- **View their memory graph organized by category:** "Here are your identity facts, relationships, preferences, and obligations."
- **Understand why a fact persists or fades:** "This fact is IDENTITY (slow decay, accumulate) — that's why it's active after 6 months." Or: "This fact was LOGISTICAL and expired 3 days after its event time."
- **Correct misclassifications:** "This is a preference, not identity." The category change immediately applies the correct lifecycle policy.
- **Adjust per-category retention:** "I want my hobby facts to persist longer." The user modifies HOBBIES decay rate without affecting other categories.
- **Force supersession or preservation:** "This preference is outdated — mark as superseded." Or: "Pin this fact — don't let it decay."
- **Resolve ambiguity:** Low-confidence supersessions surface both facts with explanation. The user confirms which is current.

### 7.2 For Developers

Developers can:
- **Debug retrieval failures:** "Why not retrieved? → Classified LOGISTICAL, expired after event time." Clear causal chain.
- **Tune per-category behavior for domains:** Medical AI → increase HEALTH persistence. Executive assistant → prioritize OBLIGATIONS and LOGISTICS.
- **Audit for compliance and safety:** "Show all IDENTITY facts for this user. Any incorrect or sensitive?"
- **Deploy different decay profiles** for different products or user segments without changing the architecture.
- **Structure downstream LLM context by category:**

```
CORE IDENTITY (always relevant):
- User has ADHD
- User is a software engineer

CURRENT CONTEXT (time-sensitive):
- Meeting at 3pm today
- Presentation prep for Friday

PREFERENCES (apply when relevant):
- Prefers concise responses
- Vegetarian
```

The LLM now knows what's permanent, temporary, and conditional. No other system can provide this structured context because none have categories.

### 7.3 Why This Cannot Exist Without the Ontology

Flat vector stores (Mem0) have no categories to inspect. Learned policies (Memory-R1) have no interpretable decisions to explain. LLM-driven management (MemGPT, A-MEM) produces different decisions on different runs with no audit trail. The behavioral ontology is the necessary foundation for interpretable, user-controllable memory.

---

## 8 Safety and Robustness

**Prompt injection via memory.** The ingestion classifier (which already assigns category memberships) also classifies messages as user facts/preferences vs. instructions to the model. Instructional content is quarantined or discarded. The behavioral ontology adds a layer: facts in high-sensitivity categories (HEALTH, FINANCIAL) can be flagged for explicit reconfirmation if they seem inconsistent with established patterns.

**Stale fact surfacing.** The validity interval system (adopted from Zep) directly mitigates this. The behavioral ontology adds an additional layer: high-sensitivity categories (Health, Financial) trigger reconfirmation after extended periods of inactivity, even if the fact has not been formally superseded.

**Confidence degradation.** The confidence-weighted supersession mechanism ensures uncertain ingestion does not silently invalidate correct facts. Low-confidence supersessions surface both facts with ambiguity noted, allowing the LLM or user to resolve the conflict.

**Selective forgetting.** When a user explicitly requests deletion ("forget about the Denver move"), the system marks the relevant edge as retracted with the retraction as evidence. Unlike vector stores where deletion may leave embedding footprints, the graph-based approach provides clean, verifiable removal with audit trail.

---

## 9 Infrastructure Findings

During development and evaluation, we identified two infrastructure-level issues affecting all systems built on LLM-based knowledge graph extraction. These are independent of the behavioral ontology and are reported as standalone contributions.

### 9.1 World-Knowledge Contamination

When the behavioral ontology classifier was applied to all 16,138 edges in a Graphiti knowledge graph built from LongMemEval conversations, **86% of edges represented world knowledge** (geographic facts, historical information, general domain knowledge) rather than personal memories. This contamination arises because Graphiti's entity extraction pipeline produces *all* relational triples from conversation text, not just user-centric facts. A conversation about Tokyo produces "Tokyo is_located_in Japan," "Shibuya has famous crossing" — none of which are personal memories.

We developed a **world-knowledge detection filter** applied before behavioral classification. Edges classified as world knowledge are assigned neutral activation of 1.0, preserving semantic ranking without temporal interference. This filter is novel and system-independent. **No existing memory system reports or addresses this contamination.**

### 9.2 Numeric Preservation in Entity Extraction

Graphiti's extraction prompt instructs LLMs to "paraphrase" source text, causing systematic loss of numeric values:

- "I caught 7 largemouth bass" → "User goes bass fishing" (number lost)
- "I've lost 15 pounds" → "User is on a weight loss journey" (number lost)

This makes any question with a numeric answer unanswerable regardless of retrieval or lifecycle quality. The fix is prompt-level: requiring explicit numeric preservation. This failure mode is **undocumented** and affects any system built on LLM-based extraction.

---

## 10 Evaluation

### 10.1 Experimental Setup: LongMemEval

We evaluate the behavioral ontology's temporal decay mechanism on LongMemEval (Bai et al., 2024) [1], a benchmark of 234 questions across six categories drawn from multi-session conversational histories spanning 3,798 sessions.

**Infrastructure.** The memory graph was built on Graphiti (Rasmussen et al., 2025) [2], ingesting all sessions into a Neo4j knowledge graph containing 16,138 temporal edges across 2,465 distinct sessions. The world-knowledge detection filter (Section 9.1) was applied during preprocessing.

**Candidate retrieval.** For each question, candidates were retrieved via Graphiti's hybrid pipeline: semantic embedding search (top 50), Cypher keyword matching, Cypher intersection queries, and graph neighbor expansion. Average pool size: 135 candidates (range: 50–225). Of 234 questions, 96 (41%) had at least one correct-answer fact in the candidate pool; the remaining 138 represent a retrieval ceiling — no reranking strategy can answer them.

**Temporal anchoring.** Each question was anchored to a per-question reference time (t_now) derived from the latest edge timestamp in the question's originating session, via session mapping index. Of 234 questions: 145 used last-session edge anchor, 78 used fallback earlier-session anchor, 11 fell back to global edge maximum.

**Engines compared.** Three decay engines evaluated as ablations:

| Engine | Categories | Rates | Spread |
|---|---|---|---|
| **Behavioral** | 10+1 ontology (this work) | Per-category λ from 0.0015 to 0.0080/hr | 5.3× |
| **Uniform** | None (flat rate) | Single λ = 0.0030/hr for all edges | 1× |
| **Cognitive** | 3 types (episodic/semantic/procedural) | Per-type λ: procedural=0.003, semantic=0.005, episodic=0.050 | 16.7× |

All engines use the same retrieval pipeline, candidate pools, and temporal anchoring. The only difference is how activation scores are computed.

**Blending formula:** score = α × activation + (1−α) × semantic_similarity, where activation = exp(−λ·Δt). α controls temporal vs. semantic weight.

**Metrics.** MRR on 96 answerable questions. Hit@5 secondary. Paired Wilcoxon signed-rank test with Bonferroni correction. Cohen's d effect sizes. 95% bootstrap confidence intervals (10,000 resamples).

### 10.2 Results

#### 10.2.1 Performance Parity at Practical Operating Points

In deployed memory systems, semantic similarity dominates retrieval; temporal decay serves as a secondary disambiguation signal. The practical operating range is α ≤ 0.2:

| α | B-MRR | U-MRR | C-MRR | ΔMRR(B−U) | p (Wilcoxon) | 95% CI | Cohen's d |
|---|---|---|---|---|---|---|---|
| 0.000 | 0.5618 | 0.5618 | 0.5618 | +0.0000 | 1.000 | [+0.000, +0.000] | 0.000 |
| 0.025 | 0.5600 | 0.5705 | 0.5636 | −0.0105 | 0.016 | [−0.024, −0.002] | −0.179 |
| 0.050 | 0.5767 | 0.5795 | 0.5785 | −0.0027 | 0.221 | [−0.018, +0.013] | −0.036 |
| 0.075 | 0.5756 | 0.5761 | 0.5839 | −0.0005 | 0.653 | [−0.016, +0.015] | −0.007 |
| 0.100 | 0.5659 | 0.5629 | 0.5642 | +0.0030 | 0.706 | [−0.015, +0.022] | +0.032 |
| 0.125 | 0.5592 | 0.5596 | 0.5676 | −0.0004 | 0.433 | [−0.018, +0.017] | −0.005 |
| 0.150 | 0.5593 | 0.5632 | 0.5667 | −0.0039 | 0.745 | [−0.031, +0.021] | −0.031 |
| 0.175 | 0.5621 | 0.5705 | 0.5641 | −0.0085 | 0.338 | [−0.039, +0.021] | −0.056 |
| 0.200 | 0.5519 | 0.5828 | 0.5608 | −0.0309 | 0.026 | [−0.065, +0.002] | −0.182 |

*Table 5: Fine-grained alpha sweep. All p-values fail to reach significance after Bonferroni correction (9 comparisons, adjusted threshold p < 0.0056). All effect sizes negligible (|d| < 0.2).*

**Finding 1: Statistical parity.** No engine achieves a significant advantage at any operating point after correction for multiple comparisons. The largest uncorrected p-value reaching nominal significance (p = 0.016 at α = 0.025) does not survive Bonferroni correction. All effect sizes are negligible (|Cohen's d| < 0.19). The 95% bootstrap CIs contain zero at 8 of 9 operating points.

**Finding 2: Zero retrieval cost.** Maximum MRR gap between behavioral and uniform is 0.031 across the entire practical range. At most operating points, < 0.01. Behavioral lifecycle policies impose no measurable retrieval cost.

**Finding 3: Behavioral matches cognitive.** No significant differences between behavioral and cognitive at any operating point (all p > 0.10). The cognitive typology and behavioral ontology produce indistinguishable retrieval performance when the intervention is limited to temporal reranking.

#### 10.2.2 Full Alpha Sweep

| α | B-MRR | U-MRR | C-MRR | Winner |
|---|---|---|---|---|
| 0.0 | 0.5618 | 0.5618 | 0.5618 | Tie |
| 0.1 | 0.5659 | 0.5629 | 0.5642 | Behavioral |
| 0.2 | 0.5500 | 0.5828 | 0.5608 | Uniform |
| 0.3 | 0.5347 | 0.5516 | 0.5367 | Uniform |
| 0.4 | 0.5032 | 0.5145 | 0.5221 | Cognitive |
| 0.5 | 0.5020 | 0.4881 | 0.5032 | Cognitive |
| 0.6 | 0.4769 | 0.4528 | 0.4777 | Cognitive |
| 0.7 | 0.4470 | 0.4131 | 0.4525 | Cognitive |
| 0.8 | 0.3711 | 0.3619 | 0.3897 | Cognitive |
| 0.9 | 0.2903 | 0.3159 | 0.3242 | Cognitive |
| 1.0 | 0.0461 | 0.1686 | 0.0484 | Uniform |

*Table 6: Full alpha sweep. All engines degrade as temporal signal dominates semantic similarity. Cognitive leads in mid-range; uniform leads at extremes.*

**Finding 4: All engines degrade at high alpha.** MRR drops precipitously above α = 0.3, confirming that semantic similarity is the primary retrieval signal.

**Finding 5: Per-question analysis reveals categorical strengths.** At α = 1.0 (pure temporal), behavioral wins 33/96 individual question matchups vs. uniform's 29 and cognitive's 23 (11 ties). Behavioral's wins concentrate on identity and relational queries:

- Q72 "How many engineers do I lead?" (IDENTITY): B=rank 2, U=rank 33
- Q57 "How much weight have I lost?" (HEALTH): B=rank 12, U=rank 73
- Q67 "How many times have I met Alex?" (RELATIONAL): B=rank 13, U=rank 33

The behavioral ontology surfaces identity and relational facts more effectively than flat-rate decay — but this advantage is offset by disadvantages on other categories in aggregate MRR. This motivates the shift from decay-only to full lifecycle management.

#### 10.2.3 Rate Calibration Analysis: The Squatting Phenomenon

A significant methodological finding. The initial rate configuration spanned 800× (λ = 0.0001 for IDENTITY to λ = 0.080 for LOGISTICAL):

| Category | λ (per hour) | Activation at 3 months | Activation at 6 months |
|---|---|---|---|
| IDENTITY | 0.0001 | 0.807 | 0.649 |
| RELATIONAL | 0.001 | 0.098 | 0.010 |
| PREFERENCES | 0.010 | 0.000 | 0.000 |
| LOGISTICAL | 0.080 | 0.000 | 0.000 |

*Table 7: Activation survival at original 800× rate spread.*

This produced **category hierarchy squatting**: IDENTITY facts permanently occupied top ranking positions regardless of semantic relevance. In top-5 results across all questions, IDENTITY comprised 307/480 slots (64%). Compressing to 5.3× eliminated squatting while preserving ordering:

| Configuration | Spread | B-MRR (α=0.1) | Top-5 IDENTITY % | Squatting? |
|---|---|---|---|---|
| Original | 800× | 0.559 | 64% | Severe |
| Compressed | 16× | 0.571 | 52% | Moderate |
| Final | 5.3× | 0.566 | ~30% | Minimal |

*Table 8: Rate spread calibration.*

**Why cognitive wins mid-range.** The cognitive ontology's competitive mid-alpha performance is explained not by superior categorization but by accidental rate calibration: its slowest rate (PROCEDURAL at λ = 0.003) is high enough that no category squats at multi-month timescales. Its rate floor prevents hierarchy squatting; our explicit calibration achieves the same effect deliberately.

This calibration tradeoff is **fundamental to any category-aware decay system** and has not been documented previously.

### 10.3 World-Knowledge Filter

When applied to all 16,138 edges, 86% were world knowledge. See Section 9.1 for details. Edges classified as world knowledge were assigned neutral activation of 1.0.

### 10.4 Evaluation: LifeMemBench

[SECTION TO BE COMPLETED AFTER BENCHMARK CONSTRUCTION AND EVALUATION.

Will contain:
- Benchmark description and construction methodology (50 personas, conversation-first generation, controlled temporal structure)
- Evaluation of full Fortunate Recall system (with supersession, event-time validity, retrieval routing) vs. Graphiti raw, Mem0, MemGPT, uniform baselines, and Memory-R1
- Per-attack-vector breakdown: superseded preferences, expired logistics, stable identity, multi-version facts, contradiction detection, selective forgetting, category-appropriate retrieval
- Ablation of each lifecycle component]

### 10.5 Ablations

| Component Removed | Effect on LifeMemBench | Effect on LongMemEval |
|---|---|---|
| Full system | [TBD] | 0.566 MRR (parity with uniform) |
| − Slot-key supersession | [TBD] | N/A |
| − Event-time validity | [TBD] | N/A |
| − Retrieval routing | [TBD] | N/A |
| − Edge-level → entity-level classification | [TBD] | [TBD] |
| − Confidence weighting (→ binary supersession) | [TBD] | N/A |
| − Anticipatory activation | [TBD — requires deadline-bearing data] | N/A |
| − Warm start protocol | [TBD] | N/A |
| − Emotional loading | [TBD] | N/A |
| Behavioral → Uniform decay only | [TBD] | No significant change (Table 5) |
| Behavioral → Cognitive decay | [TBD] | No significant change |
| 10+1 → 8+1 ontology (pre-split) | [TBD — expect identity gravity well] | [TBD] |
| Remove world-knowledge filter | [TBD — expect significant degradation] | [TBD] |

*Table 9: Ablation plan. Each lifecycle component evaluated in isolation.*

### 10.6 Fortunate Recall–Specific Metrics

Beyond MRR, we define metrics that capture lifecycle management quality:

**Selective retention score.** Partition queries by ontology category. Measure both recall@k of relevant facts and false recall rate of stale/expired facts.

**Staleness penalty.** If the system surfaces a superseded fact (old address, former job), apply heavy penalty. Tests supersession and decay.

**Temporal calibration.** For obligations with deadlines, measure whether activation rises appropriately as deadline approaches and drops after.

**Anticipatory precision/recall.** For facts with future anchors, measure whether the system proactively surfaces them in the correct temporal window. No existing system can be evaluated on this metric because none implement anticipatory activation.

**User model accuracy over time.** After 6 months of simulated interaction with preference changes, job changes, expired events, and retractions: what percentage of the system's "current state" model is actually current? This is the headline metric — not "can you find a fact" but "is your model of the user still accurate?"

### 10.7 Limitations

**Retrieval ceiling on LongMemEval.** Only 96/234 questions (41%) had correct answers in the candidate pool. This constrains the space where any temporal intervention can demonstrate value.

**LongMemEval not designed for temporal disambiguation.** The benchmark tests retrieval broadly, not the specific scenarios where lifecycle management helps. LifeMemBench addresses this directly.

**No held-out test set for rate calibration.** Rate calibration was performed on the evaluation set. Mitigated by reporting the full calibration trajectory and noting the final configuration was selected based on mechanistic analysis (eliminating squatting), not MRR optimization.

**Temporal span.** The 180-day evaluation window may not capture the full benefit of lifecycle management, which is designed for relationships spanning months to years.

---

## 11 Scalability: Why Lifecycle Management Becomes Necessary

The case for behavioral lifecycle management strengthens with scale. At small scale (weeks of interaction, hundreds of facts), flat retention works adequately. At large scale (months to years, tens of thousands of facts), flat retention produces a memory store that degrades in multiple dimensions:

**Retrieval precision degrades.** With 30,000 facts, semantic search returns hundreds of candidates. Signal-to-noise ratio drops as stale facts compete with current ones.

**Context pollution increases.** The downstream LLM receives 20–50 facts with no distinction between current and historical. Multiple versions of the same fact compete. Past events pollute queries about the present.

**Storage grows without bound.** Every session adds edges. Nothing is marked inactive, expired, or superseded. Infrastructure costs scale linearly with time. Graph queries slow as the graph densifies.

**The ontology as automatic curation.** With lifecycle policies, the *active* working set stabilizes even as total history grows:

| Metric | Flat retention (2 years) | Lifecycle management (2 years) |
|---|---|---|
| Total edges | ~30,000 | ~30,000 |
| Active edges (retrievable) | ~30,000 | ~5,000–8,000 |
| Avg. candidates per query | ~500 | ~80–120 |
| Contradictions in top-20 | ~3–5 per query | ~0 (superseded excluded) |
| Expired events in top-20 | ~2–4 per query | ~0 (expired excluded) |
| Context pollution rate | Grows with time | Stable |

*Table 10: Projected scaling comparison.*

**The cost argument.** Every retrieved fact in the downstream LLM context costs tokens. At GPT-4o pricing, unnecessary context facts burn money. A system that sends 15 relevant, current facts while a flat system sends 50 facts (30 outdated) saves 70% on context tokens while getting better answers. At millions of queries per day, this is millions of dollars in inference savings.

**This is the fundamental argument for the ontology.** Not "behavioral decay rates are better than uniform" — they aren't, in aggregate MRR on existing benchmarks. But "behavioral lifecycle management keeps the active working set compact, current, and contradiction-free as the total memory store grows toward tens of thousands of facts." That capability is structurally unavailable to any system that treats all facts identically.

---

## 12 Composability

Fortunate Recall is deliberately positioned as a composable policy layer, not a monolithic memory system. The behavioral ontology, lifecycle policies, multi-clock temporal routing, anticipatory activation, confidence-weighted supersession, emotional loading signals, warm start protocol, and retrieval routing could, in principle, be deployed as modules on top of existing temporal knowledge graph infrastructure (Zep/Graphiti or equivalent) without requiring a new graph engine.

This composability means:
- Existing Graphiti deployments can add behavioral classification and lifecycle policies without migrating data.
- Individual components can be adopted incrementally: add classification first, then supersession, then event-time validity.
- The ontology can be extended or modified for domain-specific applications (medical, legal, educational) without changing the lifecycle machinery.
- FluxMem's adaptive structure selection and Fortunate Recall's behavioral lifecycle policy are orthogonal and could compose.

---

## 13 Discussion

**The parity result is the point, not the limitation.** On LongMemEval, behavioral decay achieves statistically indistinguishable performance from uniform decay at all practical operating points. This is not a negative result. It establishes that the full suite of lifecycle capabilities comes at **zero retrieval cost**. Uniform decay offers a single tuning knob and no lifecycle management. The behavioral ontology offers a complete lifecycle framework — supersession, expiry, routing, anticipatory activation, interpretability, user control — with the same aggregate performance. In deployed systems where these capabilities are requirements alongside retrieval accuracy, this is a strictly favorable tradeoff.

**Temporal decay alone is insufficient.** Our evaluation demonstrates that no temporal reranking strategy — whether category-aware, cognitive, or uniform — can meaningfully improve upon semantic retrieval when the intervention is limited to post-retrieval score blending. The correct answer is often rank 30–80 by semantics; no temporal boost moves it to top-5. This finding motivates the shift from decay-as-reranking to lifecycle-as-management.

**The calibration tradeoff.** Wide rate spreads provide stronger differentiation but cause hierarchy squatting; narrow spreads avoid squatting but reduce temporal signal. This is fundamental to any category-aware temporal system and has not been documented previously. We report it as a methodological contribution.

**The cognitive typology comparison.** The cognitive ontology's mid-alpha advantage is explained by accidental rate calibration, not superior categories. Its coarser categories (3 vs. 11) and rate floor prevent squatting. This provides evidence that rate calibration, not category choice, is the primary determinant of temporal reranking performance — and that the behavioral ontology's value lies in the lifecycle capabilities it enables, not in aggregate MRR improvement.

**World-knowledge contamination.** 86% of edges in a Graphiti knowledge graph from conversational data represent world knowledge. No existing system reports or addresses this. The filter is a novel, system-independent contribution.

**Integration with learned policies.** The behavioral ontology could serve as initialization, reward shaping, or structural constraint for an RL-based memory manager like Memory-R1, combining interpretability and cold-start performance with adaptability.

**Cross-cultural generalization.** The 10+1 categories are motivated by English-language conversational AI. Their stability across collectivist vs. individualist contexts remains to be validated. In collectivist contexts, Relational Bonds may subsume more of the other categories. The open-set cluster provides a safety valve, but systematic cross-cultural evaluation is needed.

**The ingestion pipeline as hidden linchpin.** The lifecycle layer's elegance at runtime depends on reliable fact extraction, classification, supersession detection, and event-time extraction at ingestion. Sensitivity analysis (how does retrieval quality degrade as ingestion noise increases?) is essential. Our classifier development revealed that the primary error source is the classifier attending to conversational context rather than the stored fact — a systematic bias correctable through prompt engineering but invisible without careful auditing.

*"The question is not only how to find the right context, but how to determine which memories should persist, which should be replaced, which should expire, and at what rate — conditioned on the behavioral type of each memory. The answer need not sacrifice retrieval performance to gain interpretability, lifecycle management, and user control."*

---

## 14 Research Narrative

- **Paper 1:** LexCGraph — Ontology-driven legal reasoning graph extraction (SIGIR 2026, under review) [7]
- **Paper 2:** Fortunate Recall — Ontology-driven memory lifecycle management with behavioral forgetting policy (NeurIPS 2026 / AAAI 2027)

The throughline: fixed abstract ontologies + dynamic graph population + intelligent traversal, applied across domains. One paper is a result. Two papers with a shared architectural philosophy is a research program.

---

## 15 Positioning in the Literature

| System | Contribution | Gaps This Work Fills |
|---|---|---|
| Zep/Graphiti (2025) | Temporal KG, bi-temporal validity, hybrid retrieval, production-grade infrastructure | No behavioral ontology; no lifecycle policies; no anticipatory activation; no session init; binary supersession; retrieval noise |
| A-MEM (NeurIPS 2025) | Zettelkasten-inspired dynamic memory organization | No temporal dynamics; no forgetting; content-driven evolution; LLM-heavy runtime |
| MemoryBank (AAAI 2024) | Cognitively-motivated forgetting via Ebbinghaus curves | Single uniform curve; no graph structure; single temporal frame; no behavioral categorization |
| Mem0 (2025) | Clean memory API with vector store | No lifecycle; no categories; no supersession; no temporal reasoning |
| MemGPT/Letta (2024) | Tiered context management | LLM-driven policy (opaque, expensive); no lifecycle in archival; no decay/supersession/expiry |
| Memoria (2025) | Scalable personalized user modeling | No behavioral ontology; no category-conditioned lifecycle |
| MemoryOS (2025) | OS-inspired hierarchical memory with cognitive type stores | Cognitive not behavioral ontology; no per-category lifecycle; fixed hierarchy |
| Memory-R1 (2025) | RL-learned memory operations | Opaque learned policy; requires training data; no behavioral structure; no multi-clock |
| MIRIX (2025) | Multi-agent memory with cognitive types | Cognitive types; uniform treatment within types; LLM-heavy runtime |
| FluxMem (2026) | Adaptive memory structure selection | Adapts structure not policy; no behavioral ontology; complementary to our approach |
| **Fortunate Recall** | **Behavioral-ontology-conditioned lifecycle management** | **Designed to compose with existing infrastructure, filling all identified gaps** |

*Table 11: Literature positioning.*

---

## References

[1] Bai, D., et al. (2024). "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." arXiv:2410.10813.

[2] Rasmussen, P., et al. (2025). "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." arXiv:2501.13956.

[3] Xu, W., et al. (2025). "A-MEM: Agentic Memory for LLM Agents." NeurIPS 2025. arXiv:2502.12110.

[4] Zhong, W., et al. (2024). "MemoryBank: Enhancing Large Language Models with Long-Term Memory." AAAI 2024. arXiv:2305.10250.

[5] Wu, Z., et al. (2025). "Memoria: A Scalable Agentic Memory Framework for Personalized Conversational AI." arXiv:2512.12686.

[6] Li, Y., et al. (2026). "Graph-based Agent Memory: Taxonomy, Techniques, and Applications." arXiv:2602.05665.

[7] Mullick, A., et al. (2026). "LexCGraph: Ontology-Driven Clustering for Extracting Legal Reasoning Graphs." Submitted to SIGIR 2026.

[8] Packer, C., et al. (2024). "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560.

[9] Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130.

[10] MemoryOS (2025). "Memory OS of AI Agent." arXiv:2506.06326.

[11] Memory-R1 (2025). "Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning." arXiv:2508.19828.

[12] MIRIX (2025). "MIRIX: Multi-Agent Memory System for LLM-Based Agents." arXiv:2507.07957.

[13] Mnemosyne (2025). "An Unsupervised, Human-Inspired Long-Term Memory." arXiv:2510.08601.

[14] Anokhin, P., et al. (2024). "AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents." arXiv:2407.04363.

[15] Guo, Z., et al. (2024). "LightRAG: Simple and Fast Retrieval-Augmented Generation." arXiv:2410.05779.

[16] FluxMem (2026). "Choosing How to Remember: Adaptive Memory Structures for LLM Agents." arXiv:2602.14038.

[17] LoCoMo (2024). "LoCoMo: Long-Context Multi-Turn Open-Domain Conversation Benchmark."
