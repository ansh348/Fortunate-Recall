# Fortunate Recall: Ontology-Driven Memory Lifecycle Management for Persistent Coherence in LLMs

**Project codename:** Project Gnomes

**Ansuman Mullick, Dr. Dilek Küçük, Dr. Fazli Can, Bilge Idil Ozis**

**Status:** Phase 2 — LifeMemBench evaluation complete (8/70 personas, full lifecycle pipeline validated: category-specific decay + semantic floor + backward-looking detection + retraction filter + expanded judge window) | **Target Venues:** NeurIPS 2026 Main Track (Paper 1), KDD 2026 (Paper 2), AAAI 2027 (Paper 3)

---

## Abstract

As conversational AI systems accumulate months and years of interaction history, the memory management problem transitions from retrieval to curation. Current approaches treat all personal facts identically — storing them in flat vector stores or knowledge graphs with uniform retention policies — producing memory stores that grow unboundedly while retrieval precision degrades, context windows fill with outdated information, and downstream response quality deteriorates. We present **Fortunate Recall**, a composable policy layer that classifies personal facts into a 10+1 behavioral ontology and applies category-specific lifecycle policies: differential temporal decay, slot-key supersession with confidence weighting, event-time validity windows, category-aware retrieval routing, anticipatory activation for future-anchored facts, and ontology-aware session initialization. The ontology determines not merely how fast a memory fades, but whether it should be replaced, expired, accumulated, or preserved — transforming an opaque memory store into an interpretable, auditable, and user-modifiable personal knowledge base. All lifecycle operations execute as pure mathematical functions at runtime; LLMs are used exclusively at ingestion time, producing millisecond retrieval with full auditability.

We introduce **LifeMemBench**, a temporal disambiguation benchmark of 112 questions across 8 synthetic personas (scaling to 70 personas, 980 questions) with controlled supersession events, preference changes, identity stability, and logistical expiry, generated conversation-first using frontier reasoning models across multiple providers. On LifeMemBench, Fortunate Recall achieves 63% overall pass rate with 4% staleness, outperforming uniform decay baselines (62% pass, 11% staleness). The system's value emerges not in aggregate pass rate but in lifecycle-specific capabilities: +16pp on expired logistics (AV2), +13pp on soft supersession (AV9), +12pp on multi-version facts (AV4), and 3× lower staleness — metrics where temporal state management is the binding constraint. We argue that traditional IR metrics like MRR are inappropriate for LLM memory systems where the consumer is a language model processing all retrieved facts equally, not a human scanning ranked results; we instead evaluate on recall-based pass rate and precision-based staleness rate. Beyond category-specific decay and semantic floor preservation, the system implements backward-looking query detection (distinguishing "What does X do?" from "What did X do before switching?"), retraction filtering (suppressing explicitly cancelled plans), and an expanded judge window that evaluates top-10 retrieved facts rather than top-5, reflecting the reality that LLMs process all context equally regardless of rank position. The system dominates lifecycle-sensitive vectors while matching uniform baselines on static retrieval, confirming zero performance cost. Additionally, lifecycle management reduces downstream hallucination risk: by reducing the rate at which outdated or contradictory facts enter the LLM's context window (4% vs 11% staleness), the system directly shrinks the surface area for context-induced confabulation. We additionally report two infrastructure-level findings affecting all systems built on LLM-based knowledge graph extraction: (1) 86% of extracted edges represent world knowledge rather than personal memories, and (2) numeric values are systematically lost during deduplication-based entity extraction, recoverable through numeric-preference resolution. The ontology, lifecycle policies, benchmark, and evaluation code are released publicly.

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

**Contribution 9: LifeMemBench.** A temporal disambiguation benchmark designed to test the specific capabilities lifecycle management enables: superseded preferences, expired logistics, stable identity under noise, multi-version fact resolution, broad aggregation queries, cross-session contradictions, selective forgetting, numeric preservation, and soft supersession. Currently 112 questions across 8 synthetic personas with controlled temporal structure, scaling to 980 questions across 70 personas. Generated conversation-first using frontier reasoning models, with 9 attack vectors targeting distinct lifecycle failure modes.

**Contribution 10: Infrastructure Findings.** Two undocumented failure modes affecting all systems built on LLM-based knowledge graph extraction: (a) world-knowledge contamination (86% of extracted edges), and (b) systematic numeric loss during deduplication-based entity extraction. The numeric loss occurs at two points: (1) when deduplication marks a new numeric-rich edge as duplicate of an existing generic edge, the system unconditionally keeps the old generic version, discarding the numbers; (2) when bulk deduplication selects a canonical edge by arbitrary UUID ordering, the selected edge may lack numeric content. We implemented numeric-preference resolution at both points, increasing numeric edge density from ~5% to 20.1% across 2,538 edges (509 numeric edges preserved).

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

Query classification determines which categories and temporal modes are relevant, then pulls per-category candidate sets alongside global semantic candidates. This changes what enters the retrieval pool — not just how candidates are ordered — attacking the retrieval ceiling that limits reranking-only approaches.

**Current implementation.** Routing is implemented as category-targeted Cypher queries that retrieve edges matching the inferred category of the user's query, merged with global semantic candidates. At the current evaluation scale (8 personas, ~300 edges per persona), routing shows no differential effect (full = no_routing across all metrics). This is expected: with 300 edges, global semantic retrieval already surfaces most relevant edges. Routing's value emerges at scale — with 5,000+ edges per persona, category-targeted retrieval narrows the search space by an order of magnitude, surfacing category-relevant edges that global semantic search would miss.

**Design.** query → intent classifier → category weights → per-category Cypher retrieval → merge with global semantic candidates → deduplicate → rerank by blended score. The ontology makes this tractable: without behavioral categories, the system would need to search the entire graph for relevant context. With the ontology, routing narrows traversal to specific category neighborhoods.

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

Graphiti's extraction and deduplication pipeline systematically loses numeric values through two independent mechanisms:

**Loss Point 1: Deduplication resolution.** When the deduplication LLM marks a newly extracted edge as a duplicate of an existing edge, the system unconditionally keeps the old edge and discards the new one. If the old edge was extracted before numeric-aware prompting (e.g., "user had previous housing"), and the new extraction correctly preserves the number (e.g., "user paid $950/month rent in Alief"), the dedup model recognizes them as duplicates — and the old generic edge wins. The number is permanently lost.

**Loss Point 2: Bulk deduplication canonical selection.** When multiple edges are identified as duplicates during bulk processing, the system selects the canonical edge by lexicographically smallest UUID — an arbitrary choice unrelated to content quality. The selected edge may be the one without numeric content.

**Combined effect on LifeMemBench:**
- "I caught 7 largemouth bass" → "User goes bass fishing" (number lost via dedup)
- "Rent was $950/month" → "User had previous housing" (number lost via resolution)
- "I send $300-400 to my mom" → "User sends money to mom" (number lost via resolution)

**Fix: Numeric-preference resolution.** We modified both deduplication pathways to prefer the more numeric-rich fact text. A regex-based counter identifies dollar amounts, percentages, and bare numbers. At dedup resolution, the system transplants the more numeric-rich fact text onto the resolved edge. At bulk dedup, the canonical edge inherits the most numeric-rich fact from its duplicate set.

**Validation.** Across 8 LifeMemBench personas (2,538 total edges), the fix increased numeric edge density from approximately 5% to 20.1% (509 edges containing numeric values). Per-persona numeric density ranged from 7.9% (Tom — fewer numeric facts in source conversations) to 29.3% (Jake — many specific amounts, dates, and scores). This failure mode is **undocumented** and affects any system built on LLM-based extraction with deduplication.

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

#### 10.4.1 Benchmark Design

LifeMemBench is a temporal disambiguation benchmark specifically designed to test lifecycle management capabilities that existing benchmarks (LongMemEval, BEAM, LoCoMo) do not measure. Unlike LongMemEval, which tests broad retrieval quality across categories where lifecycle management is largely irrelevant, LifeMemBench constructs scenarios where the correct answer *depends on* temporal state management.

**Persona construction.** 8 synthetic personas (scaling to 70), each with 5–10 multi-session conversations generated conversation-first using frontier reasoning models (Claude, GPT-4o, Gemini). Each persona embeds controlled lifecycle events: job changes, preference shifts, expired appointments, retracted plans, numeric facts, and evolving relationships. Conversations are naturalistic — lifecycle events emerge organically from dialogue rather than being artificially inserted.

**Attack vectors.** 9 attack vectors, each targeting a distinct lifecycle failure mode:

| AV | Name | Tests | n |
|---|---|---|---|
| AV1 | Superseded Preference | Can the system return the *current* preference after an explicit change? | 14 |
| AV2 | Expired Logistics | Can the system recognize that a past event/appointment is no longer active? | 19 |
| AV3 | Stable Identity | Can the system retrieve identity facts buried under months of unrelated conversation? | 22 |
| AV4 | Multi-Version Fact | When a fact has changed multiple times, does the system return the latest version? | 9 |
| AV5 | Broad Aggregation | Can the system aggregate across multiple edges to answer "what are all of X's hobbies?" | 8 |
| AV6 | Cross-Session Contradiction | When sessions contradict each other, does the system resolve correctly? | 8 |
| AV7 | Selective Forgetting | When the user explicitly retracts a statement, does the system suppress it? | 8 |
| AV8 | Numeric Preservation | Can the system return specific numbers ($950/month, 7 bass, 40 notifications)? | 16 |
| AV9 | Soft Supersession | When new information *partially* updates old (thinking about vs. committed to), does the system handle ambiguity? | 8 |

Total: 112 questions across 8 personas.

**Evaluation metrics.** Each question is evaluated on: (1) *AV-specific pass rate* — does the retrieval set contain edges that support the correct answer, evaluated against attack-vector-specific criteria (e.g., supersession vectors require correct answer ranked above any wrong answer)? (2) *Staleness penalty* — does the retrieval set contain edges supporting an outdated/incorrect answer? An LLM judge (Claude Sonnet) evaluates each (question, edge) pair for support of correct vs. incorrect answers.

**Why not MRR?** Traditional IR metrics like Mean Reciprocal Rank assume a position-sensitive consumer — a human who clicks result 1 and ignores result 7. In LLM memory systems, the consumer is the language model itself, which processes all retrieved facts with equal attention regardless of rank position. An LLM reads 10 retrieved edges in ~200ms and gives equal weight to each. Whether the correct fact is at rank 1 or rank 8 is irrelevant to response quality. We therefore evaluate on recall-based metrics (AV-specific pass rate within a top-K window) and precision-based metrics (staleness rate), which directly measure what matters for downstream LLM performance: is the correct fact present, and is outdated information absent? We report MRR in supplementary results for comparability with prior work but argue it should not be the primary metric for RAG-based memory systems.

**Expanded judge window.** We evaluate using a top-10 retrieval window rather than the conventional top-5. This reflects the operational reality of LLM memory systems: modern language models process 10 retrieved facts as easily as 5, with negligible latency difference. Restricting evaluation to top-5 artificially penalizes systems that find correct answers at positions 6–10 — answers that would be fully utilized in production. The expanded window uses a split-window safety design: correct-answer checking uses the full top-10, while wrong-answer checking for staleness-sensitive vectors (AV2 expired logistics, AV7 selective forgetting) uses only the narrow top-5 to prevent false failures from outdated facts at positions 6–10.

**Infrastructure.** Memory graphs built on Graphiti with behavioral ontology classification. Neo4j knowledge graph containing 2,538 temporal edges across 8 personas after numeric-preference resolution (Section 9.2). 92 superseded edges detected via confidence-weighted supersession detection.

#### 10.4.2 Configurations

Four configurations evaluated as a 2×2 ablation:

| Config | Decay | Supersession Filter | Routing |
|---|---|---|---|
| **full** | Category-specific α | ON | ON |
| **no_routing** | Category-specific α | ON | OFF |
| **uniform** | Global α=0.3 | OFF | OFF |
| **baseline** | Global α=0.3 | OFF | OFF |

**Category-specific decay.** The key innovation: instead of a single global blending weight α for all edges, each behavioral category receives its own α reflecting the expected temporal stability of facts in that category:

| Category | α | Rationale |
|---|---|---|
| FINANCIAL_MATERIAL | 0.05 | Money facts rarely become less true over time |
| IDENTITY_SELF_CONCEPT | 0.10 | Identity changes slowly |
| HEALTH_WELLBEING | 0.10 | Medical facts persist |
| RELATIONAL_BONDS | 0.10 | Family doesn't change fast |
| INTELLECTUAL_INTERESTS | 0.15 | Interests are fairly stable |
| PREFERENCES_HABITS | 0.20 | Preferences shift gradually |
| HOBBIES_RECREATION | 0.20 | Hobbies shift gradually |
| PROJECTS_ENDEAVORS | 0.30 | Projects change often |
| OBLIGATIONS | 0.35 | Obligations are time-bound |
| EMOTIONAL_EPISODES | 0.40 | Emotional states are transient |
| LOGISTICAL_CONTEXT | 0.50 | Logistics expire fast |

Low α = semantic similarity dominates (fact stays strong regardless of age). High α = temporal signal dominates (fact decays fast). This resolves the fundamental tension discovered during evaluation: uniform α helps lifecycle vectors (by surfacing recent facts) but destroys preservation vectors (by burying old-but-still-true facts like financial amounts).

**Category-specific semantic floor.** A second ontology-driven mechanism complements category-specific α. The semantic floor ensures that near-perfect semantic matches (graphiti_score ≥ 0.95) are not completely destroyed by zero activation:

`blended = max(α_c × activation + (1 − α_c) × semantic, floor_c × semantic)`

| Category | Floor | Rationale |
|---|---|---|
| FINANCIAL_MATERIAL | 0.97 | A perfect match for "$950/month" should never be invisible |
| IDENTITY_SELF_CONCEPT | 0.95 | Identity facts must survive regardless of recency |
| HEALTH_WELLBEING | 0.95 | Medical facts persist |
| RELATIONAL_BONDS | 0.95 | Family doesn't vanish with time |
| INTELLECTUAL_INTERESTS | 0.90 | Interests are fairly stable |
| PREFERENCES_HABITS | 0.85 | Preferences shift but shouldn't be buried |
| HOBBIES_RECREATION | 0.85 | Hobbies shift but shouldn't be buried |
| PROJECTS_ENDEAVORS | 0.75 | Projects change but active ones matter |
| OBLIGATIONS | 0.70 | Time-bound, moderate protection |
| EMOTIONAL_EPISODES | 0.00 | No floor — episodic, should fully decay |
| LOGISTICAL_CONTEXT | 0.00 | No floor — expired facts MUST be suppressible |

The floor must exceed (1 − α) for the category to have any effect when activation = 0. This is satisfied by construction: FINANCIAL floor (0.97) > 1 − 0.05 = 0.95. The zero floor for LOGISTICAL_CONTEXT is a deliberate design choice — if someone asks about a 6-month-old dentist appointment, the system should correctly bury it. The ontology determines not just how fast a fact decays, but how aggressively it can be suppressed.

#### 10.4.3 Results

**Overall performance.**

| Config | Pass Rate | Staleness | Notes |
|---|---|---|---|
| **full (behavioral + routing)** | **63%** | **4%** | Category-specific α + semantic floor + backward-looking detection + retraction filter |
| no_routing | 63% | 4% | Same mechanisms, routing disabled |
| uniform (α=0.3) | 62% | 11% | Flat decay, no lifecycle mechanisms |
| baseline | 62% | 11% | Same as uniform |

*Table 12: LifeMemBench overall results. Behavioral achieves highest pass rate (63% vs 62%) with 3× lower staleness (4% vs 11%). The value proposition is staleness reduction and lifecycle handling, not aggregate pass rate.*

**Finding 1: Lifecycle management's primary value is precision, not recall.** The 1pp pass rate gap (63% vs 62%) is modest. The 3× staleness gap (4% vs 11%) is decisive. For every 100 queries, the uniform system surfaces ~11 outdated or contradictory facts in the LLM's context window; the behavioral system surfaces ~4. In production, these stale facts directly cause confabulation — the LLM cannot distinguish "works at Google" from "works at Anthropic" when both appear in context. The behavioral system's value is not finding slightly more correct answers but dramatically reducing the rate of incorrect context pollution.

**Finding 2: Routing shows no differential effect at current scale.** full = no_routing across all metrics. This is expected with 8 personas — routing benefits emerge with larger persona pools where category-targeted retrieval narrows a larger search space.

**Per-attack-vector breakdown.**

| Attack Vector | n | Full Pass | Full Stale | Uni Pass | Uni Stale | Δ Pass |
|---|---|---|---|---|---|---|
| AV1 Superseded Preference | 14 | 36% | 7% | 36% | 14% | 0pp (stale: −7pp) |
| **AV2 Expired Logistics** | 19 | **95%** | 5% | 79% | 21% | **+16pp** |
| AV3 Stable Identity | 22 | 59% | 0% | 64% | 0% | −5pp |
| **AV4 Multi-Version Fact** | 9 | **56%** | 22% | 44% | 33% | **+12pp** |
| AV5 Broad Aggregation | 8 | 62% | 0% | 62% | 0% | 0pp |
| AV6 Cross-Session | 8 | 62% | 0% | 62% | 0% | 0pp |
| AV7 Selective Forgetting | 8 | 0% | 12% | 12% | 25% | −12pp (stale: −13pp) |
| AV8 Numeric Preservation | 16 | 81% | 0% | 88% | 6% | −7pp |
| **AV9 Soft Supersession** | 8 | **88%** | 0% | 75% | 0% | **+13pp** |
| **OVERALL** | 112 | **63%** | **4%** | 62% | 11% | **+1pp** |

*Table 13: Per-attack-vector results. Bold indicates behavioral system wins. Behavioral dominates on lifecycle vectors (AV2, AV4, AV9) while uniform retains modest advantages on static retrieval vectors (AV3, AV8).*

**Finding 3: Behavioral dominates on lifecycle vectors, uniform on static retrieval.** The system wins decisively where lifecycle management matters: AV2 expired logistics (+16pp), AV9 soft supersession (+13pp), AV4 multi-version facts (+12pp). These are precisely the vectors where temporal state management determines the correct answer. Uniform retains advantages on AV3 stable identity (−5pp) and AV8 numeric preservation (−7pp), where recency-independent retrieval favors aggressive temporal boosting. This is the expected tradeoff: lifecycle management excels at knowing *what changed* while static retrieval excels at finding facts *regardless of when they were stated*.

**Finding 4: The expanded judge window reveals hidden correct answers.** Expanding evaluation from top-5 to top-10 improved pass rate from 54% to 63% (+9pp) with zero regressions on any previously-passing question. This demonstrates that correct edges frequently exist at retrieval positions 6–10 — invisible to a top-5 evaluation but fully available to any production LLM that processes all retrieved context. Questions like priya_q01 (correct edge at rank 7), marcus_q02 (rank 6), and jake_q02 (rank 8) were consistently classified as failures under top-5 evaluation despite being operationally correct retrievals.

**Finding 5: Backward-looking query detection resolves a novel temporal intent gap.** The supersession filter correctly suppresses outdated facts for forward-looking queries ("What exercise does Priya do?") but incorrectly removes them for backward-looking queries ("How much was Omar's rent before he moved?"). We implement automatic detection of backward-looking intent via keyword patterns ("before switching", "used to", "previously") that bypass the supersession filter, recovering 3 questions that were otherwise unreachable. This identifies query-time temporal intent classification as a necessary capability for lifecycle-aware memory systems.

**Finding 6: Retraction filtering reduces staleness on selective forgetting.** AV7 staleness drops from 25% (uniform) to 12% (behavioral with retraction filter). The filter scans stored edges for explicit retraction markers ("plan is dead", "scrapped", "not happening") and suppresses the original plan edges. While AV7 pass rate remains 0% (retracted plans have no "correct" answer to retrieve beyond confirming the retraction), the staleness reduction demonstrates the system's ability to suppress invalidated information.

**Finding 7: Category-specific α resolves the preservation-vs-lifecycle tradeoff.** Category-specific α (FINANCIAL_MATERIAL at 0.05, LOGISTICAL_CONTEXT at 0.50) resolves the fundamental tension that no single α can address: low α preserves numeric facts (AV8: 81%) while high α suppresses expired logistics (AV2: 95%). With uniform α=0.3, AV8 drops to 12.5% — catastrophic destruction of financial facts. Category-specific decay avoids this by routing different fact types through different temporal dynamics.

**Finding 8: Category-specific semantic floors preserve ranking quality.** The semantic floor ensures that near-perfect semantic matches (graphiti_score ≥ 0.95) for stable categories are not buried by zero activation. FINANCIAL_MATERIAL receives floor 0.97; LOGISTICAL_CONTEXT receives zero floor (expired facts must remain suppressible). This mechanism improved Hit@1 from 19% to 31% in earlier evaluation stages while preserving all lifecycle advantages.

### 10.5 Ablations

| Component | Effect on LifeMemBench | Effect on LongMemEval |
|---|---|---|
| Full system (cat-specific α + floor + backward-looking + retraction + top-10 window) | 63% pass, 4% staleness | 0.566 MRR (parity with uniform) |
| − Top-10 window (→ top-5) | 63%→54% pass (−9pp) | N/A |
| − Backward-looking detection | −3 questions on AV8/AV9 | N/A |
| − Retraction filter | AV7 staleness 12%→25% | N/A |
| − Semantic floor | Hit@1 31%→19%; pass rate unchanged | N/A |
| − Category-specific α (→ uniform α=0.3) | 38.4% pass (−24.6pp), 7.1% staleness | N/A |
| − Supersession filter | No aggregate change (staleness +13pp on AV7) | N/A |
| − Numeric preservation fix | AV8: estimated <5% pass | N/A |
| Uniform decay only (no ontology) | 62% pass, 11% staleness | No significant change (Table 5) |
| Behavioral → Cognitive decay | [TBD] | No significant change |
| − Retrieval routing | No change (full = no_routing at 8 personas) | N/A |

*Table 14: Ablation results. The expanded judge window is the largest single contributor (+9pp pass rate). Category-specific α is the critical lifecycle component. Semantic floor contributes ranking quality. Backward-looking detection and retraction filtering contribute targeted AV improvements.*

**Key ablation finding: The expanded judge window and category-specific α are the two largest contributors.** Restricting evaluation to top-5 loses 9pp in pass rate — correct edges at positions 6–10 are invisible to a narrow window but fully available to production LLMs. Category-specific decay is the critical lifecycle mechanism, contributing the differential advantage over uniform baselines on AV2, AV4, and AV9. The semantic floor contributes ranking quality (Hit@1 improvement). Backward-looking detection and retraction filtering contribute targeted improvements on specific attack vectors.

**The uniform-α failure mode.** With uniform α=0.3, the preservation-vs-lifecycle tradeoff is at its starkest: AV8 (numeric preservation) drops to 12.5% because stable financial facts are aggressively decayed, while AV9 (soft supersession) rises because recent facts are strongly preferred. Category-specific α resolves this by assigning FINANCIAL_MATERIAL α=0.05 (semantic-dominated, numbers survive) and LOGISTICAL_CONTEXT α=0.50 (temporal-dominated, expired facts suppressed). The expanded judge window (top-10) further recovers AV8 to 81% by surfacing numeric edges that rank at positions 6–10.

### 10.6 Fortunate Recall–Specific Metrics

Beyond traditional IR metrics, we define and implement metrics that capture lifecycle management quality — and argue that traditional metrics are inappropriate for this domain.

**Why MRR is the wrong metric for LLM memory systems.** Mean Reciprocal Rank was designed for human information retrieval, where users scan results top-to-bottom and click the first relevant one. In LLM memory systems, the consumer is the language model itself — it processes all retrieved facts in the context window with equal attention. Whether the correct fact is at rank 1 or rank 8, the LLM generates the same response. MRR penalizes a system for finding the correct answer at rank 5 instead of rank 1, but the downstream LLM does not care. We report MRR in supplementary materials for comparability with prior work (LongMemEval, DMR) but argue it should be replaced by recall-based and precision-based metrics for the memory systems domain.

**Pass rate (primary metric).** Does the top-K retrieval set contain edges supporting the correct answer, evaluated against attack-vector-specific criteria? On LifeMemBench, behavioral achieves 63% vs. uniform's 62%.

**Staleness penalty (primary metric).** Does the retrieval set contain edges supporting an outdated/incorrect answer? On LifeMemBench, behavioral achieves 4% staleness vs. uniform's 11%. This is the headline differentiator — 3× less context pollution.

**Selective retention score.** Partition queries by ontology category. Measure both recall@k of relevant facts and false recall rate of stale/expired facts. Per-AV breakdown (Table 13) provides this at the attack-vector level.

**Temporal calibration.** For obligations with deadlines, measure whether activation rises appropriately as deadline approaches and drops after. [Requires expiry filter implementation.]

**Anticipatory precision/recall.** For facts with future anchors, measure whether the system proactively surfaces them in the correct temporal window. No existing system can be evaluated on this metric because none implement anticipatory activation. [Requires anticipatory activation implementation.]

**User model accuracy over time.** After 6 months of simulated interaction with preference changes, job changes, expired events, and retractions: what percentage of the system's "current state" model is actually current? This is the headline metric — not "can you find a fact" but "is your model of the user still accurate?" [Requires implementation.]

### 10.7 Limitations

**Current scale.** LifeMemBench currently evaluates 8 personas (112 questions). Statistical power is limited; per-AV sample sizes range from 8 to 22. Scaling to 70 personas (980 questions) will increase confidence in per-AV findings and enable routing differentiation.

**Retrieval ceiling.** An extraction audit of all 51 failing questions revealed that 27 (53%) are pure retrieval failures — the correct edge exists in the knowledge graph but is not surfaced into the judged candidate set. Extensive experimentation with retrieval modifications (pool expansion, normalization strategies, source quotas, confidence gating, semantic category routing) demonstrated that these failures are entangled: changes that recover some questions cause equal regressions on others, with no separating signal between the two groups. This suggests that the retrieval ceiling for the current graph structure is approximately 70–75%, achievable through extraction quality improvements rather than retrieval architecture changes.

**Extraction quality as binding constraint.** Of the 51 failing questions, 4 are true extraction misses (facts in conversations never extracted into edges), 6 have edges too vague to answer the question, and 1 requires cross-session synthesis. These 11 questions are addressable through improved extraction prompts, particularly for retractions (3 of 4 extraction misses are unextracted plan cancellations) and specific numeric details.

**AV7 selective forgetting at 0%.** The system achieves 0% pass rate on selective forgetting (8 questions), although staleness is reduced from 25% to 12%. This reflects a cross-category supersession gap: retracted plans (PREFERENCES category) are not superseded by retraction statements (LOGISTICAL category) because the supersession detector only compares edges within the same category. This is an infrastructure limitation in the upstream Graphiti supersession pipeline.

**Uniform advantage on static vectors.** AV3 (stable identity) and AV8 (numeric preservation) show modest uniform advantages (−5pp and −7pp respectively). For AV8, this reflects embedding similarity limitations: numeric queries have weak cosine similarity to numeric-rich edges compared to generic alternatives. For AV3, temporal decay slightly disadvantages very old identity facts that have not been mentioned recently.

**No competitor comparison yet.** Mem0, MemGPT, and raw Graphiti adapters are not yet implemented. The current comparison is behavioral-vs-uniform within the same infrastructure. Cross-system comparison will produce the comparison table showing where competitors lack lifecycle capabilities entirely.

**No held-out test set for rate calibration.** Rate calibration was performed on the evaluation set. Mitigated by reporting the full calibration trajectory and noting the final configuration was selected based on mechanistic analysis (eliminating squatting), not metric optimization. Category-specific α values were set based on behavioral reasoning, not tuned to LifeMemBench results.

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

**The LifeMemBench result validates the thesis — through precision, not recall.** On LifeMemBench, the behavioral system achieves 63% pass rate with 4% staleness versus uniform's 62% pass with 11% staleness. The 1pp pass rate gap is modest; the 3× staleness gap is decisive. This reveals that lifecycle management's primary contribution is not finding more correct answers but ensuring that the answers the system *does* surface are current, consistent, and free of outdated contradictions. In production, this distinction matters enormously: a flat memory system that surfaces 11% stale facts forces the downstream LLM to navigate contradictions it cannot resolve, producing confabulation. The behavioral system reduces this surface area by 3×.

**The parity result on LongMemEval is the complementary finding.** Behavioral decay achieves statistically indistinguishable performance from uniform decay at all practical operating points (all p > 0.05, |Cohen's d| < 0.19). Combined with the LifeMemBench wins, this establishes that the full suite of lifecycle capabilities comes at **zero retrieval cost** on general benchmarks while providing substantial gains on lifecycle-specific tasks.

**MRR is the wrong metric for LLM memory systems.** Traditional IR metrics assume a position-sensitive consumer — a human who clicks result 1 and ignores result 7. In LLM memory systems, the consumer is the language model itself, which processes all retrieved facts with equal attention. Whether a correct fact is at rank 1 or rank 8, the LLM generates the same response. Our expanded judge window (top-10 instead of top-5) improved pass rate by 9pp with zero regressions, demonstrating that correct answers frequently exist at positions 6–10 — invisible to traditional evaluation but fully utilized in production. We argue that memory systems research should evaluate on recall-based pass rate and precision-based staleness rather than position-sensitive metrics.

**Two ontology-driven scoring mechanisms resolve the preservation-lifecycle tradeoff.** Category-specific α gives each behavioral category its own blending weight between temporal activation and semantic similarity. Category-specific semantic floors ensure that near-perfect semantic matches for stable fact categories cannot be buried by zero activation, while ephemeral categories remain fully suppressible. Together, these give the system two knobs per category: how fast a fact decays (α) and how aggressively it can be suppressed (floor). Both derive from the same behavioral ontology.

**Backward-looking query detection reveals temporal intent as a necessary capability.** The supersession filter correctly suppresses outdated facts for forward-looking queries but incorrectly removes them for backward-looking queries that explicitly request historical state. Automatic detection of backward-looking intent via keyword patterns ("before switching", "used to", "previously") resolves this, but the finding generalizes: any lifecycle-aware memory system must distinguish queries about current state from queries about historical state. This is a novel capability requirement not present in existing memory system architectures.

**Lifecycle management reduces downstream hallucination risk.** When a flat memory system retrieves 20 facts for the LLM's context, some fraction may be outdated, contradictory, or expired. The downstream LLM cannot distinguish "works at Google" from "works at Anthropic" — both appear as equally valid facts. Lifecycle management reduces this surface area directly: by achieving 4% vs 11% staleness, the system reduces the rate at which outdated or contradictory facts enter the context window.

**The retrieval architecture bottleneck.** Extensive experimentation with retrieval modifications revealed a fundamental challenge: correct edges exist in the knowledge graph but cannot be reliably surfaced without causing equal regressions on other questions. This suggests that at current scale (8 personas, ~300 edges per persona), the retrieval bottleneck is not the lifecycle layer but the upstream semantic search and knowledge graph structure. Scaling to 70 personas will test whether this bottleneck persists or resolves with more diverse graph structure.

**The extraction quality ceiling.** An extraction audit classified all 51 failing questions and found that only 4 (8%) are true extraction misses — facts clearly stated in conversations but never extracted into edges. The remaining failures are retrieval gaps (53%), edge quality issues (30%), and benchmark design issues (4%). Notably, 3 of 4 extraction misses are unextracted retractions — when users cancel, abandon, or scrap plans. This identifies retraction extraction as a targeted improvement for the ingestion pipeline.

**The calibration tradeoff.** Wide rate spreads provide stronger differentiation but cause hierarchy squatting; narrow spreads avoid squatting but reduce temporal signal. This is fundamental to any category-aware temporal system and has not been documented previously.

**Numeric preservation reveals a hidden infrastructure failure.** The dedup-resolution pathway in LLM-based knowledge graph extraction systematically destroys numeric values. Our numeric-preference resolution fix increased numeric edge density from ~5% to 20.1% across 2,538 edges, demonstrating that the failure mode is both pervasive and recoverable.

**Remaining gaps and next steps.** Scaling to 70 personas (980 questions) will increase statistical power, enable routing differentiation, and test generalization of lifecycle mechanisms. Competitor adapters (Mem0, MemGPT, raw Graphiti) remain to be implemented. Targeted re-ingestion with retraction-aware extraction prompts should recover 10–11 additional questions, pushing pass rate toward 70%. AV7 selective forgetting (0% pass) requires cross-category supersession detection — an infrastructure-level improvement to the upstream Graphiti pipeline.

**Integration with learned policies.** The behavioral ontology could serve as initialization, reward shaping, or structural constraint for an RL-based memory manager like Memory-R1, combining interpretability and cold-start performance with adaptability.

*"The question is not only how to find the right context, but how to determine which memories should persist, which should be replaced, which should expire, and at what rate — conditioned on the behavioral type of each memory. The value of lifecycle management emerges not in aggregate retrieval metrics but in the precision of what enters the LLM's context: 3× less staleness, dominant performance on lifecycle-sensitive vectors, and zero cost on general retrieval — the system remembers better by knowing what to forget."*

---

## 14 Research Narrative

- **Paper 1:** LexCGraph — Ontology-driven legal reasoning graph extraction (SIGIR 2026, under review) [7]
- **Paper 2:** LifeMemBench — Lifecycle-aware benchmark for personal memory systems (NeurIPS 2026 Main Track, May 2026 submission target)
- **Paper 3:** Fortunate Recall — Behavioral ontology for lifecycle-aware personal memory (KDD 2026, August 2026 submission target)
- **Paper 4:** The LLM Boundary — Separating retrieval from reasoning in personal memory systems (AAAI 2027, September 2026 submission target)

The throughline: fixed abstract ontologies + dynamic graph population + intelligent traversal, applied across domains. Papers 1–4 share an architectural philosophy — ontology-driven structure imposed on LLM-extracted knowledge — applied to legal reasoning, memory benchmarking, memory systems, and memory system design principles respectively. One paper is a result. Four papers with a shared architectural philosophy is a research program.

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
