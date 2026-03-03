"""
decay_engine.py — Category-Conditioned Temporal Decay Engine
Fortunate Recall | Goal 2 | NeurIPS 2026

Core computation module for behavioral-ontology-conditioned memory decay.
Pure math at retrieval time. No I/O, no LLM calls, no database access.

This is the hot path. Everything here must be fast and toggleable.

Usage:
    from decay_engine import DecayEngine, DecayConfig, FactNode, TemporalContext

    engine = DecayEngine()  # default behavioral ontology config
    activation = engine.compute_activation(fact_node, temporal_context)

Ablation presets:
    engine = DecayEngine.uniform()          # Ablation 1: no per-cluster decay
    engine = DecayEngine.single_clock()     # Ablation 2: no multi-clock routing
    engine = DecayEngine.no_anticipatory()  # Ablation 3: no anticipatory activation
    engine = DecayEngine.no_emotional()     # Ablation 8: no emotional loading
    engine = DecayEngine.cognitive()        # Ablation 9: cognitive typology replaces behavioral
    engine = DecayEngine.arithmetic_blend() # Ablation 10: arithmetic mean blending
    engine = DecayEngine.max_only()         # Ablation 11: primary cluster only, no soft blending
    engine = DecayEngine.ontology_8plus1()  # Ablation 12: original 8+1 (no Identity split)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ============================================================================
# Categories — must match classify_facts.py canonical set
# ============================================================================

CATEGORIES = [
    'OBLIGATIONS', 'RELATIONAL_BONDS', 'HEALTH_WELLBEING', 'IDENTITY_SELF_CONCEPT',
    'HOBBIES_RECREATION', 'PREFERENCES_HABITS', 'INTELLECTUAL_INTERESTS',
    'LOGISTICAL_CONTEXT', 'PROJECTS_ENDEAVORS', 'FINANCIAL_MATERIAL', 'OTHER',
]
K = len(CATEGORIES)  # 11
CAT_INDEX = {cat: i for i, cat in enumerate(CATEGORIES)}

# Cognitive typology mapping (for Ablation 9)
# Maps each behavioral category to its cognitive type
COGNITIVE_MAP = {
    'OBLIGATIONS':           'SEMANTIC',    # fact about a task
    'RELATIONAL_BONDS':      'SEMANTIC',    # fact about a relationship
    'HEALTH_WELLBEING':      'SEMANTIC',    # fact about health
    'IDENTITY_SELF_CONCEPT': 'SEMANTIC',    # fact about identity
    'HOBBIES_RECREATION':    'PROCEDURAL',  # skill-based activity
    'PREFERENCES_HABITS':    'SEMANTIC',    # fact about preferences
    'INTELLECTUAL_INTERESTS': 'SEMANTIC',   # fact about interests
    'LOGISTICAL_CONTEXT':    'EPISODIC',    # event/scheduling detail
    'PROJECTS_ENDEAVORS':    'PROCEDURAL',  # ongoing work with process
    'FINANCIAL_MATERIAL':    'SEMANTIC',    # fact about finances
    'OTHER':                 'SEMANTIC',    # default
}

COGNITIVE_TYPES = ['EPISODIC', 'SEMANTIC', 'PROCEDURAL']

# 8+1 collapse mapping (for Ablation 12)
# In the original 8+1, HOBBIES and PREFERENCES fold back into IDENTITY
COLLAPSE_8PLUS1 = {
    'HOBBIES_RECREATION':   'IDENTITY_SELF_CONCEPT',
    'PREFERENCES_HABITS':   'IDENTITY_SELF_CONCEPT',
}


# ============================================================================
# Configuration enums
# ============================================================================

class BlendingMode(Enum):
    """How to combine per-cluster decay rates for multi-cluster facts."""
    HARMONIC = "harmonic"       # Default: weighted harmonic mean (emphasizes slow-decaying clusters)
    ARITHMETIC = "arithmetic"   # Ablation 10: weighted arithmetic mean
    MAX_ONLY = "max_only"       # Ablation 11: primary cluster only, no soft blending


class ClockMode(Enum):
    """Which temporal signals to use."""
    MULTI = "multi"             # Default: three clocks with per-cluster routing
    ABSOLUTE_ONLY = "absolute"  # Ablation 2: single absolute timestamp only


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class DecayConfig:
    """All decay parameters. Every field is an ablation switch.

    To run an ablation, create a config with the relevant field changed
    and pass it to DecayEngine. Or use a preset: DecayEngine.uniform(), etc.
    """

    # --- Per-cluster base decay rates (λ_k, per hour) ---
    # Higher λ = faster decay. Initial priors; per-user tuning adjusts these.
    #
    # Compressed 16x spread (0.0005–0.0080) so no category hits 0.0 before
    # 3 months on a 180-day timeline. Ordering preserved: IDENTITY slowest,
    # LOGISTICAL fastest. Previous 800x spread (0.0001–0.080) caused IDENTITY
    # facts to squat top-5 ranks regardless of semantic relevance.
    #
    #   IDENTITY (0.0005/hr):    ~1.2% per day. Core self-concept, very stable.
    #   RELATIONAL (0.0015/hr):  ~3.5% per day. Relationships persist across months.
    #   INTELLECTUAL (0.0020/hr):~4.7% per day. Curiosities dormant but reactivatable.
    #   HEALTH (0.0025/hr):      ~5.8% per day. Conditions persist but context shifts.
    #   PROJECTS (0.0025/hr):    ~5.8% per day. Milestone-gated; active projects relevant.
    #   HOBBIES (0.0035/hr):     ~8.1% per day. Skills persist; dormancy via reactivation.
    #   PREFERENCES (0.0050/hr): ~11.3% per day. Tastes shift; supersession handles updates.
    #   FINANCIAL (0.0055/hr):   ~12.4% per day. Variable — purchases fade, debts persist.
    #   OBLIGATIONS (0.0060/hr): ~13.5% per day. Anticipatory activation overrides pre-deadline.
    #   LOGISTICAL (0.0080/hr):  ~17.5% per day. "Meeting at 3pm" fades within weeks.
    #   OTHER (0.0050/hr):       Default moderate rate.
    cluster_decay_rates: dict = field(default_factory=lambda: {
        'IDENTITY_SELF_CONCEPT':  0.000543,   # Tier 1 (slow)
        'RELATIONAL_BONDS':       0.000543,   # Tier 1
        'HEALTH_WELLBEING':       0.001320,   # Tier 2 (medium)
        'INTELLECTUAL_INTERESTS': 0.001320,   # Tier 2
        'PROJECTS_ENDEAVORS':     0.001320,   # Tier 2
        'HOBBIES_RECREATION':     0.000349,   # Tier 3 (faster)
        'PREFERENCES_HABITS':     0.000349,   # Tier 3
        'FINANCIAL_MATERIAL':     0.000349,   # Tier 3
        'OBLIGATIONS':            0.000092,   # Tier 4 (fastest)
        'LOGISTICAL_CONTEXT':     0.000092,   # Tier 4
        'OTHER':                  0.000349,   # Tier 3
    })

    # --- Per-cluster sensitivity to three clocks ---
    # S[k] = (s_abs, s_rel, s_conv), sums to 1.0 per cluster.
    #
    # s_abs:  absolute wall-clock time (calendar drift, deadline proximity)
    # s_rel:  relative inter-session time (dormancy, relationship gaps)
    # s_conv: conversational time — messages in current session
    #
    # Design rationale:
    #   LOGISTICAL: almost pure absolute (0.8) — "meeting at 3pm" is calendar-anchored
    #   OBLIGATIONS: mostly absolute (0.7) — deadlines are calendar dates
    #   PREFERENCES: mostly absolute (0.6) — tastes drift by calendar months
    #   HEALTH: balanced absolute+relative (0.5/0.3) — conditions evolve on both scales
    #   PROJECTS: balanced (0.5/0.2/0.3) — milestones are calendar, but session context matters
    #   FINANCIAL: balanced (0.5/0.3/0.2) — prices are calendar, context is session
    #   IDENTITY: near-uniform (barely matters — rate is ~0 anyway)
    #   RELATIONAL: mostly relative (0.6) — "haven't talked to mom in 3 months" is a session gap
    #   HOBBIES: mostly relative (0.5) — dormancy is inter-session, not calendar
    #   INTELLECTUAL: balanced relative (0.4) — reactivation is session-driven
    clock_sensitivity: dict = field(default_factory=lambda: {
        'OBLIGATIONS':           (0.7, 0.1, 0.2),
        'RELATIONAL_BONDS':      (0.2, 0.6, 0.2),
        'HEALTH_WELLBEING':      (0.5, 0.3, 0.2),
        'IDENTITY_SELF_CONCEPT': (0.4, 0.3, 0.3),
        'HOBBIES_RECREATION':    (0.3, 0.5, 0.2),
        'PREFERENCES_HABITS':    (0.6, 0.2, 0.2),
        'INTELLECTUAL_INTERESTS': (0.3, 0.4, 0.3),
        'LOGISTICAL_CONTEXT':    (0.8, 0.1, 0.1),
        'PROJECTS_ENDEAVORS':    (0.5, 0.2, 0.3),
        'FINANCIAL_MATERIAL':    (0.5, 0.3, 0.2),
        'OTHER':                 (0.4, 0.3, 0.3),
    })

    # --- Conversational time normalization ---
    # Converts message count to "effective hours" for comparable scaling with real clocks.
    # 1 message ≈ conv_time_scale effective hours.
    conv_time_scale: float = 0.5

    # --- Anticipatory activation (Goal 3) ---
    anticipatory_enabled: bool = True
    anticipatory_window_hours: float = 168.0   # 7 days: activation starts rising
    anticipatory_peak_multiplier: float = 3.0  # At deadline: activation = base × 3
    anticipatory_post_decay_rate: float = 0.15 # Per-hour decay rate AFTER deadline passes

    # --- Reactivation ---
    reactivation_enabled: bool = True
    reactivation_threshold: float = 0.10   # Below this → fact is "dormant"
    reactivation_boost: float = 0.70       # On reactivation signal → set activation here
    reactivation_eligible: set = field(default_factory=lambda: {
        'HOBBIES_RECREATION', 'INTELLECTUAL_INTERESTS', 'RELATIONAL_BONDS',
    })

    # --- Emotional loading boost ---
    emotional_loading_enabled: bool = True
    emotional_boost_initial: float = 0.30   # Additive boost when emotion detected
    emotional_boost_decay_rate: float = 0.10  # Emotional boost fades fast (per hour)

    # --- Blending and clock mode (ablation switches) ---
    blending_mode: BlendingMode = BlendingMode.HARMONIC
    clock_mode: ClockMode = ClockMode.MULTI

    # --- Uniform decay override (Ablation 1) ---
    uniform_decay: bool = False
    uniform_decay_rate: float = 0.0030

    # --- Cognitive typology mode (Ablation 9) ---
    # When True, ignores behavioral categories and uses cognitive types instead.
    cognitive_typology: bool = False
    cognitive_decay_rates: dict = field(default_factory=lambda: {
        'EPISODIC':   0.050,   # Events fade fast
        'SEMANTIC':   0.005,   # Facts persist
        'PROCEDURAL': 0.003,   # Skills persist longer
    })
    # Cognitive types get a single clock (absolute only) since the cognitive
    # framework has no concept of per-type clock routing
    cognitive_clock_sensitivity: dict = field(default_factory=lambda: {
        'EPISODIC':   (0.7, 0.2, 0.1),
        'SEMANTIC':   (0.5, 0.3, 0.2),
        'PROCEDURAL': (0.4, 0.3, 0.3),
    })

    # --- 8+1 ontology mode (Ablation 12) ---
    # When True, collapses HOBBIES and PREFERENCES into IDENTITY
    ontology_8plus1: bool = False

    # --- Base activation for newly ingested facts ---
    base_activation: float = 1.0

    # --- Staleness floor ---
    # Facts below this activation are effectively dead for retrieval.
    staleness_floor: float = 0.01


@dataclass
class TemporalContext:
    """The three clocks, computed at retrieval time by the caller.

    The decay engine doesn't touch the database — the caller computes these
    from timestamps and passes them in.
    """
    absolute_hours: float          # Hours since fact's last_updated_ts
    relative_hours: float          # Hours since user's last session start
    conversational_messages: int   # Messages in current session since fact last touched

    # For anticipatory activation and emotional boost timing
    current_timestamp: Optional[float] = None  # Unix timestamp of "now"


@dataclass
class FactNode:
    """Minimal fact representation for decay computation.

    In production, these fields map onto graphiti_core EntityNode extensions.
    The decay engine only reads these fields — it never writes to the database.
    """
    fact_id: str
    membership_weights: dict    # {category: weight}, sums to 1.0
    primary_category: str
    last_updated_ts: float      # Unix timestamp of last update
    base_activation: float = 1.0

    # Anticipatory activation (Goal 3)
    future_anchor_ts: Optional[float] = None     # Deadline/event timestamp

    # Emotional loading
    emotional_loading: bool = False
    emotional_loading_ts: Optional[float] = None  # When the emotional signal was detected

    # Reactivation tracking
    last_reactivation_ts: Optional[float] = None
    access_count: int = 0


# ============================================================================
# Core engine
# ============================================================================

class DecayEngine:
    """Category-conditioned temporal decay with multi-clock routing.

    Pure math. No I/O, no LLM, no database. This is the retrieval-time hot path.

    The engine computes activation for a fact given the current temporal context.
    Activation determines retrieval ranking — higher activation = more relevant.

    Architecture:
        activation = base_decay(f, t) × anticipatory(f, t) + emotional_boost(f, t)

    Where:
        base_decay  = Σ_k w_k · exp(-λ_k · Δt_k)     [per-cluster, per-clock]
        anticipatory = multiplier for future-anchored facts approaching deadline
        emotional    = fast-decaying additive boost for emotionally-loaded facts
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
        self._validate_config()

    def _validate_config(self):
        """Sanity checks. Fail fast on misconfiguration."""
        cfg = self.config

        if not cfg.cognitive_typology and not cfg.uniform_decay:
            for cat in CATEGORIES:
                assert cat in cfg.cluster_decay_rates, f"Missing decay rate: {cat}"
                assert cat in cfg.clock_sensitivity, f"Missing clock sensitivity: {cat}"
                s = cfg.clock_sensitivity[cat]
                assert len(s) == 3, f"Clock sensitivity must have 3 values: {cat}"
                assert abs(sum(s) - 1.0) < 0.01, (
                    f"Clock sensitivity for {cat} sums to {sum(s):.3f}, expected 1.0"
                )
                assert cfg.cluster_decay_rates[cat] >= 0, (
                    f"Decay rate for {cat} must be >= 0"
                )

        if cfg.cognitive_typology:
            for ctype in COGNITIVE_TYPES:
                assert ctype in cfg.cognitive_decay_rates, f"Missing cognitive rate: {ctype}"

    # ----------------------------------------------------------------
    # Step 1: Resolve effective category weights
    # ----------------------------------------------------------------

    def _resolve_weights(self, fact: FactNode) -> dict:
        """Resolve effective membership weights, handling 8+1 collapse and cognitive mapping.

        Returns {category_or_type: weight} depending on mode.
        """
        cfg = self.config
        weights = dict(fact.membership_weights)

        if cfg.cognitive_typology:
            # Ablation 9: map behavioral categories → cognitive types
            cog_weights = {t: 0.0 for t in COGNITIVE_TYPES}
            for cat, w in weights.items():
                if w > 0 and cat in COGNITIVE_MAP:
                    cog_weights[COGNITIVE_MAP[cat]] += w
            # Renormalize
            total = sum(cog_weights.values())
            if total > 0:
                cog_weights = {t: w / total for t, w in cog_weights.items()}
            return cog_weights

        if cfg.ontology_8plus1:
            # Ablation 12: collapse HOBBIES + PREFERENCES → IDENTITY
            merged = {}
            for cat, w in weights.items():
                target = COLLAPSE_8PLUS1.get(cat, cat)
                merged[target] = merged.get(target, 0.0) + w
            # Renormalize
            total = sum(merged.values())
            if total > 0:
                merged = {c: w / total for c, w in merged.items()}
            return merged

        return weights

    def _resolve_primary(self, fact: FactNode) -> str:
        """Resolve primary category under current ontology mode."""
        cfg = self.config

        if cfg.cognitive_typology:
            return COGNITIVE_MAP.get(fact.primary_category, 'SEMANTIC')

        if cfg.ontology_8plus1:
            return COLLAPSE_8PLUS1.get(fact.primary_category, fact.primary_category)

        return fact.primary_category

    # ----------------------------------------------------------------
    # Step 2: Compute per-cluster temporal deltas (multi-clock routing)
    # ----------------------------------------------------------------

    def _get_decay_rate(self, category: str) -> float:
        """Get decay rate for a category or cognitive type."""
        cfg = self.config
        if cfg.cognitive_typology:
            return cfg.cognitive_decay_rates.get(category, 0.005)
        return cfg.cluster_decay_rates.get(category, cfg.uniform_decay_rate)

    def _get_clock_sensitivity(self, category: str) -> tuple:
        """Get (s_abs, s_rel, s_conv) for a category or cognitive type."""
        cfg = self.config
        if cfg.cognitive_typology:
            return cfg.cognitive_clock_sensitivity.get(category, (0.5, 0.3, 0.2))
        return cfg.clock_sensitivity.get(category, (0.4, 0.3, 0.3))

    def compute_effective_delta(self, category: str, ctx: TemporalContext) -> float:
        """Compute effective temporal delta for one category given the three clocks.

        Δt_k = s_abs × Δt_absolute + s_rel × Δt_relative + s_conv × (msgs × scale)
        """
        if self.config.clock_mode == ClockMode.ABSOLUTE_ONLY:
            return ctx.absolute_hours

        s_abs, s_rel, s_conv = self._get_clock_sensitivity(category)
        conv_hours = ctx.conversational_messages * self.config.conv_time_scale

        return s_abs * ctx.absolute_hours + s_rel * ctx.relative_hours + s_conv * conv_hours

    # ----------------------------------------------------------------
    # Step 3: Compute base decay (the core formula)
    # ----------------------------------------------------------------

    def compute_base_decay(self, fact: FactNode, ctx: TemporalContext) -> float:
        """Base activation after temporal decay.

        For each category k with nonzero membership w_k:
            cluster_activation_k = exp(-λ_k × Δt_k)

        Blending depends on mode:
            HARMONIC (default): per-cluster decay then blend
                A = Σ_k  w_k × exp(-λ_k × Δt_k)
            This is more principled than blending rates first because
            each cluster has its own temporal delta from multi-clock routing.

        The harmonic mean formula from the paper (λ_eff = Σw / Σ(w/λ)) is a special
        case that applies when all clusters share the same Δt. With multi-clock routing,
        per-cluster computation is strictly more accurate.
        """
        cfg = self.config

        # Ablation 1: uniform decay
        if cfg.uniform_decay:
            return fact.base_activation * math.exp(
                -cfg.uniform_decay_rate * ctx.absolute_hours
            )

        weights = self._resolve_weights(fact)
        primary = self._resolve_primary(fact)

        # Ablation 11: primary cluster only
        if cfg.blending_mode == BlendingMode.MAX_ONLY:
            rate = self._get_decay_rate(primary)
            delta = self.compute_effective_delta(primary, ctx)
            return fact.base_activation * math.exp(-rate * delta)

        # Per-cluster decay with blending
        if cfg.blending_mode == BlendingMode.HARMONIC:
            # Default: weighted harmonic mean of rates → single effective rate.
            # λ_eff = Σ w_k / Σ (w_k / λ_k)
            # Harmonic mean emphasizes slow-decaying clusters, preserving
            # long-lived facts in mixed-category edges.
            w_sum = 0.0
            w_over_lambda_sum = 0.0
            delta_weighted_sum = 0.0
            for cat, w in weights.items():
                if w <= 0:
                    continue
                rate = self._get_decay_rate(cat)
                delta = self.compute_effective_delta(cat, ctx)
                w_sum += w
                if rate > 0:
                    w_over_lambda_sum += w / rate
                delta_weighted_sum += w * delta

            if w_sum <= 0:
                return fact.base_activation

            lambda_eff = w_sum / w_over_lambda_sum if w_over_lambda_sum > 0 else cfg.uniform_decay_rate
            delta_eff = delta_weighted_sum / w_sum
            activation = math.exp(-lambda_eff * delta_eff)

        elif cfg.blending_mode == BlendingMode.ARITHMETIC:
            # Ablation 10: weighted arithmetic mean of per-cluster activations.
            # Each cluster independently decays with its own rate and delta,
            # then activations are averaged. Fast-decaying clusters pull the
            # average down more aggressively than harmonic blending.
            activation = 0.0
            total_weight = 0.0
            for cat, w in weights.items():
                if w <= 0:
                    continue
                rate = self._get_decay_rate(cat)
                delta = self.compute_effective_delta(cat, ctx)
                cluster_activation = math.exp(-rate * delta)
                activation += w * cluster_activation
                total_weight += w

            if total_weight > 0:
                activation /= total_weight
            else:
                activation = 0.0
        else:
            activation = 0.0

        return fact.base_activation * activation

    def compute_blended_rate(self, fact: FactNode) -> float:
        """Compute the effective blended decay rate (for reporting/debugging).

        This is the paper formula: λ_eff = Σ_k w_k / Σ_k (w_k / λ_k)
        Note: this is an approximation when multi-clock is active (each cluster
        has a different effective Δt). Use compute_base_decay for actual computation.
        """
        cfg = self.config

        if cfg.uniform_decay:
            return cfg.uniform_decay_rate

        weights = self._resolve_weights(fact)

        if cfg.blending_mode == BlendingMode.MAX_ONLY:
            primary = self._resolve_primary(fact)
            return self._get_decay_rate(primary)

        numerator = 0.0
        denominator = 0.0
        for cat, w in weights.items():
            if w <= 0:
                continue
            rate = self._get_decay_rate(cat)
            numerator += w
            if rate > 0:
                denominator += w / rate

        if denominator <= 0:
            return cfg.uniform_decay_rate

        return numerator / denominator

    # ----------------------------------------------------------------
    # Step 4: Anticipatory activation (Goal 3)
    # ----------------------------------------------------------------

    def compute_anticipatory(self, fact: FactNode, ctx: TemporalContext) -> tuple:
        """Anticipatory activation for future-anchored facts.

        Returns (is_active: bool, activation_override: float).

        When is_active is True, the returned activation_override REPLACES base decay
        entirely (not multiplied). This is critical: an obligation created 200 hours ago
        would normally have base ≈ 0 from exponential decay, but if its deadline is
        approaching, it should have RISING activation. Multiplying 0 × 3 = 0; we need
        to override to the anticipatory curve value.

        Pre-deadline window:  linear ramp from base_floor → peak.
        At deadline:          peak activation.
        Post-deadline:        rapid exponential decay from peak.
        Outside window:       (False, 0) — use normal base decay.

        Only applies to OBLIGATIONS and PROJECTS_ENDEAVORS.
        """
        cfg = self.config

        if not cfg.anticipatory_enabled:
            return (False, 0.0)

        if fact.future_anchor_ts is None or ctx.current_timestamp is None:
            return (False, 0.0)

        # Only for deadline-bearing categories (checked on original, not resolved)
        if fact.primary_category not in ('OBLIGATIONS', 'PROJECTS_ENDEAVORS'):
            return (False, 0.0)

        hours_until = (fact.future_anchor_ts - ctx.current_timestamp) / 3600.0
        window = cfg.anticipatory_window_hours
        peak = cfg.anticipatory_peak_multiplier

        if hours_until > window:
            # Deadline too far away — no anticipatory effect
            return (False, 0.0)
        elif hours_until > 0:
            # Inside window, approaching deadline: ramp from floor → peak
            progress = 1.0 - (hours_until / window)  # 0 at window edge → 1 at deadline
            # Floor ensures the fact is meaningfully active even at window edge
            floor = 0.3
            activation = floor + (peak - floor) * progress
            return (True, fact.base_activation * activation)
        else:
            # Past deadline: rapid exponential decay from peak
            hours_past = -hours_until
            decayed = peak * math.exp(-cfg.anticipatory_post_decay_rate * hours_past)
            if decayed < cfg.staleness_floor:
                return (False, 0.0)  # Fully decayed post-deadline, revert to base
            return (True, fact.base_activation * decayed)

    # ----------------------------------------------------------------
    # Step 5: Emotional loading boost
    # ----------------------------------------------------------------

    def compute_emotional_boost(self, fact: FactNode, ctx: TemporalContext) -> float:
        """Emotional loading: fast-decaying additive boost.

        Returns >= 0.0 (0.0 = no effect).

        Emotional loading is a SIGNAL, not a category. Frustration about a deadline
        boosts the OBLIGATIONS cluster activation, not a separate "emotion" bucket.
        """
        cfg = self.config

        if not cfg.emotional_loading_enabled:
            return 0.0
        if not fact.emotional_loading or fact.emotional_loading_ts is None:
            return 0.0
        if ctx.current_timestamp is None:
            return 0.0

        hours_since = (ctx.current_timestamp - fact.emotional_loading_ts) / 3600.0
        if hours_since < 0:
            return 0.0

        return cfg.emotional_boost_initial * math.exp(
            -cfg.emotional_boost_decay_rate * hours_since
        )

    # ----------------------------------------------------------------
    # Step 6: Full activation (main entry point)
    # ----------------------------------------------------------------

    def compute_activation(self, fact: FactNode, ctx: TemporalContext) -> float:
        """Compute full activation for a fact. THIS IS THE MAIN ENTRY POINT.

        Logic:
            1. Check anticipatory activation (deadline-bearing facts).
               If active, it OVERRIDES base decay entirely.
            2. Otherwise, compute base decay from behavioral ontology.
            3. Add emotional loading boost (additive, fast-decaying).
            4. Clamp to valid range.

        Returns: activation in [0, max_possible].
        """
        # Step 1: Anticipatory activation (overrides base if active)
        anticipatory_active, anticipatory_value = self.compute_anticipatory(fact, ctx)

        if anticipatory_active:
            base = anticipatory_value
        else:
            # Step 2: Normal category-conditioned decay
            base = self.compute_base_decay(fact, ctx)

        # Step 3: Emotional loading boost (always additive)
        emotional = self.compute_emotional_boost(fact, ctx)
        activation = base + emotional

        # Step 4: Clamp
        max_possible = (
            self.config.base_activation
            * self.config.anticipatory_peak_multiplier
            + self.config.emotional_boost_initial
        )
        activation = max(0.0, min(activation, max_possible))

        # Below staleness floor → effectively dead
        if activation < self.config.staleness_floor:
            activation = 0.0

        return activation

    # ----------------------------------------------------------------
    # Reactivation
    # ----------------------------------------------------------------

    def is_dormant(self, fact: FactNode, ctx: TemporalContext) -> bool:
        """Check if a fact is dormant (activation below reactivation threshold).

        Caller is responsible for checking if a relevance signal exists.
        If dormant AND relevance signal → call reactivate().
        """
        if not self.config.reactivation_enabled:
            return False
        if fact.primary_category not in self.config.reactivation_eligible:
            return False
        return self.compute_activation(fact, ctx) < self.config.reactivation_threshold

    def reactivate(self, fact: FactNode, current_ts: float) -> float:
        """Reactivate a dormant fact. Returns new activation.

        Updates fact's tracking fields in-place.
        Caller should persist these changes to the graph.
        """
        fact.last_reactivation_ts = current_ts
        fact.access_count += 1
        fact.base_activation = self.config.reactivation_boost
        fact.last_updated_ts = current_ts
        return self.config.reactivation_boost

    # ----------------------------------------------------------------
    # Batch operations (retrieval helpers)
    # ----------------------------------------------------------------

    def rank_facts(self, facts: list, ctx: TemporalContext) -> list:
        """Compute activation for all facts, return sorted descending.

        Returns: [(fact, activation), ...] sorted by activation desc.
        """
        scored = [(f, self.compute_activation(f, ctx)) for f in facts]
        scored.sort(key=lambda x: -x[1])
        return scored

    def filter_stale(self, facts: list, ctx: TemporalContext,
                     threshold: float = 0.05) -> list:
        """Return only facts with activation >= threshold, sorted descending."""
        ranked = self.rank_facts(facts, ctx)
        return [(f, a) for f, a in ranked if a >= threshold]

    def top_k(self, facts: list, ctx: TemporalContext, k: int = 10) -> list:
        """Return top-k facts by activation."""
        return self.rank_facts(facts, ctx)[:k]

    # ----------------------------------------------------------------
    # Diagnostics (for debugging and paper figures)
    # ----------------------------------------------------------------

    def activation_curve(self, fact: FactNode, hours_range: range,
                         session_gap_hours: float = 24.0,
                         session_messages: int = 0) -> list:
        """Compute activation over a time range. For plotting decay curves.

        Returns: [(hours, activation), ...]
        """
        base_ts = fact.last_updated_ts
        points = []
        for h in hours_range:
            ctx = TemporalContext(
                absolute_hours=float(h),
                relative_hours=session_gap_hours,
                conversational_messages=session_messages,
                current_timestamp=base_ts + h * 3600,
            )
            points.append((h, self.compute_activation(fact, ctx)))
        return points

    def anticipatory_curve(self, fact: FactNode, hours_before: int = 200,
                           hours_after: int = 48) -> list:
        """Plot anticipatory activation curve around a deadline.

        Returns: [(hours_relative_to_deadline, activation), ...]
        Negative = before deadline, positive = after.
        """
        if fact.future_anchor_ts is None:
            return []

        deadline = fact.future_anchor_ts
        points = []
        for h in range(-hours_before, hours_after + 1):
            now = deadline + h * 3600
            hours_since_creation = (now - fact.last_updated_ts) / 3600.0
            if hours_since_creation < 0:
                continue
            ctx = TemporalContext(
                absolute_hours=hours_since_creation,
                relative_hours=24.0,
                conversational_messages=0,
                current_timestamp=now,
            )
            points.append((h, self.compute_activation(fact, ctx)))
        return points

    def category_report(self, fact: FactNode) -> dict:
        """Diagnostic: per-category breakdown for a single fact.

        Returns {category: {weight, rate, blended_rate_contribution}}.
        """
        weights = self._resolve_weights(fact)
        report = {}
        for cat, w in weights.items():
            if w <= 0:
                continue
            rate = self._get_decay_rate(cat)
            report[cat] = {
                'weight': round(w, 4),
                'decay_rate_per_hour': rate,
                'half_life_hours': round(math.log(2) / rate, 1) if rate > 0 else float('inf'),
            }
        report['_blended_rate'] = round(self.compute_blended_rate(fact), 6)
        blended_rate = self.compute_blended_rate(fact)
        report['_blended_half_life_hours'] = (
            round(math.log(2) / blended_rate, 1) if blended_rate > 0 else float('inf')
        )
        return report

    # ----------------------------------------------------------------
    # Ablation presets
    # ----------------------------------------------------------------

    @classmethod
    def default(cls) -> 'DecayEngine':
        """Full system: behavioral ontology, multi-clock, all features."""
        return cls(DecayConfig())

    @classmethod
    def uniform(cls) -> 'DecayEngine':
        """Ablation 1: Uniform decay rate across all categories."""
        return cls(DecayConfig(uniform_decay=True))

    @classmethod
    def single_clock(cls) -> 'DecayEngine':
        """Ablation 2: Single absolute timestamp, no multi-clock routing."""
        return cls(DecayConfig(clock_mode=ClockMode.ABSOLUTE_ONLY))

    @classmethod
    def no_anticipatory(cls) -> 'DecayEngine':
        """Ablation 3: No anticipatory activation for future-anchored facts."""
        return cls(DecayConfig(anticipatory_enabled=False))

    @classmethod
    def no_emotional(cls) -> 'DecayEngine':
        """Ablation 8: No emotional loading boost."""
        return cls(DecayConfig(emotional_loading_enabled=False))

    @classmethod
    def cognitive(cls) -> 'DecayEngine':
        """Ablation 9: Cognitive typology (episodic/semantic/procedural) replaces behavioral.

        THE critical ablation. If this matches behavioral ontology performance,
        the core thesis is disproven.
        """
        return cls(DecayConfig(cognitive_typology=True))

    @classmethod
    def arithmetic_blend(cls) -> 'DecayEngine':
        """Ablation 10: Arithmetic mean blending instead of harmonic mean."""
        return cls(DecayConfig(blending_mode=BlendingMode.ARITHMETIC))

    @classmethod
    def max_only(cls) -> 'DecayEngine':
        """Ablation 11: Primary cluster only, no soft blending."""
        return cls(DecayConfig(blending_mode=BlendingMode.MAX_ONLY))

    @classmethod
    def ontology_8plus1(cls) -> 'DecayEngine':
        """Ablation 12: Original 8+1 ontology (HOBBIES + PREFERENCES collapse into IDENTITY)."""
        return cls(DecayConfig(ontology_8plus1=True))


# ============================================================================
# Unit tests (Goal 2 requirement: verify decay ordering invariants)
# ============================================================================

def _make_fact(primary, weights=None, future_anchor=None, emotional=False,
               ts=0.0, emotional_ts=None):
    """Helper: create a FactNode for testing."""
    if weights is None:
        weights = {primary: 1.0}
    # Fill missing categories with 0
    for cat in CATEGORIES:
        weights.setdefault(cat, 0.0)
    return FactNode(
        fact_id=f"test_{primary}",
        membership_weights=weights,
        primary_category=primary,
        last_updated_ts=ts,
        future_anchor_ts=future_anchor,
        emotional_loading=emotional,
        emotional_loading_ts=emotional_ts,
    )


def run_tests():
    """Run all decay engine invariant tests. Call this to validate."""

    engine = DecayEngine.default()
    passed = 0
    failed = 0
    total = 0

    def check(name, condition):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            failed += 1
            print(f"  ❌ {name}")

    now = 1_000_000.0  # arbitrary reference timestamp

    # --- Test 1: Health decays slower than Logistical ---
    print("\n[Test 1] Health fact decays slower than Logistical fact")
    health = _make_fact('HEALTH_WELLBEING', ts=now)
    logistical = _make_fact('LOGISTICAL_CONTEXT', ts=now)
    ctx_24h = TemporalContext(
        absolute_hours=24.0, relative_hours=24.0,
        conversational_messages=0, current_timestamp=now + 24 * 3600,
    )
    a_health = engine.compute_activation(health, ctx_24h)
    a_logistical = engine.compute_activation(logistical, ctx_24h)
    check(f"Health ({a_health:.4f}) > Logistical ({a_logistical:.4f}) after 24h",
          a_health > a_logistical)

    # --- Test 2: Identity decays slowest ---
    print("\n[Test 2] Identity fact decays slowest of all categories")
    identity = _make_fact('IDENTITY_SELF_CONCEPT', ts=now)
    ctx_720h = TemporalContext(
        absolute_hours=720.0, relative_hours=720.0,
        conversational_messages=0, current_timestamp=now + 720 * 3600,
    )
    a_identity = engine.compute_activation(identity, ctx_720h)
    for cat in CATEGORIES:
        if cat == 'IDENTITY_SELF_CONCEPT':
            continue
        other = _make_fact(cat, ts=now)
        a_other = engine.compute_activation(other, ctx_720h)
        check(f"Identity ({a_identity:.4f}) > {cat} ({a_other:.4f}) after 30 days",
              a_identity > a_other)

    # --- Test 3: Hobby reactivates after dormancy ---
    print("\n[Test 3] Hobby fact is dormant after long gap, can reactivate")
    hobby = _make_fact('HOBBIES_RECREATION', ts=now)
    ctx_long = TemporalContext(
        absolute_hours=2000.0, relative_hours=2000.0,
        conversational_messages=0, current_timestamp=now + 2000 * 3600,
    )
    check(f"Hobby is dormant after 2000h", engine.is_dormant(hobby, ctx_long))
    new_activation = engine.reactivate(hobby, now + 2000 * 3600)
    check(f"Reactivated to {new_activation:.2f}", new_activation >= 0.5)

    # --- Test 4: Preference supersedes faster than Identity ---
    print("\n[Test 4] Preference decays faster than Identity")
    pref = _make_fact('PREFERENCES_HABITS', ts=now)
    ident = _make_fact('IDENTITY_SELF_CONCEPT', ts=now)
    ctx_7d = TemporalContext(
        absolute_hours=168.0, relative_hours=168.0,
        conversational_messages=0, current_timestamp=now + 168 * 3600,
    )
    a_pref = engine.compute_activation(pref, ctx_7d)
    a_ident = engine.compute_activation(ident, ctx_7d)
    check(f"Identity ({a_ident:.4f}) > Preference ({a_pref:.4f}) after 7 days",
          a_ident > a_pref)

    # --- Test 5: Multi-cluster fact decays at blended rate ---
    print("\n[Test 5] Multi-cluster fact decays between its component rates")
    # Bike helmet: 0.5 FINANCIAL + 0.4 HOBBIES + 0.1 LOGISTICAL
    multi = _make_fact('FINANCIAL_MATERIAL', weights={
        'FINANCIAL_MATERIAL': 0.5, 'HOBBIES_RECREATION': 0.4, 'LOGISTICAL_CONTEXT': 0.1,
    }, ts=now)
    pure_financial = _make_fact('FINANCIAL_MATERIAL', ts=now)
    pure_hobby = _make_fact('HOBBIES_RECREATION', ts=now)
    a_multi = engine.compute_activation(multi, ctx_7d)
    a_fin = engine.compute_activation(pure_financial, ctx_7d)
    a_hob = engine.compute_activation(pure_hobby, ctx_7d)
    check(f"Multi ({a_multi:.4f}) is between Financial ({a_fin:.4f}) and Hobby ({a_hob:.4f})",
          min(a_fin, a_hob) <= a_multi <= max(a_fin, a_hob))

    # --- Test 6: Lazy computation matches continuous ---
    print("\n[Test 6] Lazy computation consistency")
    fact = _make_fact('RELATIONAL_BONDS', ts=now)
    # Compute at 24h directly
    ctx_direct = TemporalContext(
        absolute_hours=24.0, relative_hours=24.0,
        conversational_messages=0, current_timestamp=now + 24 * 3600,
    )
    a_direct = engine.compute_activation(fact, ctx_direct)
    # Compute at 12h, then at 24h (should give same result since decay is memoryless)
    a_direct2 = engine.compute_activation(fact, ctx_direct)
    check(f"Same fact+context gives same activation: {a_direct:.6f} == {a_direct2:.6f}",
          abs(a_direct - a_direct2) < 1e-10)

    # --- Test 7: Anticipatory activation ---
    print("\n[Test 7] Anticipatory activation rises before deadline, drops after")
    deadline_ts = now + 200 * 3600  # 200 hours from now
    obligation = _make_fact('OBLIGATIONS', ts=now, future_anchor=deadline_ts)

    # 7 days before deadline
    ctx_7d_before = TemporalContext(
        absolute_hours=32.0, relative_hours=24.0,
        conversational_messages=0, current_timestamp=deadline_ts - 168 * 3600,
    )
    # 1 hour before deadline
    ctx_1h_before = TemporalContext(
        absolute_hours=199.0, relative_hours=24.0,
        conversational_messages=0, current_timestamp=deadline_ts - 1 * 3600,
    )
    # 24 hours after deadline
    ctx_24h_after = TemporalContext(
        absolute_hours=224.0, relative_hours=24.0,
        conversational_messages=0, current_timestamp=deadline_ts + 24 * 3600,
    )

    a_7d = engine.compute_activation(obligation, ctx_7d_before)
    a_1h = engine.compute_activation(obligation, ctx_1h_before)
    a_post = engine.compute_activation(obligation, ctx_24h_after)
    check(f"1h before ({a_1h:.4f}) > 7d before ({a_7d:.4f})", a_1h > a_7d)
    check(f"1h before ({a_1h:.4f}) > 24h after ({a_post:.4f})", a_1h > a_post)

    # --- Test 8: Emotional boost is additive and fast-decaying ---
    print("\n[Test 8] Emotional loading boost")
    plain = _make_fact('OBLIGATIONS', ts=now)
    emotional = _make_fact('OBLIGATIONS', ts=now, emotional=True, emotional_ts=now)
    ctx_1h = TemporalContext(
        absolute_hours=1.0, relative_hours=1.0,
        conversational_messages=0, current_timestamp=now + 3600,
    )
    ctx_48h = TemporalContext(
        absolute_hours=48.0, relative_hours=48.0,
        conversational_messages=0, current_timestamp=now + 48 * 3600,
    )
    a_plain = engine.compute_activation(plain, ctx_1h)
    a_emo_1h = engine.compute_activation(emotional, ctx_1h)
    a_emo_48h = engine.compute_activation(emotional, ctx_48h)
    check(f"Emotional ({a_emo_1h:.4f}) > Plain ({a_plain:.4f}) at 1h",
          a_emo_1h > a_plain)
    check(f"Emotional boost fades: 1h ({a_emo_1h:.4f}) > 48h ({a_emo_48h:.4f})",
          a_emo_1h > a_emo_48h)

    # --- Test 9: Ablation 1 — uniform decay makes all categories equal ---
    print("\n[Test 9] Ablation 1: Uniform decay")
    eng_uniform = DecayEngine.uniform()
    a_health_u = eng_uniform.compute_activation(health, ctx_24h)
    a_logist_u = eng_uniform.compute_activation(logistical, ctx_24h)
    check(f"Uniform: Health ({a_health_u:.4f}) ≈ Logistical ({a_logist_u:.4f})",
          abs(a_health_u - a_logist_u) < 0.001)

    # --- Test 10: Ablation 9 — cognitive typology collapses behavioral distinctions ---
    print("\n[Test 10] Ablation 9: Cognitive typology")
    eng_cog = DecayEngine.cognitive()
    # Health and Preferences are both SEMANTIC under cognitive typology
    a_health_c = eng_cog.compute_activation(health, ctx_7d)
    a_pref_c = eng_cog.compute_activation(pref, ctx_7d)
    check(f"Cognitive: Health ({a_health_c:.4f}) ≈ Preference ({a_pref_c:.4f}) (both SEMANTIC)",
          abs(a_health_c - a_pref_c) < 0.001)
    # Confirm behavioral ontology DOES distinguish them
    a_health_b = engine.compute_activation(health, ctx_7d)
    a_pref_b = engine.compute_activation(pref, ctx_7d)
    check(f"Behavioral: Health ({a_health_b:.4f}) ≠ Preference ({a_pref_b:.4f})",
          abs(a_health_b - a_pref_b) > 0.01)

    # --- Test 11: Ablation 12 — 8+1 collapses Hobbies/Preferences into Identity ---
    print("\n[Test 11] Ablation 12: 8+1 ontology collapse")
    eng_8 = DecayEngine.ontology_8plus1()
    hobby_fact = _make_fact('HOBBIES_RECREATION', ts=now)
    pref_fact = _make_fact('PREFERENCES_HABITS', ts=now)
    ident_fact = _make_fact('IDENTITY_SELF_CONCEPT', ts=now)
    a_hob_8 = eng_8.compute_activation(hobby_fact, ctx_7d)
    a_pref_8 = eng_8.compute_activation(pref_fact, ctx_7d)
    a_ident_8 = eng_8.compute_activation(ident_fact, ctx_7d)
    check(f"8+1: Hobby ({a_hob_8:.4f}) ≈ Identity ({a_ident_8:.4f}) (collapsed)",
          abs(a_hob_8 - a_ident_8) < 0.001)
    check(f"8+1: Preference ({a_pref_8:.4f}) ≈ Identity ({a_ident_8:.4f}) (collapsed)",
          abs(a_pref_8 - a_ident_8) < 0.001)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Tests: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All invariants hold. Decay engine is ready.")
    else:
        print(f"⚠️  {failed} invariant(s) violated — investigate before proceeding.")
    print(f"{'='*60}")

    return failed == 0


if __name__ == '__main__':
    run_tests()