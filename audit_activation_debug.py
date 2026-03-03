r"""
audit_activation_debug.py — Activation compression diagnostic.

Confirms that all Priya edges have `created_at` ≈ ingestion time (not the
persona's session dates), causing activation scores to compress to 0.96-0.99
regardless of the 17-month persona timeline.

Usage:
    python audit_activation_debug.py
"""

import asyncio
import json
import os
import sys
import time as time_module
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent

def load_env():
    for env_path in [PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"]:
        if not env_path.exists():
            continue
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

load_env()

from neo4j import AsyncGraphDatabase
from decay_engine import DecayEngine, DecayConfig, TemporalContext, FactNode, CATEGORIES

sys.path.insert(0, str(PROJECT_ROOT))
from evaluate_lifemembench import _to_unix_ts, _edge_attrs_to_fact_node, load_edge_cache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERSONA = "priya"
GROUP_ID = "lifemembench_priya"
MANIFEST_PATH = PROJECT_ROOT / "LifeMemEval" / "1_priya" / "sessions" / "manifest.json"
SAMPLES_PER_CATEGORY = 2


async def main():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    print("=" * 80)
    print("ACTIVATION COMPRESSION DIAGNOSTIC")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load edge cache
    # ------------------------------------------------------------------
    print("\n[1] Loading edge cache...")
    edge_cache = await load_edge_cache(driver, GROUP_ID)
    print(f"    Loaded {len(edge_cache)} enriched edges for {PERSONA}")

    # ------------------------------------------------------------------
    # 2. created_at_ts statistics
    # ------------------------------------------------------------------
    print("\n[2] created_at_ts statistics (ingestion timestamps)")
    timestamps = [v["created_at_ts"] for v in edge_cache.values() if v.get("created_at_ts", 0) > 0]
    if timestamps:
        ts_min, ts_max = min(timestamps), max(timestamps)
        ts_mean = sum(timestamps) / len(timestamps)
        print(f"    Count:  {len(timestamps)}")
        print(f"    Min:    {datetime.fromtimestamp(ts_min, tz=timezone.utc).isoformat()} ({ts_min:.0f})")
        print(f"    Max:    {datetime.fromtimestamp(ts_max, tz=timezone.utc).isoformat()} ({ts_max:.0f})")
        print(f"    Mean:   {datetime.fromtimestamp(ts_mean, tz=timezone.utc).isoformat()} ({ts_mean:.0f})")
        print(f"    Spread: {(ts_max - ts_min) / 3600:.1f} hours")
    else:
        print("    No timestamps found!")

    # ------------------------------------------------------------------
    # 3. Session date range from manifest
    # ------------------------------------------------------------------
    print("\n[3] Session date range from manifest.json")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    session_dates = [s["date"] for s in manifest["sessions"]]
    print(f"    Time span: {manifest.get('time_span', 'N/A')}")
    print(f"    First session: {session_dates[0]}")
    print(f"    Last session:  {session_dates[-1]}")
    first_ts = datetime.strptime(session_dates[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    last_ts = datetime.strptime(session_dates[-1], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    print(f"    Session span:  {(last_ts - first_ts) / 3600:.0f} hours ({(last_ts - first_ts) / 3600 / 24:.0f} days)")

    # ------------------------------------------------------------------
    # 4. Compute activations for sample edges
    # ------------------------------------------------------------------
    print("\n[4] Sample edge activations (2 per category)")
    print("-" * 80)

    t_now = time_module.time()
    behavioral_engine = DecayEngine()  # default behavioral
    uniform_engine = DecayEngine.uniform()

    ctx = TemporalContext(
        absolute_hours=0.0,  # placeholder, overridden per edge
        relative_hours=720.0,
        conversational_messages=0,
        current_timestamp=t_now,
    )

    # Group edges by category
    by_category: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for uuid, attrs in edge_cache.items():
        cat = attrs.get("primary_category", "OTHER")
        by_category[cat].append((uuid, attrs))

    all_behavioral_activations = []
    all_uniform_activations = []
    category_activations: dict[str, list[float]] = defaultdict(list)

    fmt = "{:<12} {:<40} {:>10} {:>10} {:>8} {:>8} {:>8}"
    print(fmt.format("Category", "Fact (truncated)", "created_at", "abs_hours", "λ_eff", "behav.", "unif."))
    print("-" * 80)

    for cat in CATEGORIES:
        edges = by_category.get(cat, [])
        samples = edges[:SAMPLES_PER_CATEGORY]
        for uuid, attrs in samples:
            fact_node = _edge_attrs_to_fact_node(attrs, uuid)
            created_ts = attrs.get("created_at_ts", 0.0) or 0.0
            abs_hours = (t_now - created_ts) / 3600.0 if created_ts > 0 else 0.0

            behav_act = compute_edge_activation_inline(uuid, ctx, behavioral_engine, edge_cache)
            unif_act = compute_edge_activation_inline(uuid, ctx, uniform_engine, edge_cache)

            all_behavioral_activations.append(behav_act)
            all_uniform_activations.append(unif_act)
            category_activations[cat].append(behav_act)

            # Get effective lambda for primary category
            cfg = behavioral_engine.config
            lambda_eff = cfg.cluster_decay_rates.get(cat, 0.005)

            fact_trunc = (attrs.get("fact", "") or "")[:38]
            created_str = datetime.fromtimestamp(created_ts, tz=timezone.utc).strftime("%m/%d %H:%M") if created_ts > 0 else "N/A"
            print(fmt.format(
                cat[:12], fact_trunc, created_str,
                f"{abs_hours:.1f}", f"{lambda_eff:.4f}",
                f"{behav_act:.4f}", f"{unif_act:.4f}",
            ))

    # ------------------------------------------------------------------
    # 5. Aggregate statistics
    # ------------------------------------------------------------------
    print("\n[5] Activation statistics (all edges)")
    print("-" * 60)

    # Compute all activations
    for uuid in edge_cache:
        if uuid not in {u for edges in by_category.values() for u, _ in edges[:SAMPLES_PER_CATEGORY]}:
            behav_act = compute_edge_activation_inline(uuid, ctx, behavioral_engine, edge_cache)
            unif_act = compute_edge_activation_inline(uuid, ctx, uniform_engine, edge_cache)
            all_behavioral_activations.append(behav_act)
            all_uniform_activations.append(unif_act)
            cat = edge_cache[uuid].get("primary_category", "OTHER")
            category_activations[cat].append(behav_act)

    print(f"  Behavioral engine:")
    print(f"    Min: {min(all_behavioral_activations):.4f}  Max: {max(all_behavioral_activations):.4f}  Mean: {sum(all_behavioral_activations)/len(all_behavioral_activations):.4f}")
    print(f"  Uniform engine:")
    print(f"    Min: {min(all_uniform_activations):.4f}  Max: {max(all_uniform_activations):.4f}  Mean: {sum(all_uniform_activations)/len(all_uniform_activations):.4f}")

    print("\n  Per-category (behavioral):")
    fmt2 = "    {:<28} n={:<4} min={:.4f}  max={:.4f}  mean={:.4f}"
    for cat in CATEGORIES:
        acts = category_activations.get(cat, [])
        if acts:
            print(fmt2.format(cat, len(acts), min(acts), max(acts), sum(acts)/len(acts)))

    # ------------------------------------------------------------------
    # 6. What-if: activations with session dates
    # ------------------------------------------------------------------
    print("\n[6] WHAT-IF: activations if created_at used session dates")
    print("-" * 60)

    # Map edges to hypothetical session dates spread across the timeline
    # Use first and last session dates as range
    hypothetical_activations_behav = []
    hypothetical_activations_unif = []
    n_edges = len(edge_cache)
    edge_list = list(edge_cache.items())

    # Spread edges evenly across session date range
    for i, (uuid, attrs) in enumerate(edge_list):
        # Assign hypothetical timestamp spread across the persona timeline
        frac = i / max(n_edges - 1, 1)
        hypo_ts = first_ts + frac * (last_ts - first_ts)
        hypo_abs_hours = (t_now - hypo_ts) / 3600.0

        fact_node = _edge_attrs_to_fact_node(attrs, uuid)
        # Override last_updated_ts for this computation
        fact_node_hypo = FactNode(
            fact_id=uuid,
            membership_weights=fact_node.membership_weights,
            primary_category=fact_node.primary_category,
            last_updated_ts=hypo_ts,
            base_activation=1.0,
            future_anchor_ts=None,
            emotional_loading=False,
            emotional_loading_ts=None,
            last_reactivation_ts=None,
            access_count=0,
        )
        hypo_ctx = TemporalContext(
            absolute_hours=max(0.0, hypo_abs_hours),
            relative_hours=720.0,
            conversational_messages=0,
            current_timestamp=t_now,
        )
        behav_act = behavioral_engine.compute_activation(fact_node_hypo, hypo_ctx)
        unif_act = uniform_engine.compute_activation(fact_node_hypo, hypo_ctx)
        hypothetical_activations_behav.append(behav_act)
        hypothetical_activations_unif.append(unif_act)

    print(f"  Behavioral engine (hypothetical session dates):")
    print(f"    Min: {min(hypothetical_activations_behav):.4f}  Max: {max(hypothetical_activations_behav):.4f}  Mean: {sum(hypothetical_activations_behav)/len(hypothetical_activations_behav):.4f}")
    print(f"  Uniform engine (hypothetical session dates):")
    print(f"    Min: {min(hypothetical_activations_unif):.4f}  Max: {max(hypothetical_activations_unif):.4f}  Mean: {sum(hypothetical_activations_unif)/len(hypothetical_activations_unif):.4f}")

    print(f"\n  Spread with actual timestamps:      {max(all_behavioral_activations) - min(all_behavioral_activations):.4f}")
    print(f"  Spread with session-date timestamps: {max(hypothetical_activations_behav) - min(hypothetical_activations_behav):.4f}")
    print(f"  Improvement factor: {(max(hypothetical_activations_behav) - min(hypothetical_activations_behav)) / max(max(all_behavioral_activations) - min(all_behavioral_activations), 1e-9):.1f}x")

    print("\n" + "=" * 80)
    print("CONCLUSION: If activation spread is near-zero with actual timestamps but")
    print("significant with session dates, the root cause is confirmed: e.created_at")
    print("reflects ingestion time, not the persona's session reference_time.")
    print("=" * 80)

    await driver.close()


def compute_edge_activation_inline(
    edge_uuid: str,
    ctx: TemporalContext,
    engine: DecayEngine,
    edge_cache: dict[str, dict],
) -> float:
    """Compute activation (mirrors evaluate_lifemembench.compute_edge_activation)."""
    attrs = edge_cache.get(edge_uuid)
    if attrs is None:
        return 0.5

    if attrs.get("is_world_knowledge"):
        return 0.0

    fact_node = _edge_attrs_to_fact_node(attrs, edge_uuid)

    if ctx.current_timestamp and fact_node.last_updated_ts > 0:
        abs_hours = (ctx.current_timestamp - fact_node.last_updated_ts) / 3600.0
    else:
        abs_hours = 0.0

    fact_ctx = TemporalContext(
        absolute_hours=max(0.0, abs_hours),
        relative_hours=ctx.relative_hours,
        conversational_messages=ctx.conversational_messages,
        current_timestamp=ctx.current_timestamp,
    )
    return engine.compute_activation(fact_node, fact_ctx)


if __name__ == "__main__":
    asyncio.run(main())
