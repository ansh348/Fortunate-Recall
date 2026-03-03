"""
Statistical significance tests for the alpha-sweep retrieval experiment.

Reads the 9 fine-grained JSON result files (alpha 0.000–0.200) and compares
per-question reciprocal ranks across engines using:
  1. Paired Wilcoxon signed-rank test
  2. Bootstrap 95% CI on ΔMRR
  3. Cohen's d
"""

import json
import pathlib

import numpy as np
from scipy.stats import wilcoxon

# ── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = pathlib.Path(__file__).parent / "LongMemEval" / "data" / "full_artifacts"

ALPHAS = [0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]

ENGINES = ["behavioral", "uniform", "cognitive"]

N_BOOTSTRAP = 10_000
RNG_SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────────────

def alpha_to_filename(alpha: float) -> str:
    """0.025 → 'kill_gate_results_v4_alpha_0_025.json'"""
    s = f"{alpha:.3f}".replace(".", "_")
    return f"kill_gate_results_v4_alpha_{s}.json"


def extract_rr(questions: list[dict]) -> np.ndarray:
    """Return per-question reciprocal-rank array for answerable questions."""
    rrs = []
    for q in questions:
        if not q.get("answerable"):
            continue
        rank = q.get("answer_rank")
        rrs.append(1.0 / rank if rank is not None else 0.0)
    return np.array(rrs)


def bootstrap_ci(diff: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED):
    """Percentile bootstrap 95% CI on the mean of diff."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(diff)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(means)), float(lo), float(hi)


def cohens_d(diff: np.ndarray) -> float:
    std = diff.std(ddof=1)
    if std == 0:
        return 0.0
    return float(diff.mean() / std)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Header
    hdr = (
        f"{'alpha':>7s}  "
        f"{'B-MRR':>7s}  {'U-MRR':>7s}  {'C-MRR':>7s}  "
        f"{'dMRR(B-U)':>10s}  {'p(Wilcox)':>10s}  {'95% CI':>18s}  {'d(B-U)':>7s}  "
        f"{'dMRR(B-C)':>10s}  {'p(Wilcox)':>10s}  {'95% CI':>18s}  {'d(B-C)':>7s}"
    )
    sep = "-" * len(hdr)

    print("\n  Statistical Significance -- Fine-Grained Alpha Sweep (0.0-0.2)")
    print(f"  {len(ALPHAS)} alpha values x {len(ENGINES)} engines x 96 answerable questions")
    print(f"  Tests: paired Wilcoxon signed-rank | bootstrap 95% CI ({N_BOOTSTRAP:,} resamples) | Cohen's d\n")
    print(sep)
    print(hdr)
    print(sep)

    all_p_bu, all_p_bc = [], []

    for alpha in ALPHAS:
        fname = alpha_to_filename(alpha)
        path = DATA_DIR / fname
        with open(path) as f:
            data = json.load(f)

        rr = {eng: extract_rr(data[eng]) for eng in ENGINES}

        # Sanity: all engines should have same number of answerable Qs
        n = len(rr["behavioral"])
        assert all(len(rr[e]) == n for e in ENGINES), f"Length mismatch at alpha={alpha}"

        mrr = {e: float(rr[e].mean()) for e in ENGINES}

        # B vs U
        diff_bu = rr["behavioral"] - rr["uniform"]
        delta_bu = float(diff_bu.mean())
        if np.all(diff_bu == 0):
            p_bu = 1.0
        else:
            _, p_bu = wilcoxon(diff_bu[diff_bu != 0])
        boot_mean_bu, lo_bu, hi_bu = bootstrap_ci(diff_bu)
        d_bu = cohens_d(diff_bu)

        # B vs C
        diff_bc = rr["behavioral"] - rr["cognitive"]
        delta_bc = float(diff_bc.mean())
        if np.all(diff_bc == 0):
            p_bc = 1.0
        else:
            _, p_bc = wilcoxon(diff_bc[diff_bc != 0])
        boot_mean_bc, lo_bc, hi_bc = bootstrap_ci(diff_bc)
        d_bc = cohens_d(diff_bc)

        all_p_bu.append(p_bu)
        all_p_bc.append(p_bc)

        ci_bu = f"[{lo_bu:+.4f}, {hi_bu:+.4f}]"
        ci_bc = f"[{lo_bc:+.4f}, {hi_bc:+.4f}]"

        print(
            f"{alpha:7.3f}  "
            f"{mrr['behavioral']:7.4f}  {mrr['uniform']:7.4f}  {mrr['cognitive']:7.4f}  "
            f"{delta_bu:+10.4f}  {p_bu:10.4f}  {ci_bu:>18s}  {d_bu:+7.3f}  "
            f"{delta_bc:+10.4f}  {p_bc:10.4f}  {ci_bc:>18s}  {d_bc:+7.3f}"
        )

    print(sep)

    # Summary
    print("\n  Summary")
    print(f"  ---------")
    sig_bu = sum(1 for p in all_p_bu if p < 0.05)
    sig_bc = sum(1 for p in all_p_bc if p < 0.05)
    print(f"  B vs U: {sig_bu}/{len(all_p_bu)} alphas significant at p < 0.05")
    print(f"  B vs C: {sig_bc}/{len(all_p_bc)} alphas significant at p < 0.05")

    min_p_bu, min_p_bc = min(all_p_bu), min(all_p_bc)
    print(f"  Smallest p-value  B vs U: {min_p_bu:.4f}   B vs C: {min_p_bc:.4f}")

    if sig_bu == 0 and sig_bc == 0:
        print("\n  => No significant differences detected across any alpha.")
        print("     All three re-ranking engines yield statistically equivalent MRR.")
    elif sig_bu > 0 or sig_bc > 0:
        print(f"\n  => Significant differences found at some alpha values.")
    print()


if __name__ == "__main__":
    main()
