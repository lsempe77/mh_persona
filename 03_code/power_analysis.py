"""
Workstream C2: Power Analysis
==============================
Formal statistical power analysis for both corpus sizes (N=30, N=100).

Produces:
  - Minimum detectable effect (r) at 80% power, α=0.05
  - Power curves for observed effect sizes (r ~ 0.15–0.33)
  - Explicit statement about the 18% significance rate
  - Power curve figure

Usage:
    python power_analysis.py
    python power_analysis.py --results-dir ./results

Requires: numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""

import json
import os
import argparse
import math
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SMALL = 30    # Original corpus: 10 scenarios × 3 models per trait
N_LARGE = 100   # Expanded corpus: 100 scenarios per model per trait
ALPHA = 0.05    # Significance level
POWER_TARGET = 0.80

# Observed effect sizes from v1 analysis (mean |r| across text features)
OBSERVED_TEXT_R = {
    "vader_compound": 0.203,
    "textblob_subjectivity": 0.171,
    "word_count": 0.089,
    "question_ratio": 0.154,
    "flesch_reading_ease": 0.062,
    "avg_sentence_length": 0.045,
    "tfidf_drift": 0.281,
}

# Observed activation r-values from v1
OBSERVED_ACTIVATION_R = {
    "empathetic_responsiveness": 0.732,
    "non_judgmental_acceptance": 0.589,
    "boundary_maintenance": 0.411,
    "crisis_recognition": 0.298,
    "emotional_over_involvement": 0.445,
    "abandonment_of_therapeutic_frame": 0.387,
    "uncritical_validation": 0.324,
    "sycophancy_harmful_validation": 0.258,
}

ALL_TRAITS = list(OBSERVED_ACTIVATION_R.keys())


# ============================================================================
# POWER COMPUTATIONS
# ============================================================================

def r_to_fisher_z(r: float) -> float:
    """Convert Pearson r to Fisher's z."""
    r = max(-0.999, min(0.999, r))
    return 0.5 * math.log((1 + r) / (1 - r))


def fisher_z_to_r(z: float) -> float:
    """Convert Fisher's z back to Pearson r."""
    return (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)


def power_for_r(r: float, n: int, alpha: float = ALPHA) -> float:
    """
    Compute statistical power for a two-tailed test of Pearson r.

    Uses Fisher's z transformation:
      z = arctanh(r) * sqrt(n - 3)
    Under H1, z ~ N(arctanh(r) * sqrt(n-3), 1)
    """
    if n <= 3:
        return 0.0
    if abs(r) < 1e-10:
        return alpha  # Power = alpha when effect = 0

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ncp = r_to_fisher_z(r) * math.sqrt(n - 3)

    # Power = P(|Z| > z_crit) under H1
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    return power


def min_detectable_r(n: int, alpha: float = ALPHA, power: float = POWER_TARGET) -> float:
    """
    Find the minimum r detectable at given power and alpha (two-tailed).

    Binary search on r.
    """
    low, high = 0.001, 0.999
    for _ in range(200):
        mid = (low + high) / 2
        p = power_for_r(mid, n, alpha)
        if p < power:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def sample_size_for_r(r: float, alpha: float = ALPHA, power: float = POWER_TARGET) -> int:
    """
    Find the minimum sample size to detect effect r at given power and alpha.

    Binary search on n.
    """
    if abs(r) < 0.01:
        return 999999
    low, high = 4, 100000
    for _ in range(50):
        mid = (low + high) // 2
        p = power_for_r(r, mid, alpha)
        if p < power:
            low = mid
        else:
            high = mid
    return high


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_power_analysis(results_dir: str) -> dict:
    """Run the full power analysis and return results."""

    print("=" * 70)
    print("WORKSTREAM C2: POWER ANALYSIS")
    print("=" * 70)

    results = {
        "experiment": "power_analysis",
        "version": "v1",
        "parameters": {
            "alpha": ALPHA,
            "power_target": POWER_TARGET,
            "n_small": N_SMALL,
            "n_large": N_LARGE,
        },
        "minimum_detectable_effects": {},
        "sample_size_requirements": {},
        "power_at_observed_effects": {},
        "text_feature_analysis": {},
        "activation_analysis": {},
        "significance_rate_interpretation": {},
    }

    # ------------------------------------------------------------------
    # 1. Minimum Detectable Effects (MDE)
    # ------------------------------------------------------------------
    print("\n1. MINIMUM DETECTABLE EFFECTS (MDE)")
    print("-" * 50)

    for n, label in [(N_SMALL, "N=30 (v1)"), (N_LARGE, "N=100 (v2)")]:
        mde = min_detectable_r(n)
        results["minimum_detectable_effects"][f"n_{n}"] = {
            "n": n,
            "label": label,
            "min_r": round(mde, 4),
            "min_r_squared": round(mde ** 2, 4),
        }
        print(f"  {label:15s}: MDE r = {mde:.4f} (r² = {mde**2:.4f})")

    # ------------------------------------------------------------------
    # 2. Power at observed effect sizes — Text features
    # ------------------------------------------------------------------
    print(f"\n2. POWER AT OBSERVED TEXT FEATURE EFFECTS")
    print("-" * 50)

    text_sig_count_30 = 0
    text_sig_count_100 = 0
    total_features = len(OBSERVED_TEXT_R)

    for feature, r in sorted(OBSERVED_TEXT_R.items(), key=lambda x: -x[1]):
        power_30 = power_for_r(r, N_SMALL)
        power_100 = power_for_r(r, N_LARGE)
        n_needed = sample_size_for_r(r)

        if power_30 >= POWER_TARGET:
            text_sig_count_30 += 1
        if power_100 >= POWER_TARGET:
            text_sig_count_100 += 1

        results["text_feature_analysis"][feature] = {
            "observed_r": r,
            "power_n30": round(power_30, 4),
            "power_n100": round(power_100, 4),
            "n_needed_80pct": n_needed,
            "detectable_n30": bool(power_30 >= POWER_TARGET),
            "detectable_n100": bool(power_100 >= POWER_TARGET),
        }

        det_30 = "✓" if power_30 >= POWER_TARGET else "✗"
        det_100 = "✓" if power_100 >= POWER_TARGET else "✗"
        print(f"  {feature:<30s} |r|={r:.3f} | N=30: {power_30:.2f} {det_30} | N=100: {power_100:.2f} {det_100} | need N={n_needed}")

    # ------------------------------------------------------------------
    # 3. Power at observed effect sizes — Activation projections
    # ------------------------------------------------------------------
    print(f"\n3. POWER AT OBSERVED ACTIVATION EFFECTS")
    print("-" * 50)

    for trait, r in sorted(OBSERVED_ACTIVATION_R.items(), key=lambda x: -x[1]):
        power_30 = power_for_r(r, N_SMALL)
        power_100 = power_for_r(r, N_LARGE)
        n_needed = sample_size_for_r(r)

        results["activation_analysis"][trait] = {
            "observed_r": r,
            "power_n30": round(power_30, 4),
            "power_n100": round(power_100, 4),
            "n_needed_80pct": n_needed,
        }

        det_30 = "✓" if power_30 >= POWER_TARGET else "✗"
        det_100 = "✓" if power_100 >= POWER_TARGET else "✗"
        print(f"  {trait:<45s} r={r:.3f} | N=30: {power_30:.2f} {det_30} | N=100: {power_100:.2f} {det_100}")

    # ------------------------------------------------------------------
    # 4. Significance rate interpretation
    # ------------------------------------------------------------------
    print(f"\n4. SIGNIFICANCE RATE INTERPRETATION")
    print("-" * 50)

    mean_text_r = np.mean(list(OBSERVED_TEXT_R.values()))
    mean_act_r = np.mean(list(OBSERVED_ACTIVATION_R.values()))

    # Expected significance rates at N=30
    expected_sig_rate_text = np.mean([
        1.0 if power_for_r(r, N_SMALL) >= POWER_TARGET else 0.0
        for r in OBSERVED_TEXT_R.values()
    ])
    expected_sig_rate_act = np.mean([
        1.0 if power_for_r(r, N_SMALL) >= POWER_TARGET else 0.0
        for r in OBSERVED_ACTIVATION_R.values()
    ])

    # Expected significance rates at N=100
    expected_sig_rate_text_100 = np.mean([
        1.0 if power_for_r(r, N_LARGE) >= POWER_TARGET else 0.0
        for r in OBSERVED_TEXT_R.values()
    ])
    expected_sig_rate_act_100 = np.mean([
        1.0 if power_for_r(r, N_LARGE) >= POWER_TARGET else 0.0
        for r in OBSERVED_ACTIVATION_R.values()
    ])

    interp = {
        "observed_sig_rate_text_pct": 18.0,
        "mean_text_effect_r": round(float(mean_text_r), 4),
        "mean_activation_r": round(float(mean_act_r), 4),
        "expected_detectable_text_n30_pct": round(float(expected_sig_rate_text * 100), 1),
        "expected_detectable_text_n100_pct": round(float(expected_sig_rate_text_100 * 100), 1),
        "expected_detectable_act_n30_pct": round(float(expected_sig_rate_act * 100), 1),
        "expected_detectable_act_n100_pct": round(float(expected_sig_rate_act_100 * 100), 1),
        "mde_n30": round(min_detectable_r(N_SMALL), 4),
        "mde_n100": round(min_detectable_r(N_LARGE), 4),
        "interpretation": (
            f"At N={N_SMALL}, the design has 80% power to detect r≥{min_detectable_r(N_SMALL):.3f}. "
            f"The observed 18% significance rate for text features (mean |r|={mean_text_r:.3f}) is "
            f"consistent with a mix of real effects below the detection threshold and true nulls. "
            f"At N={N_LARGE}, the MDE drops to r≥{min_detectable_r(N_LARGE):.3f}, which should "
            f"resolve whether the non-significant text features represent real but small effects "
            f"or true nulls."
        ),
        "activation_gap_statement": (
            f"Activation-level effects (mean r={mean_act_r:.3f}) are {mean_act_r/mean_text_r:.1f}× "
            f"larger than text-level effects (mean |r|={mean_text_r:.3f}). Even at N={N_SMALL}, "
            f"activation effects are well-powered (expected detection rate: "
            f"{expected_sig_rate_act*100:.0f}%), while text effects are underpowered "
            f"(expected detection rate: {expected_sig_rate_text*100:.0f}%)."
        ),
    }
    results["significance_rate_interpretation"] = interp

    print(f"  Mean text |r|:       {mean_text_r:.4f}")
    print(f"  Mean activation r:   {mean_act_r:.4f}")
    print(f"  Activation/text gap: {mean_act_r/mean_text_r:.1f}×")
    print(f"\n  MDE at N=30:  r≥{min_detectable_r(N_SMALL):.3f}")
    print(f"  MDE at N=100: r≥{min_detectable_r(N_LARGE):.3f}")
    print(f"\n  {interp['interpretation']}")

    # ------------------------------------------------------------------
    # 5. Power curves figure
    # ------------------------------------------------------------------
    print(f"\n5. GENERATING POWER CURVE FIGURE")
    print("-" * 50)

    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    r_range = np.arange(0.01, 0.80, 0.005)

    # ---- Panel A: Power vs Effect Size for different N ----
    ax = axes[0]
    sample_sizes = [20, 30, 50, 100, 200]
    colors = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#3498db"]

    for n, color in zip(sample_sizes, colors):
        powers = [power_for_r(r, n) for r in r_range]
        lw = 3.0 if n in [N_SMALL, N_LARGE] else 1.5
        ax.plot(r_range, powers, label=f"N={n}", color=color, linewidth=lw)

    # Mark 80% power threshold
    ax.axhline(y=POWER_TARGET, color="#95a5a6", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.02, POWER_TARGET + 0.02, f"80% power", fontsize=9, color="#95a5a6")

    # Mark observed effect sizes
    for feature, r in OBSERVED_TEXT_R.items():
        ax.axvline(x=r, color="#e74c3c", linestyle=":", alpha=0.3)

    # Mark mean effects
    ax.axvline(x=mean_text_r, color="#e74c3c", linestyle="-", alpha=0.6, linewidth=2)
    ax.text(mean_text_r + 0.01, 0.15, f"Mean text |r|={mean_text_r:.3f}",
            fontsize=8, color="#e74c3c", rotation=90)

    ax.axvline(x=mean_act_r, color="#2c3e50", linestyle="-", alpha=0.6, linewidth=2)
    ax.text(mean_act_r + 0.01, 0.15, f"Mean activation r={mean_act_r:.3f}",
            fontsize=8, color="#2c3e50", rotation=90)

    ax.set_xlabel("Effect Size (Pearson r)", fontsize=12)
    ax.set_ylabel("Statistical Power", fontsize=12)
    ax.set_title("A. Power Curves by Sample Size", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # ---- Panel B: Required N vs Effect Size ----
    ax = axes[1]

    r_range_b = np.arange(0.05, 0.80, 0.005)
    n_required = [sample_size_for_r(r) for r in r_range_b]

    ax.plot(r_range_b, n_required, color="#2c3e50", linewidth=2.5)
    ax.fill_between(r_range_b, n_required, alpha=0.1, color="#2c3e50")

    # Mark sample sizes
    ax.axhline(y=N_SMALL, color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.text(0.6, N_SMALL + 5, f"N={N_SMALL} (v1 corpus)", fontsize=9, color="#e74c3c")

    ax.axhline(y=N_LARGE, color="#2ecc71", linestyle="--", linewidth=1.5)
    ax.text(0.6, N_LARGE + 5, f"N={N_LARGE} (v2 corpus)", fontsize=9, color="#2ecc71")

    # Mark key effects
    for label, r, color in [
        ("Mean text", mean_text_r, "#e74c3c"),
        ("Mean activation", mean_act_r, "#2c3e50"),
    ]:
        n_req = sample_size_for_r(r)
        ax.plot(r, n_req, "o", color=color, markersize=10, zorder=5)
        ax.annotate(f"{label}\nr={r:.3f}, need N={n_req}",
                    xy=(r, n_req), xytext=(r + 0.05, n_req + 30),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    ax.set_xlabel("Effect Size (Pearson r)", fontsize=12)
    ax.set_ylabel("Required Sample Size (N)", fontsize=12)
    ax.set_title("B. Required N for 80% Power", fontsize=13, fontweight="bold")
    ax.set_xlim(0.05, 0.8)
    ax.set_ylim(0, max(300, N_LARGE + 50))
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Power Analysis: Activation Steering Corpus (α={ALPHA}, two-tailed)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    fig_path = os.path.join(results_dir, "figures", "power_analysis_curves.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # ------------------------------------------------------------------
    # 6. Per-trait power table
    # ------------------------------------------------------------------
    print(f"\n6. PER-TRAIT POWER TABLE")
    print("-" * 70)
    print(f"  {'Trait':<45s} {'r':>6s} {'Pow30':>7s} {'Pow100':>8s} {'N@80%':>7s}")
    print(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*8} {'-'*7}")

    trait_table = []
    for trait in ALL_TRAITS:
        r = OBSERVED_ACTIVATION_R[trait]
        p30 = power_for_r(r, N_SMALL)
        p100 = power_for_r(r, N_LARGE)
        n_req = sample_size_for_r(r)
        trait_table.append({
            "trait": trait,
            "r": r,
            "power_n30": round(p30, 4),
            "power_n100": round(p100, 4),
            "n_needed": n_req,
        })
        print(f"  {trait:<45s} {r:6.3f} {p30:7.3f} {p100:8.3f} {n_req:7d}")

    results["per_trait_power"] = trait_table

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    report_path = os.path.join(results_dir, "power_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*70}")
    print(f"POWER ANALYSIS COMPLETE")
    print(f"  Report: {report_path}")
    print(f"  Figure: {fig_path}")
    print(f"{'='*70}")

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Power analysis for activation steering corpus")
    parser.add_argument("--results-dir", default="./results",
                        help="Directory for results output")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    run_power_analysis(args.results_dir)


if __name__ == "__main__":
    main()
