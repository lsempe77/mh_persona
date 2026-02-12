"""
Workstream A3: Inter-Judge Agreement Analysis
==============================================
Computes inter-judge agreement (ICC) across three LLM judges:
  1. GPT-4o-mini (original scores from steered_corpus_combined.json)
  2. Claude 3.5 Sonnet (from judge_scores_claude.json)
  3. Gemini 1.5 Pro (from judge_scores_gemini.json)

Produces:
  - ICC(2,1) for each pair and three-way, per-trait and overall
  - Bland-Altman plots (systematic bias detection)
  - Per-trait agreement heatmap
  - Summary report JSON

Usage:
    python analyze_judge_agreement.py
    python analyze_judge_agreement.py --results-dir ./results

Requires: pingouin, pandas, numpy, matplotlib, seaborn
    pip install pingouin pandas numpy matplotlib seaborn
"""

import json
import os
import argparse
import sys
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAITS = [
    "empathetic_responsiveness",
    "non_judgmental_acceptance",
    "boundary_maintenance",
    "crisis_recognition",
    "emotional_over_involvement",
    "abandonment_of_therapeutic_frame",
    "uncritical_validation",
    "sycophancy_harmful_validation",
]

JUDGE_NAMES = ["gpt4o_mini", "claude", "gemini"]
JUDGE_PAIRS = [
    ("gpt4o_mini", "claude"),
    ("gpt4o_mini", "gemini"),
    ("claude", "gemini"),
]

JUDGE_DISPLAY = {
    "gpt4o_mini": "GPT-4o-mini",
    "claude": "Claude 3.5 Sonnet",
    "gemini": "Gemini 1.5 Pro",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_scores(results_dir: str) -> pd.DataFrame:
    """
    Load and align scores from all three judges into a single DataFrame.

    Returns a DataFrame with columns:
        response_idx, model, trait, coefficient, scenario_id,
        score_gpt4o_mini, score_claude, score_gemini
    """
    # -- 1. Original GPT-4o-mini scores (embedded in the corpus) --
    corpus_path = os.path.join(results_dir, "steered_corpus_combined.json")
    print(f"  Loading corpus: {corpus_path}")
    with open(corpus_path) as f:
        corpus = json.load(f)

    # Build baseline DataFrame from responses
    # The original corpus does NOT contain judge scores inline — the original
    # GPT-4o-mini scores are in a separate scoring pass. If there is no
    # dedicated GPT-4o-mini scores file, we need to check for one.
    gpt_scores_path = os.path.join(results_dir, "judge_scores_gpt4o_mini.json")
    if os.path.exists(gpt_scores_path):
        print(f"  Loading GPT-4o-mini scores: {gpt_scores_path}")
        with open(gpt_scores_path) as f:
            gpt_data = json.load(f)
        gpt_scores = {s["response_idx"]: s["score"] for s in gpt_data["scores"]}
    else:
        # Fall back: try to extract scores from corpus responses if present
        # Some corpus formats embed a "judge_score" field
        gpt_scores = {}
        for idx, resp in enumerate(corpus["responses"]):
            if "judge_score" in resp:
                gpt_scores[idx] = resp["judge_score"]
            elif "score" in resp:
                gpt_scores[idx] = resp["score"]

    if not gpt_scores:
        print("  WARNING: No GPT-4o-mini scores found. Will compute pairwise")
        print("           agreement between Claude and Gemini only.")

    # -- 2. Claude scores --
    claude_path = os.path.join(results_dir, "judge_scores_claude.json")
    print(f"  Loading Claude scores: {claude_path}")
    with open(claude_path) as f:
        claude_data = json.load(f)
    claude_scores = {s["response_idx"]: s["score"] for s in claude_data["scores"]}

    # -- 3. Gemini scores --
    gemini_path = os.path.join(results_dir, "judge_scores_gemini.json")
    print(f"  Loading Gemini scores: {gemini_path}")
    with open(gemini_path) as f:
        gemini_data = json.load(f)
    gemini_scores = {s["response_idx"]: s["score"] for s in gemini_data["scores"]}

    # -- 4. Assemble DataFrame --
    rows = []
    for idx, resp in enumerate(corpus["responses"]):
        row = {
            "response_idx": idx,
            "model": resp["model"],
            "trait": resp["trait"],
            "coefficient": resp["coefficient"],
            "scenario_id": resp["scenario_id"],
            "score_gpt4o_mini": gpt_scores.get(idx, np.nan),
            "score_claude": claude_scores.get(idx, np.nan),
            "score_gemini": gemini_scores.get(idx, np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Assembled {len(df)} rows")
    print(f"  GPT-4o-mini scores: {df['score_gpt4o_mini'].notna().sum()}")
    print(f"  Claude scores:      {df['score_claude'].notna().sum()}")
    print(f"  Gemini scores:      {df['score_gemini'].notna().sum()}")

    return df


# ============================================================================
# ICC COMPUTATION
# ============================================================================

def compute_icc(df: pd.DataFrame, score_cols: list, trait: str = None) -> dict:
    """
    Compute ICC(2,1) — two-way random, single measures — using pingouin.

    Args:
        df: DataFrame with score columns
        score_cols: list of column names (e.g., ["score_claude", "score_gemini"])
        trait: if provided, filter to this trait first

    Returns:
        dict with ICC value, confidence interval, F-stat, p-value
    """
    import pingouin as pg

    subset = df.copy()
    if trait:
        subset = subset[subset["trait"] == trait]

    # Drop rows with any NaN in the score columns
    subset = subset.dropna(subset=score_cols)
    if len(subset) < 10:
        return {"icc": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan,
                "f_stat": np.nan, "p_value": np.nan, "n": len(subset)}

    # Reshape to long format for pingouin
    long_rows = []
    for _, row in subset.iterrows():
        for col in score_cols:
            judge_name = col.replace("score_", "")
            long_rows.append({
                "response_idx": row["response_idx"],
                "judge": judge_name,
                "score": row[col],
            })
    long_df = pd.DataFrame(long_rows)

    # Compute ICC
    icc_result = pg.intraclass_corr(
        data=long_df,
        targets="response_idx",
        raters="judge",
        ratings="score",
    )

    # ICC(2,1) is the "ICC2" row (type="ICC2")
    icc2_row = icc_result[icc_result["Type"] == "ICC2"]
    if icc2_row.empty:
        # Fallback to ICC2k
        icc2_row = icc_result[icc_result["Type"] == "ICC2k"]

    if icc2_row.empty:
        return {"icc": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan,
                "f_stat": np.nan, "p_value": np.nan, "n": len(subset)}

    row = icc2_row.iloc[0]
    return {
        "icc": float(row["ICC"]),
        "ci95_lower": float(row["CI95%"][0]) if isinstance(row["CI95%"], (list, np.ndarray)) else np.nan,
        "ci95_upper": float(row["CI95%"][1]) if isinstance(row["CI95%"], (list, np.ndarray)) else np.nan,
        "f_stat": float(row["F"]),
        "p_value": float(row["pval"]),
        "n": len(subset),
    }


def compute_all_iccs(df: pd.DataFrame) -> dict:
    """
    Compute ICC for all pairs, three-way, per-trait and overall.

    Returns a nested dict: results[pair_label][trait_or_overall] = icc_dict
    """
    results = {}

    # Determine which judges have data
    available_judges = []
    for judge in JUDGE_NAMES:
        col = f"score_{judge}"
        if col in df.columns and df[col].notna().sum() > 0:
            available_judges.append(judge)

    print(f"\n  Available judges: {[JUDGE_DISPLAY.get(j, j) for j in available_judges]}")

    # --- Pairwise ICC ---
    for j1, j2 in JUDGE_PAIRS:
        if j1 not in available_judges or j2 not in available_judges:
            continue

        pair_label = f"{j1}_vs_{j2}"
        cols = [f"score_{j1}", f"score_{j2}"]
        results[pair_label] = {}

        # Overall
        results[pair_label]["overall"] = compute_icc(df, cols)

        # Per trait
        for trait in TRAITS:
            results[pair_label][trait] = compute_icc(df, cols, trait=trait)

    # --- Three-way ICC ---
    if len(available_judges) >= 3:
        three_cols = [f"score_{j}" for j in available_judges[:3]]
        three_label = "three_way"
        results[three_label] = {}
        results[three_label]["overall"] = compute_icc(df, three_cols)
        for trait in TRAITS:
            results[three_label][trait] = compute_icc(df, three_cols, trait=trait)
    elif len(available_judges) == 2:
        # Only two judges available — three-way is same as the one pair
        print("  NOTE: Only 2 judges available, three-way ICC equals pairwise.")

    return results


# ============================================================================
# BLAND-ALTMAN PLOTS
# ============================================================================

def bland_altman_plots(df: pd.DataFrame, figures_dir: str):
    """
    Generate Bland-Altman plots for each judge pair.

    A Bland-Altman plot shows:
      x-axis: mean of two measurements
      y-axis: difference between measurements
      Horizontal lines at mean difference (bias) and +/- 1.96 SD
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available_judges = []
    for judge in JUDGE_NAMES:
        col = f"score_{judge}"
        if col in df.columns and df[col].notna().sum() > 0:
            available_judges.append(judge)

    pairs_to_plot = [(j1, j2) for j1, j2 in JUDGE_PAIRS
                     if j1 in available_judges and j2 in available_judges]

    if not pairs_to_plot:
        print("  WARNING: Not enough judges for Bland-Altman plots")
        return

    n_pairs = len(pairs_to_plot)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]

    for ax, (j1, j2) in zip(axes, pairs_to_plot):
        col1 = f"score_{j1}"
        col2 = f"score_{j2}"
        subset = df.dropna(subset=[col1, col2])

        mean_scores = (subset[col1] + subset[col2]) / 2.0
        diff_scores = subset[col1] - subset[col2]

        mean_diff = diff_scores.mean()
        std_diff = diff_scores.std()
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        # Color by trait
        trait_colors = {
            "empathetic_responsiveness": "#e74c3c",
            "non_judgmental_acceptance": "#3498db",
            "boundary_maintenance": "#2ecc71",
            "crisis_recognition": "#f39c12",
            "emotional_over_involvement": "#9b59b6",
            "abandonment_of_therapeutic_frame": "#1abc9c",
            "uncritical_validation": "#e67e22",
            "sycophancy_harmful_validation": "#34495e",
        }
        colors = [trait_colors.get(t, "#95a5a6") for t in subset["trait"]]

        ax.scatter(mean_scores, diff_scores, alpha=0.3, s=15, c=colors, edgecolors="none")
        ax.axhline(y=mean_diff, color="red", linestyle="-", linewidth=1.5,
                    label=f"Bias: {mean_diff:.2f}")
        ax.axhline(y=upper_loa, color="gray", linestyle="--", linewidth=1,
                    label=f"+1.96 SD: {upper_loa:.2f}")
        ax.axhline(y=lower_loa, color="gray", linestyle="--", linewidth=1,
                    label=f"-1.96 SD: {lower_loa:.2f}")
        ax.axhline(y=0, color="black", linestyle=":", alpha=0.3)

        ax.set_xlabel("Mean of Two Judges", fontsize=11)
        ax.set_ylabel(f"{JUDGE_DISPLAY.get(j1, j1)} - {JUDGE_DISPLAY.get(j2, j2)}", fontsize=11)
        ax.set_title(f"Bland-Altman: {JUDGE_DISPLAY.get(j1, j1)} vs {JUDGE_DISPLAY.get(j2, j2)}",
                      fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(-6.5, 6.5)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "bland_altman_judge_pairs.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Per-trait Bland-Altman (one figure per pair, 8 panels) ---
    for j1, j2 in pairs_to_plot:
        col1 = f"score_{j1}"
        col2 = f"score_{j2}"

        fig, axes_grid = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(
            f"Bland-Altman by Trait: {JUDGE_DISPLAY.get(j1, j1)} vs {JUDGE_DISPLAY.get(j2, j2)}",
            fontsize=14, fontweight="bold", y=0.98,
        )

        for t_idx, trait in enumerate(TRAITS):
            ax = axes_grid[t_idx // 4, t_idx % 4]
            subset = df[df["trait"] == trait].dropna(subset=[col1, col2])

            if len(subset) < 5:
                ax.set_title(trait.replace("_", " ").title() + " (N/A)", fontsize=9)
                continue

            mean_s = (subset[col1] + subset[col2]) / 2.0
            diff_s = subset[col1] - subset[col2]
            mean_d = diff_s.mean()
            std_d = diff_s.std()

            ax.scatter(mean_s, diff_s, alpha=0.4, s=12,
                       c=trait_colors.get(trait, "#95a5a6"), edgecolors="none")
            ax.axhline(y=mean_d, color="red", linestyle="-", linewidth=1)
            ax.axhline(y=mean_d + 1.96 * std_d, color="gray", linestyle="--", linewidth=0.8)
            ax.axhline(y=mean_d - 1.96 * std_d, color="gray", linestyle="--", linewidth=0.8)
            ax.axhline(y=0, color="black", linestyle=":", alpha=0.3)

            ax.set_title(f"{trait.replace('_', ' ').title()}\nbias={mean_d:.2f}, n={len(subset)}",
                          fontsize=9)
            ax.set_ylim(-6.5, 6.5)
            ax.grid(True, alpha=0.2)

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig_path = os.path.join(figures_dir, f"bland_altman_per_trait_{j1}_vs_{j2}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")


# ============================================================================
# AGREEMENT HEATMAP
# ============================================================================

def agreement_heatmap(icc_results: dict, figures_dir: str):
    """
    Generate a per-trait ICC heatmap showing agreement across all judge pairs.

    Rows = traits, Columns = judge pairs, Cell = ICC(2,1) value.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Build matrix
    pair_labels = [k for k in icc_results if k != "three_way"]
    display_labels = []
    for pl in pair_labels:
        parts = pl.split("_vs_")
        display_labels.append(
            f"{JUDGE_DISPLAY.get(parts[0], parts[0])}\nvs\n{JUDGE_DISPLAY.get(parts[1], parts[1])}"
        )

    # Add three-way if available
    if "three_way" in icc_results:
        pair_labels.append("three_way")
        display_labels.append("Three-Way")

    trait_labels = [t.replace("_", " ").title() for t in TRAITS]

    matrix = np.full((len(TRAITS), len(pair_labels)), np.nan)
    for col_idx, pl in enumerate(pair_labels):
        for row_idx, trait in enumerate(TRAITS):
            icc_val = icc_results.get(pl, {}).get(trait, {}).get("icc", np.nan)
            matrix[row_idx, col_idx] = icc_val

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(pair_labels)), 8))

    # Custom colormap: red (poor) -> yellow (moderate) -> green (excellent)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(display_labels, fontsize=9, ha="center")
    ax.set_yticks(range(len(TRAITS)))
    ax.set_yticklabels(trait_labels, fontsize=10)

    # Annotate cells
    for i in range(len(TRAITS)):
        for j in range(len(pair_labels)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "gray"
            else:
                text = f"{val:.2f}"
                color = "white" if val < 0.3 or val > 0.85 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    # Add overall row
    overall_vals = []
    for pl in pair_labels:
        ov = icc_results.get(pl, {}).get("overall", {}).get("icc", np.nan)
        overall_vals.append(ov)

    ax.set_title("Inter-Judge Agreement: ICC(2,1) by Trait and Judge Pair",
                  fontsize=13, fontweight="bold", pad=15)

    plt.colorbar(im, ax=ax, label="ICC(2,1)", shrink=0.8)
    plt.tight_layout()

    fig_path = os.path.join(figures_dir, "judge_agreement_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Also make an overall summary bar chart ---
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(pair_labels)), 5))

    x = range(len(pair_labels))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][:len(pair_labels)]
    bars = ax.bar(x, overall_vals, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, overall_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel("ICC(2,1)", fontsize=12)
    ax.set_title("Overall Inter-Judge Agreement", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)

    # Reference lines
    ax.axhline(y=0.75, color="green", linestyle="--", alpha=0.5, label="Excellent (>0.75)")
    ax.axhline(y=0.50, color="orange", linestyle="--", alpha=0.5, label="Moderate (>0.50)")
    ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.5, label="Poor (<0.25)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "judge_agreement_overall.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# ============================================================================
# SCORE DISTRIBUTION COMPARISON
# ============================================================================

def score_distribution_plot(df: pd.DataFrame, figures_dir: str):
    """Plot score distributions per judge for visual comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available_judges = []
    for judge in JUDGE_NAMES:
        col = f"score_{judge}"
        if col in df.columns and df[col].notna().sum() > 0:
            available_judges.append(judge)

    fig, axes = plt.subplots(1, len(available_judges), figsize=(6 * len(available_judges), 5))
    if len(available_judges) == 1:
        axes = [axes]

    judge_colors = {"gpt4o_mini": "#3498db", "claude": "#e74c3c", "gemini": "#2ecc71"}

    for ax, judge in zip(axes, available_judges):
        col = f"score_{judge}"
        scores = df[col].dropna()
        bins = np.arange(0.5, 8.5, 1.0)
        ax.hist(scores, bins=bins, color=judge_colors.get(judge, "#95a5a6"),
                edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Score (1-7)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{JUDGE_DISPLAY.get(judge, judge)}\nmean={scores.mean():.2f}, std={scores.std():.2f}",
                      fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, 8))
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Score Distributions by Judge", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "judge_score_distributions.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(df: pd.DataFrame, icc_results: dict, results_dir: str):
    """Generate the summary JSON report."""
    # Descriptive stats per judge
    judge_stats = {}
    for judge in JUDGE_NAMES:
        col = f"score_{judge}"
        if col in df.columns and df[col].notna().sum() > 0:
            scores = df[col].dropna()
            judge_stats[judge] = {
                "display_name": JUDGE_DISPLAY.get(judge, judge),
                "n_scores": int(scores.count()),
                "mean": round(float(scores.mean()), 3),
                "std": round(float(scores.std()), 3),
                "median": round(float(scores.median()), 1),
                "min": float(scores.min()),
                "max": float(scores.max()),
            }

    # Per-trait bias (mean difference between judge pairs)
    bias_analysis = {}
    for j1, j2 in JUDGE_PAIRS:
        col1, col2 = f"score_{j1}", f"score_{j2}"
        if col1 not in df.columns or col2 not in df.columns:
            continue
        pair_key = f"{j1}_vs_{j2}"
        bias_analysis[pair_key] = {}
        for trait in TRAITS:
            subset = df[df["trait"] == trait].dropna(subset=[col1, col2])
            if len(subset) < 5:
                continue
            diff = subset[col1] - subset[col2]
            bias_analysis[pair_key][trait] = {
                "mean_diff": round(float(diff.mean()), 3),
                "std_diff": round(float(diff.std()), 3),
                "n": len(subset),
            }

    # ICC interpretation
    def interpret_icc(val):
        if np.isnan(val):
            return "N/A"
        if val >= 0.75:
            return "excellent"
        if val >= 0.60:
            return "good"
        if val >= 0.40:
            return "fair"
        return "poor"

    icc_summary = {}
    for pair_label, pair_data in icc_results.items():
        icc_summary[pair_label] = {}
        for key, icc_dict in pair_data.items():
            icc_val = icc_dict.get("icc", np.nan)
            icc_summary[pair_label][key] = {
                "icc": round(icc_val, 4) if not np.isnan(icc_val) else None,
                "ci95_lower": round(icc_dict.get("ci95_lower", np.nan), 4)
                    if not np.isnan(icc_dict.get("ci95_lower", np.nan)) else None,
                "ci95_upper": round(icc_dict.get("ci95_upper", np.nan), 4)
                    if not np.isnan(icc_dict.get("ci95_upper", np.nan)) else None,
                "f_stat": round(icc_dict.get("f_stat", np.nan), 4)
                    if not np.isnan(icc_dict.get("f_stat", np.nan)) else None,
                "p_value": round(icc_dict.get("p_value", np.nan), 6)
                    if not np.isnan(icc_dict.get("p_value", np.nan)) else None,
                "n": icc_dict.get("n", 0),
                "interpretation": interpret_icc(icc_val),
            }

    report = {
        "metadata": {
            "analysis": "Inter-Judge Agreement (Workstream A3)",
            "judges": list(judge_stats.keys()),
            "n_traits": len(TRAITS),
            "traits": TRAITS,
            "n_responses": len(df),
            "icc_type": "ICC(2,1) — two-way random, single measures",
        },
        "judge_descriptives": judge_stats,
        "icc_results": icc_summary,
        "bias_analysis": bias_analysis,
    }

    report_path = os.path.join(results_dir, "judge_agreement_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report: {report_path}")

    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze inter-judge agreement")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Path to results directory (default: ./results)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"WORKSTREAM A3: INTER-JUDGE AGREEMENT ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Results dir: {results_dir}")
    print(f"  Figures dir: {figures_dir}")
    print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Step 1: Loading scores...")
    try:
        df = load_scores(results_dir)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Ensure all judge scoring scripts have been run first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Compute ICCs
    # ------------------------------------------------------------------
    print("\nStep 2: Computing ICC(2,1)...")
    icc_results = compute_all_iccs(df)

    # Print summary table
    print(f"\n  {'=' * 70}")
    print(f"  ICC(2,1) RESULTS")
    print(f"  {'=' * 70}")

    for pair_label, pair_data in icc_results.items():
        print(f"\n  --- {pair_label} ---")
        overall = pair_data.get("overall", {})
        icc_val = overall.get("icc", float("nan"))
        ci_lo = overall.get("ci95_lower", float("nan"))
        ci_hi = overall.get("ci95_upper", float("nan"))
        n = overall.get("n", 0)
        print(f"  Overall: ICC = {icc_val:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], n = {n}")

        print(f"  {'Trait':<45} {'ICC':>6} {'CI95':>16} {'N':>6}")
        print(f"  {'-' * 75}")
        for trait in TRAITS:
            t_data = pair_data.get(trait, {})
            t_icc = t_data.get("icc", float("nan"))
            t_lo = t_data.get("ci95_lower", float("nan"))
            t_hi = t_data.get("ci95_upper", float("nan"))
            t_n = t_data.get("n", 0)
            ci_str = f"[{t_lo:.3f}, {t_hi:.3f}]"
            print(f"  {trait:<45} {t_icc:>6.3f} {ci_str:>16} {t_n:>6}")

    # ------------------------------------------------------------------
    # 3. Generate plots
    # ------------------------------------------------------------------
    print("\nStep 3: Generating visualizations...")

    print("\n  Bland-Altman plots...")
    bland_altman_plots(df, figures_dir)

    print("\n  Agreement heatmap...")
    agreement_heatmap(icc_results, figures_dir)

    print("\n  Score distributions...")
    score_distribution_plot(df, figures_dir)

    # ------------------------------------------------------------------
    # 4. Generate report
    # ------------------------------------------------------------------
    print("\nStep 4: Generating report...")
    report = generate_report(df, icc_results, results_dir)

    # ------------------------------------------------------------------
    # 5. Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Judges analyzed: {len(report['judge_descriptives'])}")
    print(f"  ICC pairs computed: {len(report['icc_results'])}")
    print(f"  Report: {os.path.join(results_dir, 'judge_agreement_report.json')}")
    print(f"  Figures: {figures_dir}/")
    print(f"    - bland_altman_judge_pairs.png")
    print(f"    - bland_altman_per_trait_*.png")
    print(f"    - judge_agreement_heatmap.png")
    print(f"    - judge_agreement_overall.png")
    print(f"    - judge_score_distributions.png")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
