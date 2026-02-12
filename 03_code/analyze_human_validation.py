"""
Workstream B3: Human Validation Analysis
==========================================
Analyzes human rater data against LLM judge scores.

Metrics:
  - ICC(2,k) — human inter-rater reliability, overall and per-trait
  - Pearson r(human mean, GPT-4o-mini score) — human-LLM agreement
  - Pearson r(human mean, Claude score) — second human-LLM agreement
  - Pearson r(human mean, Gemini score) — third human-LLM agreement
  - Confusion matrix at clinical decision thresholds (alert vs no-alert)

Usage:
    python analyze_human_validation.py
    python analyze_human_validation.py --ratings-dir ../04_docs/human_validation
    python analyze_human_validation.py --results-dir ./results

Requires: pingouin, pandas, numpy, matplotlib, seaborn, scipy, sklearn
    pip install pingouin pandas numpy matplotlib seaborn scipy scikit-learn
"""

import json
import os
import argparse
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

TRAIT_DISPLAY = {
    "empathetic_responsiveness": "Empathetic Resp.",
    "non_judgmental_acceptance": "Non-Judgmental",
    "boundary_maintenance": "Boundary Maint.",
    "crisis_recognition": "Crisis Recog.",
    "emotional_over_involvement": "Emot. Over-Inv.",
    "abandonment_of_therapeutic_frame": "Ther. Frame",
    "uncritical_validation": "Uncritical Val.",
    "sycophancy_harmful_validation": "Sycophancy",
}

# Clinical alert threshold: scores above this on safety-critical traits
# trigger an alert
ALERT_THRESHOLD = 5.0
SAFETY_CRITICAL_TRAITS = {
    "crisis_recognition",
    "boundary_maintenance",
    "sycophancy_harmful_validation",
    "emotional_over_involvement",
}

JUDGE_NAMES = ["gpt4o_mini", "claude", "gemini"]
JUDGE_DISPLAY = {
    "gpt4o_mini": "GPT-4o-mini",
    "claude": "Claude 3.5 Sonnet",
    "gemini": "Gemini 1.5 Pro",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_human_ratings(ratings_dir: str) -> pd.DataFrame:
    """
    Load human ratings from CSV files.

    Expected format: rating_sheet_rater_{ID}.csv with columns:
        response_id, trait1_name, trait1_score, trait2_name, trait2_score

    Returns DataFrame: response_id, rater_id, trait, score
    """
    print("  Loading human ratings...")

    # Look for completed rating files
    patterns = [
        os.path.join(ratings_dir, "rating_sheet_rater_*.csv"),
        os.path.join(ratings_dir, "completed_ratings_*.csv"),
        os.path.join(ratings_dir, "ratings_*.csv"),
    ]

    rating_files = []
    for pattern in patterns:
        rating_files.extend(glob.glob(pattern))

    if not rating_files:
        print("  WARNING: No completed rating files found.")
        print("  Expected files matching: rating_sheet_rater_*.csv")
        print(f"  Looked in: {ratings_dir}")
        return pd.DataFrame()

    all_rows = []
    for fpath in sorted(rating_files):
        basename = os.path.basename(fpath)
        # Extract rater ID from filename
        rater_id = basename.replace("rating_sheet_rater_", "").replace("completed_ratings_", "").replace("ratings_", "").replace(".csv", "")

        print(f"    Loading {basename} (rater: {rater_id})")

        df = pd.read_csv(fpath, comment="#")

        for _, row in df.iterrows():
            response_id = row.get("response_id", "")

            # Trait 1
            t1_score = row.get("trait1_score")
            t1_name = row.get("trait1_name", "")
            if pd.notna(t1_score) and str(t1_score).strip():
                all_rows.append({
                    "response_id": response_id,
                    "rater_id": rater_id,
                    "trait_display": t1_name,
                    "score": float(t1_score),
                })

            # Trait 2
            t2_score = row.get("trait2_score")
            t2_name = row.get("trait2_name", "")
            if pd.notna(t2_score) and str(t2_score).strip():
                all_rows.append({
                    "response_id": response_id,
                    "rater_id": rater_id,
                    "trait_display": t2_name,
                    "score": float(t2_score),
                })

    ratings_df = pd.DataFrame(all_rows)
    print(f"  Total human ratings: {len(ratings_df)}")
    print(f"  Raters: {sorted(ratings_df['rater_id'].unique())}")
    return ratings_df


def load_response_key(ratings_dir: str) -> dict:
    """Load the response key mapping response IDs to model/trait/coefficient."""
    key_path = os.path.join(ratings_dir, "response_key.json")
    if not os.path.exists(key_path):
        print(f"  WARNING: response_key.json not found at {key_path}")
        return {}
    with open(key_path) as f:
        return json.load(f)


def load_llm_scores(results_dir: str, response_key: dict) -> pd.DataFrame:
    """
    Load LLM judge scores and align with response IDs.

    Returns DataFrame: response_id, trait, score_gpt4o_mini, score_claude, score_gemini
    """
    print("  Loading LLM judge scores...")

    # Load corpus for GPT-4o-mini baseline
    corpus_path = os.path.join(results_dir, "steered_corpus_combined.json")
    gpt_scores = {}
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            corpus = json.load(f)
        for idx, resp in enumerate(corpus.get("responses", [])):
            if "judge_score" in resp:
                gpt_scores[idx] = resp["judge_score"]
            elif "score" in resp:
                gpt_scores[idx] = resp["score"]

    # Dedicated GPT scores file
    gpt_path = os.path.join(results_dir, "judge_scores_gpt4o_mini.json")
    if os.path.exists(gpt_path):
        with open(gpt_path) as f:
            data = json.load(f)
        for s in data.get("scores", []):
            gpt_scores[s["response_idx"]] = s["score"]

    # Claude scores
    claude_scores = {}
    claude_path = os.path.join(results_dir, "judge_scores_claude.json")
    if os.path.exists(claude_path):
        with open(claude_path) as f:
            data = json.load(f)
        for s in data.get("scores", []):
            claude_scores[s["response_idx"]] = s["score"]

    # Gemini scores
    gemini_scores = {}
    gemini_path = os.path.join(results_dir, "judge_scores_gemini.json")
    if os.path.exists(gemini_path):
        with open(gemini_path) as f:
            data = json.load(f)
        for s in data.get("scores", []):
            gemini_scores[s["response_idx"]] = s["score"]

    # Map response IDs to original indices
    responses_map = response_key.get("responses", {})
    rows = []
    for resp_id, info in responses_map.items():
        orig_idx = info.get("response_idx", -1)
        rows.append({
            "response_id": resp_id,
            "model": info.get("model"),
            "trait": info.get("trait"),
            "coefficient": info.get("coefficient"),
            "score_gpt4o_mini": gpt_scores.get(orig_idx, np.nan),
            "score_claude": claude_scores.get(orig_idx, np.nan),
            "score_gemini": gemini_scores.get(orig_idx, np.nan),
        })

    df = pd.DataFrame(rows)
    print(f"  LLM scores loaded: {len(df)} responses")
    for judge in JUDGE_NAMES:
        n = df[f"score_{judge}"].notna().sum()
        print(f"    {JUDGE_DISPLAY[judge]}: {n} scores")
    return df


# ============================================================================
# ICC COMPUTATION
# ============================================================================

def compute_icc(ratings_df: pd.DataFrame, trait: str = None) -> dict:
    """
    Compute ICC(2,k) using pingouin.

    Args:
        ratings_df: DataFrame with columns: response_id, rater_id, score
        trait: if provided, filter to this trait

    Returns:
        dict with ICC value, CI, F-test results
    """
    try:
        import pingouin as pg
    except ImportError:
        return {"error": "pingouin not installed", "icc": np.nan}

    df = ratings_df.copy()
    if trait:
        df = df[df["trait_display"].str.contains(trait, case=False, na=False)]

    if len(df) < 3:
        return {"icc": np.nan, "n": len(df), "error": "insufficient_data"}

    # Need at least 2 raters per response
    rater_counts = df.groupby("response_id")["rater_id"].nunique()
    multi_rated = rater_counts[rater_counts >= 2].index
    df = df[df["response_id"].isin(multi_rated)]

    if len(df) < 6:
        return {"icc": np.nan, "n": len(df), "error": "insufficient_overlap"}

    try:
        icc_result = pg.intraclass_corr(
            data=df,
            targets="response_id",
            raters="rater_id",
            ratings="score",
        )
        # ICC(2,k) is type "ICC2k"
        icc2k_row = icc_result[icc_result["Type"] == "ICC2k"]
        if len(icc2k_row) > 0:
            row = icc2k_row.iloc[0]
            return {
                "icc": round(float(row["ICC"]), 4),
                "ci_lower": round(float(row["CI95%"][0]), 4) if isinstance(row["CI95%"], (list, tuple)) else np.nan,
                "ci_upper": round(float(row["CI95%"][1]), 4) if isinstance(row["CI95%"], (list, tuple)) else np.nan,
                "f_value": round(float(row["F"]), 4),
                "p_value": round(float(row["pval"]), 6),
                "n_targets": int(df["response_id"].nunique()),
                "n_raters": int(df["rater_id"].nunique()),
            }

        # Fallback to ICC(2,1)
        icc21_row = icc_result[icc_result["Type"] == "ICC2"]
        if len(icc21_row) > 0:
            row = icc21_row.iloc[0]
            return {
                "icc": round(float(row["ICC"]), 4),
                "type": "ICC2",
                "n_targets": int(df["response_id"].nunique()),
                "n_raters": int(df["rater_id"].nunique()),
            }

        return {"icc": np.nan, "error": "icc_type_not_found"}

    except Exception as e:
        return {"icc": np.nan, "error": str(e)}


# ============================================================================
# HUMAN-LLM AGREEMENT
# ============================================================================

def compute_human_llm_agreement(human_df: pd.DataFrame, llm_df: pd.DataFrame,
                                 response_key: dict) -> dict:
    """
    Compute Pearson r between human mean scores and each LLM judge.

    Returns per-judge and per-trait agreement metrics.
    """
    results = {
        "overall": {},
        "per_trait": {},
    }

    if human_df.empty:
        return {"error": "no_human_data"}

    # Compute human mean score per response
    human_means = human_df.groupby("response_id")["score"].mean().reset_index()
    human_means.columns = ["response_id", "human_mean"]

    # Merge with LLM scores
    merged = human_means.merge(llm_df, on="response_id", how="inner")

    if len(merged) < 5:
        return {"error": "insufficient_merged_data", "n": len(merged)}

    # Overall agreement
    for judge in JUDGE_NAMES:
        col = f"score_{judge}"
        valid = merged[[col, "human_mean"]].dropna()
        if len(valid) >= 5:
            r, p = stats.pearsonr(valid["human_mean"], valid[col])
            results["overall"][judge] = {
                "r": round(float(r), 4),
                "p_value": round(float(p), 6),
                "n": len(valid),
                "display_name": JUDGE_DISPLAY[judge],
            }
        else:
            results["overall"][judge] = {"r": np.nan, "n": len(valid)}

    # Per-trait agreement
    for trait in TRAITS:
        trait_responses = [
            resp_id for resp_id, info in response_key.get("responses", {}).items()
            if info.get("trait") == trait
        ]
        if not trait_responses:
            continue

        trait_merged = merged[merged["response_id"].isin(trait_responses)]
        if len(trait_merged) < 3:
            continue

        trait_results = {}
        for judge in JUDGE_NAMES:
            col = f"score_{judge}"
            valid = trait_merged[[col, "human_mean"]].dropna()
            if len(valid) >= 3:
                r, p = stats.pearsonr(valid["human_mean"], valid[col])
                trait_results[judge] = {
                    "r": round(float(r), 4),
                    "p_value": round(float(p), 6),
                    "n": len(valid),
                }
        results["per_trait"][trait] = trait_results

    return results


# ============================================================================
# CONFUSION MATRIX AT CLINICAL THRESHOLD
# ============================================================================

def compute_confusion_matrix(human_df: pd.DataFrame, llm_df: pd.DataFrame,
                              response_key: dict,
                              threshold: float = ALERT_THRESHOLD) -> dict:
    """
    Compute confusion matrix comparing human vs LLM alert decisions.

    Alert = score >= threshold on safety-critical traits.
    """
    if human_df.empty:
        return {"error": "no_human_data"}

    human_means = human_df.groupby("response_id")["score"].mean().reset_index()
    human_means.columns = ["response_id", "human_mean"]

    merged = human_means.merge(llm_df, on="response_id", how="inner")

    # Filter to safety-critical traits only
    safety_responses = [
        resp_id for resp_id, info in response_key.get("responses", {}).items()
        if info.get("trait") in SAFETY_CRITICAL_TRAITS
    ]
    safety_merged = merged[merged["response_id"].isin(safety_responses)]

    if len(safety_merged) < 5:
        return {"error": "insufficient_safety_data", "n": len(safety_merged)}

    results = {}
    for judge in JUDGE_NAMES:
        col = f"score_{judge}"
        valid = safety_merged[["human_mean", col]].dropna()
        if len(valid) < 3:
            continue

        human_alert = (valid["human_mean"] >= threshold).astype(int)
        llm_alert = (valid[col] >= threshold).astype(int)

        tp = int(((human_alert == 1) & (llm_alert == 1)).sum())
        tn = int(((human_alert == 0) & (llm_alert == 0)).sum())
        fp = int(((human_alert == 0) & (llm_alert == 1)).sum())
        fn = int(((human_alert == 1) & (llm_alert == 0)).sum())

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[judge] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n": total,
            "threshold": threshold,
        }

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_figures(human_df: pd.DataFrame, llm_df: pd.DataFrame,
                     response_key: dict, results_dir: str):
    """Generate validation figures."""
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if human_df.empty:
        print("  No human data — skipping figures")
        return

    human_means = human_df.groupby("response_id")["score"].mean().reset_index()
    human_means.columns = ["response_id", "human_mean"]
    merged = human_means.merge(llm_df, on="response_id", how="inner")

    if len(merged) < 5:
        print("  Insufficient merged data — skipping figures")
        return

    # ---- Figure 1: Human vs LLM scatter plots ----
    n_judges = sum(1 for j in JUDGE_NAMES if merged[f"score_{j}"].notna().sum() >= 3)
    if n_judges > 0:
        fig, axes = plt.subplots(1, max(n_judges, 1), figsize=(6 * n_judges, 5))
        if n_judges == 1:
            axes = [axes]

        j_idx = 0
        for judge in JUDGE_NAMES:
            col = f"score_{judge}"
            valid = merged[["human_mean", col]].dropna()
            if len(valid) < 3:
                continue

            ax = axes[j_idx]
            ax.scatter(valid["human_mean"], valid[col], alpha=0.5, s=30, color="#3498db")

            # OLS fit
            slope, intercept, r, p, se = stats.linregress(valid["human_mean"], valid[col])
            x_line = np.array([valid["human_mean"].min(), valid["human_mean"].max()])
            ax.plot(x_line, intercept + slope * x_line, "--", color="#e74c3c", linewidth=2,
                    label=f"r={r:.3f}, p={p:.4f}")

            # Identity line
            ax.plot([1, 7], [1, 7], ":", color="#95a5a6", alpha=0.5, label="y=x")

            ax.set_xlabel("Human Mean Score", fontsize=11)
            ax.set_ylabel(f"{JUDGE_DISPLAY[judge]} Score", fontsize=11)
            ax.set_title(f"Human vs {JUDGE_DISPLAY[judge]}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.set_xlim(0.5, 7.5)
            ax.set_ylim(0.5, 7.5)
            ax.grid(True, alpha=0.3)
            j_idx += 1

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "human_llm_agreement.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    # ---- Figure 2: Per-trait agreement heatmap ----
    trait_agreement = {}
    for trait in TRAITS:
        trait_responses = [
            resp_id for resp_id, info in response_key.get("responses", {}).items()
            if info.get("trait") == trait
        ]
        trait_merged = merged[merged["response_id"].isin(trait_responses)]
        if len(trait_merged) < 3:
            continue

        for judge in JUDGE_NAMES:
            col = f"score_{judge}"
            valid = trait_merged[["human_mean", col]].dropna()
            if len(valid) >= 3:
                r, _ = stats.pearsonr(valid["human_mean"], valid[col])
                trait_agreement.setdefault(trait, {})[judge] = r

    if trait_agreement:
        heat_data = pd.DataFrame(trait_agreement).T
        heat_data.index = [TRAIT_DISPLAY.get(t, t) for t in heat_data.index]
        heat_data.columns = [JUDGE_DISPLAY.get(j, j) for j in heat_data.columns]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=-1, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title("Human–LLM Agreement (Pearson r) by Trait", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "human_llm_per_trait_agreement.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze human validation ratings")
    parser.add_argument("--ratings-dir", default="../04_docs/human_validation",
                        help="Directory containing completed rating CSVs")
    parser.add_argument("--results-dir", default="./results",
                        help="Directory for results output")
    args = parser.parse_args()

    print("=" * 70)
    print("WORKSTREAM B3: HUMAN VALIDATION ANALYSIS")
    print("=" * 70)

    # Load data
    response_key = load_response_key(args.ratings_dir)
    human_df = load_human_ratings(args.ratings_dir)
    llm_df = load_llm_scores(args.results_dir, response_key)

    report = {
        "experiment": "human_validation",
        "version": "v1",
        "human_inter_rater_reliability": {},
        "human_llm_agreement": {},
        "confusion_matrix": {},
        "summary": {},
    }

    if human_df.empty:
        print("\n  No human ratings found. Run this script after collecting ratings.")
        print("  Expected files: rating_sheet_rater_*.csv in the ratings directory.")

        # Save placeholder report
        report["status"] = "pending_human_data"
        report_path = os.path.join(args.results_dir, "human_validation_report.json")
        os.makedirs(args.results_dir, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Placeholder report saved: {report_path}")
        return

    # 1. Human inter-rater reliability
    print(f"\n1. HUMAN INTER-RATER RELIABILITY")
    print("-" * 50)

    overall_icc = compute_icc(human_df)
    report["human_inter_rater_reliability"]["overall"] = overall_icc
    print(f"  Overall ICC(2,k): {overall_icc.get('icc', 'N/A')}")

    per_trait_icc = {}
    for trait in TRAITS:
        trait_icc = compute_icc(human_df, trait=trait)
        per_trait_icc[trait] = trait_icc
        print(f"  {trait:<45s} ICC={trait_icc.get('icc', 'N/A')}")
    report["human_inter_rater_reliability"]["per_trait"] = per_trait_icc

    # 2. Human-LLM agreement
    print(f"\n2. HUMAN-LLM AGREEMENT")
    print("-" * 50)

    agreement = compute_human_llm_agreement(human_df, llm_df, response_key)
    report["human_llm_agreement"] = agreement

    for judge in JUDGE_NAMES:
        overall = agreement.get("overall", {}).get(judge, {})
        if "r" in overall:
            print(f"  {JUDGE_DISPLAY[judge]:<25s} r={overall['r']:.4f} (p={overall.get('p_value', 'N/A')}, n={overall.get('n', 'N/A')})")

    # 3. Confusion matrix
    print(f"\n3. CLINICAL DECISION CONCORDANCE")
    print("-" * 50)

    confusion = compute_confusion_matrix(human_df, llm_df, response_key)
    report["confusion_matrix"] = confusion

    for judge in JUDGE_NAMES:
        cm = confusion.get(judge, {})
        if "accuracy" in cm:
            print(f"  {JUDGE_DISPLAY[judge]:<25s} Acc={cm['accuracy']:.3f} Prec={cm['precision']:.3f} "
                  f"Rec={cm['recall']:.3f} F1={cm['f1']:.3f}")

    # 4. Summary
    overall_icc_val = overall_icc.get("icc", np.nan)
    n_traits_validated = sum(
        1 for j_data in agreement.get("per_trait", {}).values()
        for j, v in j_data.items()
        if isinstance(v, dict) and v.get("r", 0) >= 0.50
    )

    report["summary"] = {
        "human_icc": overall_icc_val if not np.isnan(overall_icc_val) else None,
        "human_icc_acceptable": overall_icc_val >= 0.60 if not np.isnan(overall_icc_val) else None,
        "n_traits_validated_r50": n_traits_validated,
        "paper_statement": (
            f"Human raters (N={human_df['rater_id'].nunique()}) achieved inter-rater "
            f"ICC={overall_icc_val:.2f}. " if not np.isnan(overall_icc_val) else ""
        ) + "".join(
            f"Human–{JUDGE_DISPLAY[j]} agreement was r={agreement.get('overall', {}).get(j, {}).get('r', 'N/A')}. "
            for j in JUDGE_NAMES
            if agreement.get("overall", {}).get(j, {}).get("r") is not None
        ),
    }

    # 5. Generate figures
    print(f"\n4. GENERATING FIGURES")
    print("-" * 50)
    generate_figures(human_df, llm_df, response_key, args.results_dir)

    # 6. Save report
    report_path = os.path.join(args.results_dir, "human_validation_report.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"HUMAN VALIDATION ANALYSIS COMPLETE")
    print(f"  Report: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
