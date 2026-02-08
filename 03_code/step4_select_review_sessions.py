"""
Step 4: Select Sessions for Expert Clinical Review
====================================================

Phase 4.5 — Stratified sampling of monitoring sessions for clinician rating.

Selects 30 sessions per model from Phase 3 monitoring output:
  - 10 True Alerts (Warning or Critical)
  - 5 Watch-Only (Watch but no Warning/Critical)
  - 10 Clean Sessions (no alerts)
  - 5 High-Drift Clean (no alerts but high activation drift)

Exports de-identified session packages for blinded clinician review.

Usage:
    modal run step4_select_review_sessions.py
    modal run step4_select_review_sessions.py --model llama3
"""

import modal
import json
import os
import random
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

N_PER_STRATUM = {
    "true_alert": 10,      # Warning or Critical
    "watch_only": 5,       # Watch but no Warning/Critical
    "clean": 10,           # No alerts at all
    "high_drift_clean": 5, # No alerts but top quartile drift
}

MODELS = ["llama3", "qwen2", "mistral"]
N_PARALLEL = 10  # Batches per model from monitoring run
SEED = 42

# ============================================================================
# MODAL SETUP
# ============================================================================

app = modal.App("expert-review-selection")

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

vol = modal.Volume.from_name("ai-persona-results", create_if_missing=True)


@app.function(
    image=image,
    timeout=600,
    volumes={"/results": vol},
)
def select_sessions(model_key: str):
    """Select and export sessions for expert review."""
    import numpy as np

    random.seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'='*60}")
    print(f"SESSION SELECTION — {model_key.upper()}")
    print(f"{'='*60}")

    # Load all monitoring batches
    vol.reload()
    all_sessions = []
    for i in range(N_PARALLEL):
        batch_path = f"/results/monitor_batch_{model_key}_{i}.json"
        if os.path.exists(batch_path):
            with open(batch_path) as f:
                batch = json.load(f)
            all_sessions.extend(batch)

    if not all_sessions:
        print(f"ERROR: No monitoring data found for {model_key}")
        return {"error": "no data"}

    print(f"✓ Loaded {len(all_sessions)} sessions")

    # Classify sessions into strata
    true_alerts = []    # Warning or Critical
    watch_only = []     # Watch but no Warning/Critical
    clean = []          # No alerts
    severity_order = {"none": 0, "watch": 1, "warning": 2, "critical": 3}

    for session in all_sessions:
        summary = session.get("session_summary", {})
        max_sev = summary.get("max_severity", "none")
        n_turns = session.get("n_turns", 0)

        if n_turns < 3:
            continue  # Skip short sessions

        if max_sev in ("warning", "critical"):
            true_alerts.append(session)
        elif max_sev == "watch":
            watch_only.append(session)
        else:
            clean.append(session)

    print(f"  True Alerts (Warning+): {len(true_alerts)}")
    print(f"  Watch-Only:             {len(watch_only)}")
    print(f"  Clean:                  {len(clean)}")

    # Compute activation drift for clean sessions to identify high-drift subset
    # Drift = max absolute z-score across all traits and turns
    import numpy as np

    for session in clean:
        max_z = 0.0
        for turn in session.get("turns", []):
            for trait, data in turn.get("traits", {}).items():
                z = abs(data.get("z_score", 0.0))
                if z > max_z:
                    max_z = z
        session["_max_z"] = max_z

    # Sort clean by drift (descending) and split
    clean_sorted = sorted(clean, key=lambda s: s.get("_max_z", 0), reverse=True)
    q75_idx = len(clean_sorted) // 4
    high_drift_clean = clean_sorted[:q75_idx]
    normal_clean = clean_sorted[q75_idx:]

    print(f"  High-Drift Clean (top 25%): {len(high_drift_clean)}")
    print(f"  Normal Clean:               {len(normal_clean)}")

    # Sample from each stratum
    selected = {}

    def sample_stratum(pool, n, name):
        if len(pool) < n:
            print(f"  ⚠ {name}: only {len(pool)} available (need {n}), taking all")
            return pool
        return random.sample(pool, n)

    selected["true_alert"] = sample_stratum(true_alerts, N_PER_STRATUM["true_alert"], "True Alert")
    selected["watch_only"] = sample_stratum(watch_only, N_PER_STRATUM["watch_only"], "Watch-Only")
    selected["clean"] = sample_stratum(normal_clean, N_PER_STRATUM["clean"], "Clean")
    selected["high_drift_clean"] = sample_stratum(high_drift_clean, N_PER_STRATUM["high_drift_clean"], "High-Drift Clean")

    total_selected = sum(len(v) for v in selected.values())
    print(f"\n✓ Selected {total_selected} sessions total")

    # Create de-identified packages (blinded)
    review_packages = []
    package_idx = 0

    for stratum, sessions in selected.items():
        for session in sessions:
            package_idx += 1
            package = {
                "package_id": f"REV-{model_key.upper()}-{package_idx:03d}",
                "stratum": stratum,  # For analysis, not shown to reviewers
                "n_turns": session.get("n_turns", 0),
                "scenario_category": session.get("scenario_category", "unknown"),
                "turns": [],
            }

            for turn in session.get("turns", []):
                turn_data = {
                    "turn": turn.get("turn", 0),
                    "user_message": turn.get("user_message_preview", ""),
                    "model_response": turn.get("response_preview", ""),
                }
                package["turns"].append(turn_data)

            # Hidden fields (for analysis, not shown to reviewers)
            package["_hidden"] = {
                "model": model_key,
                "session_id": session.get("session_id", ""),
                "max_severity": session.get("session_summary", {}).get("max_severity", "none"),
                "alert_traits": session.get("session_summary", {}).get("alert_traits", []),
                "max_z_score": session.get("_max_z", 0.0),
                "trait_summaries": session.get("trait_summaries", {}),
            }

            review_packages.append(package)

    # Save packages
    output_path = f"/results/expert_review_sessions_{model_key}.json"
    with open(output_path, "w") as f:
        json.dump(review_packages, f, indent=2)
    vol.commit()

    # Also save blinded version (for distribution to reviewers)
    blinded_packages = []
    for pkg in review_packages:
        blinded = {k: v for k, v in pkg.items() if k != "_hidden"}
        blinded_packages.append(blinded)

    blinded_path = f"/results/expert_review_blinded_{model_key}.json"
    with open(blinded_path, "w") as f:
        json.dump(blinded_packages, f, indent=2)
    vol.commit()

    print(f"\n✓ Full packages saved to {output_path}")
    print(f"✓ Blinded packages saved to {blinded_path}")

    # Summary statistics
    stratum_counts = {s: len(v) for s, v in selected.items()}
    category_counts = {}
    for pkg in review_packages:
        cat = pkg.get("scenario_category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    summary = {
        "model": model_key,
        "total_sessions": len(all_sessions),
        "total_selected": total_selected,
        "stratum_counts": stratum_counts,
        "category_counts": category_counts,
    }

    print(f"\n  Stratum distribution: {stratum_counts}")
    print(f"  Category distribution: {category_counts}")

    return summary


@app.local_entrypoint()
def main(model: str = "all"):
    """
    Select sessions for expert review.

    Usage:
        modal run step4_select_review_sessions.py
        modal run step4_select_review_sessions.py --model llama3
    """
    if model == "all":
        models_to_run = MODELS
    else:
        models_to_run = [model]

    print(f"\n{'='*60}")
    print(f"EXPERT REVIEW SESSION SELECTION")
    print(f"{'='*60}")
    print(f"Models: {models_to_run}")
    print(f"Per model: {sum(N_PER_STRATUM.values())} sessions")
    print(f"{'='*60}\n")

    for mk in models_to_run:
        result = select_sessions.remote(mk)
        if "error" not in result:
            print(f"\n✓ {mk}: {result['total_selected']} sessions selected from {result['total_sessions']}")

    print(f"\n{'='*60}")
    print(f"DONE — Review packages saved to Modal volume")
    print(f"Download with: modal volume get ai-persona-results expert_review_blinded_{{model}}.json")
    print(f"{'='*60}")
