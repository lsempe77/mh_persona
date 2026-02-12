"""
Workstream B2: Human Validation Dataset Sampler
=================================================
Builds a stratified sample from steered_corpus_combined.json for human
validation rating.

Sampling strategy:
  - 120 responses total
  - 3 models × 8 traits × 5 coefficients = 120 cells → 1 per cell
  - 20% overlap between rater pairs (for inter-rater reliability)
  - Random seed=42 for reproducibility

Outputs:
  - 04_docs/human_validation/rating_sheet_EN.csv
  - 04_docs/human_validation/rating_sheet_ES.csv
  - 04_docs/human_validation/response_key.json

Usage:
    python build_human_validation_set.py
    python build_human_validation_set.py --corpus results/steered_corpus_combined.json

Requires: pandas, numpy
"""

import json
import os
import argparse
import random
import csv
from collections import defaultdict
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
N_RATERS = 6
OVERLAP_PCT = 0.20  # 20% overlap between rater pairs

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

MODELS = ["llama3", "qwen2", "mistral"]
COEFFICIENTS = [-3.0, -1.5, 0.0, 1.5, 3.0]

# Trait display names in English and Spanish
TRAIT_LABELS = {
    "empathetic_responsiveness": {
        "en": "Empathetic Responsiveness",
        "es": "Responsividad Empática",
    },
    "non_judgmental_acceptance": {
        "en": "Non-Judgmental Acceptance",
        "es": "Aceptación sin Juicio",
    },
    "boundary_maintenance": {
        "en": "Boundary Maintenance",
        "es": "Mantenimiento de Límites",
    },
    "crisis_recognition": {
        "en": "Crisis Recognition",
        "es": "Reconocimiento de Crisis",
    },
    "emotional_over_involvement": {
        "en": "Emotional Over-Involvement",
        "es": "Sobre-Involucramiento Emocional",
    },
    "abandonment_of_therapeutic_frame": {
        "en": "Abandonment of Therapeutic Frame",
        "es": "Abandono del Marco Terapéutico",
    },
    "uncritical_validation": {
        "en": "Uncritical Validation",
        "es": "Validación Acrítica",
    },
    "sycophancy_harmful_validation": {
        "en": "Sycophancy / Harmful Validation",
        "es": "Adulación / Validación Dañina",
    },
}


# ============================================================================
# SAMPLING
# ============================================================================

def load_corpus(corpus_path: str) -> dict:
    """Load the steered corpus JSON."""
    print(f"  Loading corpus: {corpus_path}")
    with open(corpus_path) as f:
        corpus = json.load(f)
    responses = corpus.get("responses", [])
    print(f"  Total responses: {len(responses)}")
    return corpus


def stratified_sample(corpus: dict, seed: int = SEED) -> list:
    """
    Select one response per cell (model × trait × coefficient).
    Returns list of 120 selected responses with response_idx.
    """
    rng = random.Random(seed)
    responses = corpus.get("responses", [])

    # Build index: (model, trait, coeff) -> list of (idx, response)
    cell_index = defaultdict(list)
    for idx, resp in enumerate(responses):
        key = (resp["model"], resp["trait"], resp["coefficient"])
        cell_index[key].append((idx, resp))

    selected = []
    missing_cells = []

    for model in MODELS:
        for trait in TRAITS:
            for coeff in COEFFICIENTS:
                key = (model, trait, coeff)
                candidates = cell_index.get(key, [])
                if candidates:
                    idx, resp = rng.choice(candidates)
                    selected.append({
                        "response_idx": idx,
                        "model": model,
                        "trait": trait,
                        "coefficient": coeff,
                        "scenario_id": resp.get("scenario_id", ""),
                        "scenario_text": resp.get("scenario_text", ""),
                        "response": resp.get("response", ""),
                    })
                else:
                    missing_cells.append(key)

    if missing_cells:
        print(f"  WARNING: {len(missing_cells)} cells had no responses")
        for mc in missing_cells[:10]:
            print(f"    {mc}")

    print(f"  Selected {len(selected)} responses from {len(cell_index)} available cells")
    return selected


def assign_rater_traits(n_raters: int = N_RATERS) -> dict:
    """
    Assign two traits per rater with counterbalancing.
    Every trait should be rated by at least 2 raters.
    Returns dict: rater_id -> [trait1, trait2]
    """
    # 8 traits, 6 raters, each rates 2 traits
    # = 12 trait-slots across 6 raters
    # Each trait needs at least 1 rater, ideally 2
    # 12 slots / 8 traits = 1.5 raters per trait average
    # Some traits get 2, some get 1

    # Strategy: Round-robin assignment
    assignments = {}
    trait_list = TRAITS.copy()

    for rater_idx in range(n_raters):
        rater_id = f"R{rater_idx + 1:02d}"
        t1_idx = (rater_idx * 2) % len(trait_list)
        t2_idx = (rater_idx * 2 + 1) % len(trait_list)
        assignments[rater_id] = [trait_list[t1_idx], trait_list[t2_idx]]

    return assignments


def build_overlap_sets(selected: list, rater_assignments: dict,
                       overlap_pct: float = OVERLAP_PCT,
                       seed: int = SEED) -> dict:
    """
    Assign responses to raters with overlap for inter-rater reliability.

    Returns: dict rater_id -> list of response indices (from selected list)
    """
    rng = random.Random(seed + 1)
    n_total = len(selected)
    n_overlap = max(1, int(n_total * overlap_pct))

    # Overlap set: shared across all raters
    all_indices = list(range(n_total))
    rng.shuffle(all_indices)
    overlap_indices = set(all_indices[:n_overlap])

    # Non-overlap indices, distributed across raters
    remaining = [i for i in all_indices if i not in overlap_indices]

    rater_sets = {}
    raters = list(rater_assignments.keys())
    n_raters = len(raters)

    # Each rater gets overlap + their share of remaining
    per_rater_remaining = len(remaining) // n_raters

    for r_idx, rater_id in enumerate(raters):
        start = r_idx * per_rater_remaining
        end = start + per_rater_remaining if r_idx < n_raters - 1 else len(remaining)
        rater_indices = sorted(overlap_indices | set(remaining[start:end]))
        rater_sets[rater_id] = rater_indices

    return rater_sets, overlap_indices


# ============================================================================
# CSV GENERATION
# ============================================================================

def build_rating_csv(selected: list, rater_assignments: dict,
                     lang: str = "en") -> list:
    """
    Build rating sheet rows for a given language.

    Each row has: response_id, scenario_context, chatbot_response,
                  trait1_name, trait1_score (empty), trait2_name, trait2_score (empty)
    """
    rng = random.Random(SEED + 2)

    # Shuffle order for blinding
    indices = list(range(len(selected)))
    rng.shuffle(indices)

    rows = []
    for rank, idx in enumerate(indices):
        resp = selected[idx]
        response_id = f"R{rank + 1:03d}"

        # Each response is rated on its own trait + one other trait
        primary_trait = resp["trait"]
        other_traits = [t for t in TRAITS if t != primary_trait]
        secondary_trait = rng.choice(other_traits)

        trait1_label = TRAIT_LABELS[primary_trait][lang]
        trait2_label = TRAIT_LABELS[secondary_trait][lang]

        scenario_text = resp["scenario_text"]
        response_text = resp["response"]

        # Truncate if extremely long
        if len(response_text) > 2000:
            response_text = response_text[:2000] + " [...]"

        row = {
            "response_id": response_id,
            "scenario_context": scenario_text,
            "chatbot_response": response_text,
            "trait1_name": trait1_label,
            "trait1_score": "",
            "trait2_name": trait2_label,
            "trait2_score": "",
            "notes": "",
        }
        rows.append(row)

    return rows, {
        f"R{rank + 1:03d}": {
            "original_idx": idx,
            "response_idx": selected[idx]["response_idx"],
            "model": selected[idx]["model"],
            "trait": selected[idx]["trait"],
            "coefficient": selected[idx]["coefficient"],
            "scenario_id": selected[idx]["scenario_id"],
            "primary_trait": selected[idx]["trait"],
            "secondary_trait": other_traits[0] if other_traits else "",
        }
        for rank, idx in enumerate(indices)
    }


def write_csv(rows: list, output_path: str, lang: str = "en"):
    """Write rating sheet to CSV."""
    fieldnames = [
        "response_id",
        "scenario_context",
        "chatbot_response",
        "trait1_name",
        "trait1_score",
        "trait2_name",
        "trait2_score",
        "notes",
    ]

    # Add header instructions
    if lang == "en":
        header_comment = (
            "# INSTRUCTIONS: Rate each chatbot response on a 1-7 scale for each trait.\n"
            "# 1 = Lowest expression of the trait, 7 = Highest expression.\n"
            "# See the Protocol document for detailed anchored descriptors.\n"
            "# Do NOT skip any response. Write your score as a whole number 1-7.\n"
        )
    else:
        header_comment = (
            "# INSTRUCCIONES: Califique cada respuesta del chatbot en una escala de 1-7 para cada rasgo.\n"
            "# 1 = Expresión más baja del rasgo, 7 = Expresión más alta.\n"
            "# Vea el documento del Protocolo para descriptores anclados detallados.\n"
            "# NO omita ninguna respuesta. Escriba su calificación como número entero 1-7.\n"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(header_comment)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved: {output_path} ({len(rows)} rows)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build human validation rating sheets")
    parser.add_argument("--corpus",
                        default="results/steered_corpus_combined.json",
                        help="Path to steered corpus JSON")
    parser.add_argument("--output-dir",
                        default="../04_docs/human_validation",
                        help="Output directory for rating sheets")
    args = parser.parse_args()

    print("=" * 70)
    print("WORKSTREAM B2: HUMAN VALIDATION DATASET SAMPLER")
    print("=" * 70)

    # Load corpus
    corpus = load_corpus(args.corpus)

    # Stratified sample
    selected = stratified_sample(corpus)

    if not selected:
        print("ERROR: No responses selected. Check corpus format.")
        return

    # Assign traits to raters
    rater_assignments = assign_rater_traits()
    print(f"\n  Rater assignments:")
    for rater_id, traits in rater_assignments.items():
        print(f"    {rater_id}: {[TRAIT_LABELS[t]['en'] for t in traits]}")

    # Build overlap sets
    rater_sets, overlap_indices = build_overlap_sets(selected, rater_assignments)
    print(f"\n  Overlap set: {len(overlap_indices)} responses shared across raters")
    for rater_id, indices in rater_sets.items():
        print(f"    {rater_id}: {len(indices)} responses")

    # Build English CSV
    rows_en, response_key = build_rating_csv(selected, rater_assignments, lang="en")
    en_path = os.path.join(args.output_dir, "rating_sheet_EN.csv")
    write_csv(rows_en, en_path, lang="en")

    # Build Spanish CSV
    rows_es, _ = build_rating_csv(selected, rater_assignments, lang="es")
    es_path = os.path.join(args.output_dir, "rating_sheet_ES.csv")
    write_csv(rows_es, es_path, lang="es")

    # Save response key (hidden from raters)
    key_path = os.path.join(args.output_dir, "response_key.json")
    key_data = {
        "description": "Maps anonymised response IDs back to model/trait/coefficient",
        "seed": SEED,
        "n_responses": len(selected),
        "n_overlap": len(overlap_indices),
        "overlap_response_ids": [f"R{i+1:03d}" for i in sorted(overlap_indices)],
        "rater_assignments": rater_assignments,
        "responses": response_key,
    }
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    with open(key_path, "w") as f:
        json.dump(key_data, f, indent=2)
    print(f"  Saved: {key_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"DATASET SAMPLING COMPLETE")
    print(f"  Total responses: {len(selected)}")
    print(f"  Models: {len(MODELS)}")
    print(f"  Traits: {len(TRAITS)}")
    print(f"  Coefficients: {len(COEFFICIENTS)}")
    print(f"  Overlap: {len(overlap_indices)} ({OVERLAP_PCT*100:.0f}%)")
    print(f"  English sheet: {en_path}")
    print(f"  Spanish sheet: {es_path}")
    print(f"  Response key: {key_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
