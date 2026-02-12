# Archive Directory

This folder contains historical files preserved for reproducibility and reference. These files are **not actively used** by the current codebase but document the evolution of the project.

## Contents

### Root Level

| File | Purpose | Status |
|------|---------|--------|
| `STEERING_LESSONS_LEARNED.md` | Technical lessons from steering iterations | Referenced in `03_code/README.md` |
| `research_document_activation_steering_v2.md` | Earlier paper draft (v3+ is current) | Historical |
| `research_plan_v2.md` | Original research planning document | Superseded by `05_docs/workplan2.0.md` |
| `mental_health_datasets_for_drift_testing.md` | Dataset reference notes | Historical |
| `validation_datasets_reference.md` | Dataset reference notes | Historical |
| `phase2c_deep_research_synthesis.md` | Root cause analysis synthesis | Referenced in `04_results/` |

### `old/` Subdirectory

Contains early experimental scripts and documentation:

- `anchor_steering_diagnostics.py` - Early diagnostic implementation
- `results_report.md` - Historical results documentation
- `theory_and_diagnostics_appendix.md/.pdf` - Technical appendix

## Why Keep These?

1. **Reproducibility**: Historical scripts enable reproducing earlier experimental results
2. **Version History**: Documents methodology evolution for publication transparency
3. **Reference**: Contains lessons learned that inform current implementation

## What Was Removed

The following unused files were removed (Feb 2026):

- `v5_results/` - Empty folder
- `old/lemma_explained_for_beginners_quarto_files/` - Quarto build artifacts
- `old/DEVELOPMENT_PLAN_draft.md` - Superseded draft

## See Also

- `03_code/archive/` - Contains historical code iterations (steering v5-v29, old validation/drift scripts)
- `03_code/archive/trait_definitions_expanded.json` - **Still in use** by `step1_validate_traits.py`
