# Code Archive

This folder contains historical code iterations preserved for reproducibility. These files are **not actively used** but document the methodology evolution.

## Structure

```
archive/
├── steering_iterations/     # Steering validation iterations v5-v29
├── old_validation/          # Superseded validation scripts
├── old_drift/               # Superseded drift tracking scripts
├── old_analysis/            # Superseded analysis scripts
├── trait_definitions_*.json # Trait configuration files
└── *.md                     # Planning documents
```

## Active Files (DO NOT DELETE)

| File | Used By |
|------|---------|
| `trait_definitions_expanded.json` | `step1_validate_traits.py` (uploaded to Modal volume) |

## Historical Iterations

### `steering_iterations/`

Contains 25+ steering validation iterations (v5-v29). Key milestones:

- **v9**: First LLM-judge implementation
- **v14-v16**: Per-trait layer optimization
- **v19**: Systematic layer search
- **v20-v21**: Polarity fixes
- **v24**: Production version
- **v25-v29**: Multi-model extensions

### `old_validation/`, `old_drift/`, `old_analysis/`

Earlier implementations superseded by current pipeline:
- `step1_validate_traits.py`
- `step1b_contrastive_probing.py`
- `step2_monitor_drift.py`

## Archived Scripts (Feb 2026)

**From `03_code/` main folder:**
- `generate_steered_examples.py` - One-off example generation
- `download_and_analyze_corpus.py` - One-off corpus download
- `analyze_all_results.py` - Superseded aggregation script
- `modal_context_erosion.py` - Superseded by `modal_context_erosion_v2.py`
- `generate_steered_corpus.py` - Superseded by `generate_steered_corpus_v2.py`
- `generate_steered_corpus_v3_patch.py` - One-off patch script

**From `03_code/results/`:**
- `check_erosion.py`, `check_session.py` - Debug scripts
- `gather_all_results.py` - One-off aggregation
- `inspect_8bit.py`, `quant_for_paper.py` - One-off analysis

## What Was Removed (Deleted)

**From `old_analysis/`:**
- `analyze_v16.py`, `analyze_v8.py` - Version-specific analysis
- `check_v14_scores.py`, `check_v9.py` - Debug scripts
- `debug_esconv.py`, `test_extraction.py` - One-off scripts

**From root:**
- `test_sycophancy_prompts.py`, `test_measured_pacing_prompts.py` - Test files
- `uncritical_validation_comparison.png` - Orphaned image

## Why Keep Historical Code?

1. **Reproducibility**: Enables reproducing any published result
2. **Methodology Transparency**: Documents evolution for peer review
3. **Debugging**: Useful for understanding regressions

See `STEERING_LESSONS_LEARNED.md` in parent archive for key technical learnings.
