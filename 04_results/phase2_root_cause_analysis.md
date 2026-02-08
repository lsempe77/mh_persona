# Phase 2 Root Cause Analysis: Why Steering Works on Llama3 but Fails on Qwen2/Mistral

**Date:** Phase 2a-2b Analysis (Cross-Model Validation)  
**Status:** COMPLETED  
**Key Finding:** Behavioral impact (|DIFF|) predicts r-value with r=0.899 (p<0.000001)

---

## Executive Summary

After extensive diagnostics, we identified the **single most important predictor** of steering success: whether moving along the steering vector actually changes the model's behavior.

| Metric | Correlation with r-value | p-value | Conclusion |
|--------|-------------------------|---------|------------|
| Separation (Cohen's d) | r = -0.374 | 0.072 | **WRONG predictor** |
| Within-class variance | r = -0.370 | 0.075 | Not useful |
| Mean cosine similarity | r = +0.386 | 0.062 | Not predictive |
| **|Behavioral difference|** | **r = +0.899** | **<0.000001** | **THE predictor** |

---

## Methodology

We analyzed 500 scenarios per model (1,500 total) across 8 traits:
- **Models:** Llama3-8B, Qwen2-7B, Mistral-7B
- **Metrics extracted:**
  - `activation_drift`: Dot product of response activation with steering vector
  - `behavioral_drift`: LLM-judged score change (1-7 scale → drift)
- **Key test:** For each trait/model, compute behavioral difference between top 25% and bottom 25% of activation values

---

## Key Findings

### Finding 1: Behavioral Impact is the Only Predictor

The correlation between **|behavioral_difference|** and **r-value** is **0.899** (p < 0.000001).

```
What matters is NOT:
  ✗ Separation between high/low activations (negatively correlated!)
  ✗ Within-class variance  
  ✗ Layer selection
  ✗ Extraction method (last-token vs mean)

What DOES matter:
  ✓ Whether the activation direction predicts behavioral change
```

### Finding 2: Qwen2's Vectors Point in Wrong Directions

For `uncritical_validation`:

| Model | Activation Range | Behav @ High Act | Behav @ Low Act | **DIFFERENCE** |
|-------|-----------------|------------------|-----------------|----------------|
| Llama3 | [-1.24, 0.93] | +0.736 | -0.472 | **1.208** |
| **Qwen2** | [-5.10, 3.65] | -0.105 | -0.098 | **-0.007** |
| Mistral | [-0.06, 0.10] | +0.496 | -0.266 | **0.762** |

**Qwen2 has 5x larger activation range but ZERO behavioral difference!**

The prompts produce large activation differences in Qwen2, but these differences exist along a direction that has no relationship to the actual trait behavior.

### Finding 3: Model-Specific Prompts Don't Fix This

Phase 2b tested:
- Qwen2: Tighter clustering, paraphrase-based prompts
- Mistral: Shorter, more direct prompts

**Result: No improvement.** The prompts cluster well (cosine similarity 0.84-0.92), but the clustered direction doesn't represent the trait in Qwen2's activation space.

### Finding 4: Vector Status by Model

| Status | Llama3 | Qwen2 | Mistral |
|--------|--------|-------|---------|
| OK (|diff| ≥ 0.7) | 8 | 3 | 7 |
| Weak (0.3 ≤ |diff| < 0.7) | 0 | 4 | 1 |
| Broken (|diff| < 0.3) | 0 | 1 | 0 |

---

## Trait-by-Trait Analysis

### Traits Working on All Models (3/8):
| Trait | Llama3 Diff | Qwen2 Diff | Mistral Diff |
|-------|-------------|------------|--------------|
| empathetic_responsiveness | +1.600 | +0.968 | +1.176 |
| crisis_recognition | +1.448 | +1.440 | +0.975 |
| abandonment_of_therapeutic_frame | +2.015 | +1.608 | +1.073 |

### Traits Failing on Qwen2 Only (4/8):
| Trait | Llama3 Diff | Qwen2 Diff | Mistral Diff |
|-------|-------------|------------|--------------|
| non_judgmental_acceptance | +1.288 | +0.512 ✗ | +1.120 |
| boundary_maintenance | +1.292 | +0.684 ✗ | +1.256 |
| uncritical_validation | +1.208 | -0.007 ✗✗ | +0.762 |
| sycophancy_harmful_validation | +2.384 | +0.416 ✗ | +0.863 |

### Traits Failing on Both (1/8):
| Trait | Llama3 Diff | Qwen2 Diff | Mistral Diff |
|-------|-------------|------------|--------------|
| emotional_over_involvement | +1.400 | +0.676 ✗ | +0.470 ✗ |

---

## Root Cause Hypothesis

The same English contrast prompts like:
- HIGH: "I validate everything you say"
- LOW: "Let me help you think critically"

Produce different effects in different models:

| Model | Effect of Contrast Prompts |
|-------|---------------------------|
| **Llama3** | Activations separate along trait-relevant dimension → steering works |
| **Qwen2** | Activations separate along irrelevant dimension (syntax? formality?) → steering fails |
| **Mistral** | Mostly works, but some traits have minimal activation variance |

**Possible reasons for Qwen2's different encoding:**
1. Trained on different data distribution (more Chinese text, different instruction tuning)
2. Different embedding space geometry
3. Trait concepts encoded in different representational structures

---

## Validated Cross-Model Trait Matrix

**⚠️ CORRECTION (Feb 2026):** Earlier version used |behavioral_difference| ≥ 0.7, which is NOT the validation criterion. The correct criterion is **Pearson r > 0.3** (activation_drift ↔ behavioral_drift).

Using **r > 0.3** (correct validation criterion):

| Trait | Llama3 | Qwen2 | Mistral |
|-------|--------|-------|---------|
| empathetic_responsiveness | ✓ (0.424) | ✗ (0.240) | ✓ (0.329) |
| non_judgmental_acceptance | ✓ (0.346) | ✗ (0.140) | ✗ (0.257) |
| boundary_maintenance | ✓ (0.302) | ✗ (0.257) | ✓ (0.355) |
| crisis_recognition | ✓ (0.374) | ✓ (0.346) | ✗ (0.268) |
| emotional_over_involvement | ✓ (0.441) | ✓ (0.369) | ✗ (0.168) |
| abandonment_of_therapeutic_frame | ✓ (0.470) | ✓ (0.400) | ✗ (0.233) |
| uncritical_validation | ✓ (0.364) | ✗ (0.020) | ✗ (0.208) |
| sycophancy_harmful_validation | ✓ (0.489) | ✗ (0.128) | ✗ (0.176) |
| **TOTAL** | **8/8** | **3/8** | **2/8** |

---

## Implications for Research

### 1. Cross-Model Generalization is Non-Trivial

Activation steering vectors DO NOT transfer between model families. The same contrast prompts produce behaviorally irrelevant activation directions in Qwen2.

### 2. New Metric Proposed: Behavioral Alignment Index (BAI)

```
BAI = |mean(behavior @ top25% activation) - mean(behavior @ bottom25% activation)|

If BAI < 0.3: Vector is BROKEN
If BAI < 0.7: Vector is WEAK  
If BAI ≥ 0.7: Vector is FUNCTIONAL
```

This metric has r=0.899 correlation with steering effectiveness.

### 3. Model-Specific Vector Discovery Required

To make steering work on Qwen2, we need:
1. Generate Qwen2 responses with explicit trait variation
2. Use PCA/contrastive methods on Qwen2's OWN activations
3. Discover Qwen2-specific steering directions

This is a significant methodological contribution beyond simple prompt engineering.

---

## Files and Data

**Data files (Modal volume `/results`):**
- `vector_diagnostics_{model}.json` - Per-layer vector quality metrics
- `raw_scenario_data_{model}.json` - Per-scenario activation/behavioral drifts
- `aggregated_{model}.json` - Summary statistics

**Local copies:**
- `03_code/vector_diagnostics_*.json`
- `03_code/raw_scenario_data_*.json`

---

## Recommended Next Steps

### Option A: Model-Specific Vector Discovery (Research Paper)
- Use Qwen2 to generate responses with explicit trait levels
- Extract activations from Qwen2's own outputs
- Discover trait directions empirically for each model
- **Value:** Novel contribution to activation steering literature

### Option B: Focus on Transferable Traits (Practical)
- Use the 3 traits that work across all models
- Document Llama3's full 8/8 success
- Present cross-model analysis as limitation/future work
- **Value:** Faster path to publication

### Option C: Mistral-First Approach
- Mistral achieves 2/8 with r>0.3, but 4 more traits are close (r=0.2-0.3)
- With improved vectors (more prompts + filtering), could reach 4-6/8
- Focus validation on Llama3 + improved Mistral
- Qwen2 as case study of failure mode
- **Value:** Two model families if Mistral can be improved

**See `phase2c_deep_research_synthesis.md` for detailed root cause analysis and solutions.**

---

## Appendix: Statistical Details

### Correlation Analysis (N=24 model×trait combinations)

| Predictor | r with r-value | p-value | 95% CI |
|-----------|---------------|---------|--------|
| Separation | -0.374 | 0.072 | [-0.66, 0.03] |
| Within-var | -0.370 | 0.075 | [-0.65, 0.04] |
| Mean cosine | +0.386 | 0.062 | [-0.02, 0.67] |
| SNR | +0.376 | 0.070 | [-0.03, 0.66] |
| **|Behav Diff|** | **+0.899** | **<0.001** | **[0.78, 0.96]** |
