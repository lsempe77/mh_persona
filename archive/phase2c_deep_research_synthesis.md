# Phase 2c: Deep Research Synthesis — Why Steering Fails on Qwen2/Mistral

**Date:** Feb 2026  
**Analyses conducted:** 10 systematic tests  
**Conclusion:** The template-based steering vectors are geometrically incompatible with Qwen2/Mistral's representation spaces. Three concrete solutions proposed.

---

## Current Scorecard (r > 0.3 criterion)

| Model | Working | Close (0.2-0.3) | Failed (<0.2) |
|-------|---------|-----------------|----------------|
| Llama3-8B | **8/8** | 0 | 0 |
| Qwen2-7B | **3/8** | 2 | 3 |
| Mistral-7B | **2/8** | 4 | 2 |

---

## Part 1: What We Ruled Out

### ❌ Layer selection is NOT the problem
For every failing model×trait combo, we tested ALL 12 layers (8-19). No hidden good layer exists. The pipeline already picks the best available layer per model.

### ❌ Score compression is a confound, NOT a cause
Partial correlation analysis (the definitive test):
- `r vs is_llama | controlling for beh_std`: r=0.538, p=0.007 → **model effect persists**
- `r vs beh_std | controlling for is_llama`: r=0.237, p=0.265 → **beh_std effect disappears**

Translation: Even when Qwen2/Mistral produce the SAME amount of behavioral variation as Llama3, their r-values are still worse. The problem is not in the behavioral scores.

### ❌ Spearman rank correlation doesn't help
Mean Pearson: 0.296, Mean Spearman: 0.264. Non-normality is not the issue.

### ❌ Model-specific prompts didn't help
Phase 2b created model-specific trait definitions (tighter clustering for Qwen2, shorter phrasing for Mistral). Result: Zero improvement. Still 3/8 and 2/8.

---

## Part 2: What IS the Problem

### ✅ The steering vectors point in wrong directions for Qwen2/Mistral

**Evidence 1: Cross-model activation transfer test (the "killer test")**

We tested whether Llama3's activation projections predict Qwen2/Mistral's behavioral drift better than their OWN activations:

| Trait | Q→Q (self) | L→Q (cross) | L beats Q? |
|-------|-----------|-------------|------------|
| empathetic_responsiveness | +0.240 | **+0.278** | ★ YES |
| non_judgmental_acceptance | +0.140 | **+0.285** | ★ YES |
| uncritical_validation | +0.020 | **+0.176** | ★ YES |
| sycophancy_harmful_validation | +0.128 | **+0.174** | ★ YES |

For 4/5 failing Qwen2 traits, Llama3's vector captures Qwen2's behavior better than Qwen2's own vector. **The Qwen2 vectors are literally worse than a foreign model's vectors.**

**Evidence 2: Prompt cosine similarity predicts r-value (p=0.008)**

| Metric | Working (r>0.3) | Failing (r<0.3) | p-value |
|--------|----------------|-----------------|---------|
| High prompt cosine mean | 0.888 | 0.835 | 0.008* |
| Low prompt cosine mean | 0.868 | 0.810 | 0.001** |

When the 5 template prompts don't cluster tightly in a model's representation space, the averaged vector is noisy/wrong.

### ✅ The root cause is different representation geometries

Each model encodes the same semantic content differently:

**Llama3:** Template prompts cluster tightly (cosine 0.87-0.93, within-class variance 0.005-0.01). Averaging 5 prompts gives a clean, consistent direction vector.

**Qwen2:** Prompts point in similar directions (cosine 0.82-0.92) BUT have enormous magnitude variation (within-class variance 0.30-0.55, i.e., **40-60x higher than Llama3**). The average is dominated by whichever prompt happens to have the largest magnitude.

**Mistral:** Prompts don't even point in the same direction (cosine drops as low as 0.58). Averaging 5 prompts that point in different directions produces a vector pointing at their geometric average — which may not correspond to any meaningful concept.

---

## Part 3: Why Data-Driven Vectors Won't Work Either

The natural idea: instead of 5 template prompts, extract hidden states from the model's own responses that the judge scored high vs low, and build the vector from those.

**Feasibility check:**

| Trait | Qwen2 N(≥6) | Qwen2 N(≤2) | Mistral N(≥6) | Mistral N(≤2) |
|-------|------------|------------|--------------|--------------|
| emotional_over_involvement | **0** | 915 | 5 | 851 |
| uncritical_validation | **2** | 498 | 11 | 492 |
| sycophancy_harmful_validation | **6** | 560 | 17 | 579 |

For the most important traits (the "bad" behaviors), RLHF-aligned models almost never exhibit high-trait behavior. You can't build a contrastive vector with 0 examples of one pole.

---

## Part 4: Proposed Solutions (Ranked by Feasibility)

### Solution A: Increase Number of Template Prompts (Low Cost, Moderate Impact)

**Rationale:** With only 5 prompts per class, a single outlier can corrupt the mean vector. Increasing to 15-20 prompts would:
- Average out outlier prompts
- Better sample the concept space
- Reduce the impact of any single prompt pointing in a wrong direction

**Implementation:**
1. Generate 20 high + 20 low prompts per trait (GPT-4 can help)
2. Extract hidden states from all 40 prompts per model
3. **NEW: Filter prompts by cosine consistency** — drop any prompt with cosine sim < 0.80 relative to the group mean
4. Build vector from remaining prompts only

**Expected effect:** 
- Mistral: Should help a lot (reduces impact of outlier prompts with 0.58 cosine)
- Qwen2: Moderate help (reduces magnitude variation through more averaging)

**Cost:** ~$5 Modal run per model (same as current pipeline, just more prompts)

**Why it might work:** Mistral has 4 traits at r=0.2-0.3 (close to threshold). Just cleaning up the vector direction might push them over.

---

### Solution B: Contrastive Probing (Moderate Cost, High Impact)

**Rationale:** Instead of template prompts, train a LINEAR PROBE on the model's actual responses. This replaces hand-crafted vectors with learned ones.

**Method:**
1. From the 500 existing scenarios, we have 1000 responses per trait (t1 + tN) with judge scores
2. Binarize scores: ≥5 = "high", ≤3 = "low" (broader thresholds to get enough examples)
3. On a NEW Modal run, save the full hidden states (not just projections) for ~200 selected responses
4. Train a logistic regression: hidden_state → high/low
5. The regression weight vector IS the steering direction

**Key advantage:** The vector is learned FROM the model's own representations of actual therapeutic conversations, not from short template utterances.

**Score distribution with relaxed thresholds (≥5 vs ≤3):**

| Trait | Qwen2 N(≥5) | Qwen2 N(≤3) | Mistral N(≥5) | Mistral N(≤3) |
|-------|------------|------------|--------------|--------------|
| emotional_over_involvement | ~85 | ~915 | ~144 | ~851 |
| uncritical_validation | ~14 | ~998 | ~18 | ~989 |

Still problematic for the most extreme traits. Would need to use ≥4 vs ≤2 or similar.

**Cost:** ~$10 Modal run (need to save full hidden states)

---

### Solution C: Principal Component Approach (Low Cost, Unknown Impact)

**Rationale:** Instead of a SINGLE contrastive direction, use PCA on the template prompt activations to find the dimension of maximum variance WITHIN each class, then subtract it before computing the contrastive direction.

**Why:** The high within-class variance in Qwen2 (0.30-0.55) suggests the template prompts vary along irrelevant dimensions (syntax, length, word choice) in addition to the trait dimension. PCA can isolate and remove these.

**Method:**
1. Extract hidden states from 5 high + 5 low prompts (same as current)
2. Within each class, compute PCA and remove the top-k components (likely k=1-2)
3. Compute contrastive direction from the residuals
4. This "denoised" direction should be more focused on the actual trait

**Cost:** Zero new Modal runs — can be applied to existing vector diagnostics data

---

### Solution D: Accept Cross-Model Ceiling and Reframe (Zero Cost)

**Rationale:** The behavioral drift correlations across models cap what's achievable:

| Trait | L_behav ↔ Q_behav | L_behav ↔ M_behav |
|-------|-------------------|-------------------|
| empathetic_responsiveness | 0.532 | 0.266 |
| non_judgmental_acceptance | 0.610 | 0.391 |
| emotional_over_involvement | 0.274 | 0.236 |
| sycophancy_harmful_validation | 0.373 | 0.215 |

Even with PERFECT vectors, if two models only agree 27% on what "emotional over-involvement" looks like (r=0.274), cross-model generalization may be fundamentally limited.

**Reframe for publication:**
- Present Llama3 as the primary validation (8/8, strong evidence)
- Present Mistral/Qwen2 as revealing that activation geometry varies by architecture
- This IS a finding — it tells us monitoring must be model-specific
- Combine with Solution A (more prompts) to push Mistral from 2/8 to 4-5/8

---

## Recommended Path Forward

1. **Immediate (Solution A):** Expand to 20 prompts per class + cosine filtering. Cheapest, most likely to improve Mistral (4 traits at r=0.2-0.3).

2. **If A insufficient (Solution C):** Apply PCA denoising. Zero cost, can be tested offline with existing data.

3. **If still insufficient (Solution B):** Contrastive probing. More expensive but fundamentally different approach.

4. **For publication (Solution D):** Frame cross-model variation as a finding, not a failure. Target Llama3 (8/8) + Mistral improved to 5+/8 as the paper's evidence base.

---

## Raw Data for Reference

### Vector Diagnostics at Best Layer (All 24 model×trait combos)

```
Model    Trait                              L      r  Sep   HVar    LVar   HCos  LCos    SNR
─────────────────────────────────────────────────────────────────────────────────────────────
llama3   sycophancy_harmful_validation     19 +0.489  8.4  0.010   0.011  0.895 0.877  819.2
llama3   abandonment_therapeutic_frame     19 +0.470  9.1  0.012   0.013  0.870 0.854  736.4
llama3   emotional_over_involvement        19 +0.441  7.9  0.009   0.010  0.913 0.883  818.4
llama3   empathetic_responsiveness         17 +0.424  7.3  0.006   0.007  0.909 0.874 1152.9
llama3   crisis_recognition                19 +0.374  5.3  0.009   0.010  0.910 0.886  567.7
llama3   uncritical_validation             18 +0.364  6.8  0.007   0.009  0.902 0.864  810.8
llama3   non_judgmental_acceptance         18 +0.346  6.4  0.009   0.009  0.872 0.869  686.9
llama3   boundary_maintenance              18 +0.302  8.7  0.006   0.006  0.903 0.918 1368.8

qwen2    abandonment_therapeutic_frame     19 +0.400 44.0  0.532   0.511  0.839 0.845   84.4
qwen2    emotional_over_involvement        19 +0.369 42.0  0.380   0.487  0.898 0.857   96.8
qwen2    crisis_recognition                19 +0.346 30.1  0.457   0.428  0.891 0.902   68.0
qwen2    boundary_maintenance              14 +0.257 35.3  0.194   0.191  0.872 0.870  183.0
qwen2    empathetic_responsiveness         19 +0.240 57.4  0.297   0.462  0.918 0.847  151.2
qwen2    non_judgmental_acceptance         19 +0.140 40.9  0.524   0.461  0.874 0.888   83.2
qwen2    sycophancy_harmful_validation     19 +0.128 48.0  0.481   0.552  0.871 0.837   92.9
qwen2    uncritical_validation             19 +0.020 46.4  0.420   0.520  0.899 0.845   98.7

mistral  boundary_maintenance              15 +0.355  4.6  0.002   0.002  0.865 0.835 2460.8
mistral  empathetic_responsiveness         19 +0.329  7.8  0.007   0.008  0.874 0.825 1039.0
mistral  crisis_recognition                18 +0.268  4.1  0.008   0.007  0.801 0.789  550.7
mistral  non_judgmental_acceptance          9 +0.257  1.5  0.001   0.001  0.729 0.746 2222.2
mistral  abandonment_therapeutic_frame     19 +0.233  7.5  0.013   0.009  0.769 0.783  683.8
mistral  uncritical_validation             11 +0.208  2.1  0.001   0.001  0.764 0.736 1911.6
mistral  sycophancy_harmful_validation     19 +0.176  8.0  0.010   0.011  0.824 0.762  780.8
mistral  emotional_over_involvement        19 +0.168  6.4  0.007   0.010  0.857 0.804  747.8
```

Key patterns:
- **Llama3:** All HCos > 0.87, all LCos > 0.85, all WVar < 0.013
- **Qwen2:** HCos decent (0.84-0.92) but WVar 30-60x higher → magnitude problem
- **Mistral:** HCos/LCos drops as low as 0.73-0.76 → directional problem
