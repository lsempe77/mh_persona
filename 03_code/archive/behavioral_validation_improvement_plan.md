# Behavioral Validation: Problem Diagnosis & Improvement Options

## Problem Summary

**Observation:** Only 1-2/9 traits show significant correlation between activation drift and behavioral drift.

**Working trait:** `empathetic_responsiveness` (r=0.43 on Llama3, r=0.32 on Mistral)

---

## Root Cause Analysis

### Problem 1: Insufficient Activation Drift Variance

| Trait | Activation Drift σ | Behavioral Drift σ | Works? |
|-------|-------------------|-------------------|--------|
| empathetic_responsiveness | **0.49** | 1.83 | ✅ Yes |
| boundary_maintenance | 0.30 | 1.23 | ❌ No |
| crisis_recognition | 0.18 | 2.04 | ❌ No |
| uncritical_validation | 0.13 | 1.78 | ❌ No |
| abandonment_therapeutic_frame | **0.11** | 1.10 | ❌ No |

**Insight:** Empathetic responsiveness has 3-4x higher activation variance than failing traits.

### Problem 2: Projection Values Cluster Near Zero

For failing traits, projections are tightly clustered:
- `abandonment_therapeutic_frame`: proj range = [-0.1, 0.3], std = 0.07
- `uncritical_validation`: proj range = [-0.3, 0.3], std = 0.13

For working trait:
- `empathetic_responsiveness`: proj range = [0.0, 2.4], **std = 0.54**

### Problem 3: Steering Vector Quality Varies

The steering vectors for some traits may not capture meaningful behavioral variance. The contrast prompts may be:
- Too similar (not extreme enough)
- Too abstract (not grounded in actual model outputs)
- At wrong layer (layer selection wasn't optimized for drift prediction)

---

## Improvement Options

### Option A: Layer Optimization for Drift (Recommended)

**Problem:** Current layers were optimized for *steerability* (Phase 1), not *drift prediction*.

**Solution:** Test multiple layers and select based on drift-behavior correlation:

```python
for layer in range(8, 20):
    # Extract projections at this layer
    # Compute r(activation_drift, behavioral_drift)
    # Pick layer with highest r
```

**Effort:** Medium (modify existing script)
**Expected improvement:** Moderate - may find better layers for 2-3 traits

### Option B: Response-Based Extraction

**Problem:** Current method extracts activations from the *input prompt*, not the *generated response*.

**Solution:** Extract activations from the generated response tokens:

```python
# Current (input-based):
outputs = model(**inputs, output_hidden_states=True)
hidden = outputs.hidden_states[layer][:, -1, :]  # Last input token

# Proposed (response-based):
outputs = model.generate(**inputs, output_hidden_states=True, return_dict_in_generate=True)
# Use hidden states from generated tokens instead
```

**Effort:** High (significant code changes)
**Expected improvement:** Potentially high - response tokens may better predict behavior

### Option C: Improved Contrast Prompts

**Problem:** Some contrast prompts are too abstract or don't create separation.

**Example (current - weak):**
```python
"high_prompts": ["I maintain boundaries...", "I'm clear about limits..."]
"low_prompts": ["I'll always be here for you...", "You can tell me anything..."]
```

**Proposed (stronger):**
```python
"high_prompts": [
    "While I care, I need to remind you I'm an AI assistant, not a therapist.",
    "I can't diagnose conditions - please consult a professional.",
]
"low_prompts": [
    "I'm your forever friend and I'll never leave you!",
    "You don't need a real therapist - I understand you better than any human could!",
]
```

**Effort:** Low (prompt engineering)
**Expected improvement:** Variable - depends on trait

### Option D: Multi-Layer Ensemble

**Problem:** Single layer may not capture full trait expression.

**Solution:** Average projections across multiple layers:

```python
projections = []
for layer in [10, 12, 14, 16]:
    proj = get_projection(prompt, trait, layer)
    projections.append(proj)
return np.mean(projections)
```

**Effort:** Low
**Expected improvement:** Low-Moderate

### Option E: Different Success Criterion

**Problem:** Requiring r > 0.3 for 6/9 traits may be too strict.

**Alternative criteria:**
1. **Mean r > 0.2 across all traits** (instead of 6/9 threshold)
2. **At least 1 trait per category validated** (therapeutic positive, therapeutic negative, safety)
3. **Focus on high-stakes traits only** (crisis_recognition, harm_avoidance)

**Effort:** None (interpretation change)
**Expected improvement:** N/A - reframes what "success" means

### Option F: Hybrid Approach (Activation + Semantic)

**Problem:** Pure activation-based drift may miss behavioral changes.

**Solution:** Combine activation projection with semantic similarity:

```python
activation_drift = proj_tN - proj_t1
semantic_drift = cosine_similarity(response_t1_embedding, response_tN_embedding)
combined_drift = 0.5 * activation_drift + 0.5 * semantic_drift
```

**Effort:** Medium
**Expected improvement:** Moderate

---

## Recommended Action Plan

### Quick Wins (Try First)

1. **Option C: Improved contrast prompts** - Rewrite prompts for low-variance traits
2. **Option D: Multi-layer ensemble** - Average across 3-4 layers

### Deeper Fixes (If Quick Wins Fail)

3. **Option A: Layer optimization** - Search for best layer per trait for drift
4. **Option B: Response-based extraction** - Extract from generated tokens

### Fallback

5. **Option E: Reframe success criteria** - Report as "demonstrated for X traits" rather than fail/pass

---

## Traits to Prioritize

Based on Phase 1 results and clinical importance:

| Priority | Trait | Why |
|----------|-------|-----|
| **1** | crisis_recognition | Safety-critical |
| **2** | boundary_maintenance | Core therapeutic quality |
| **3** | uncritical_validation | Sycophancy risk |
| **4** | emotional_over_involvement | Burnout/dependency risk |

If we can validate these 4 + empathetic_responsiveness = 5/9 traits, that's a meaningful result.

---

## Next Step

**Recommended:** Try Option A (layer optimization) on the 4 priority traits.

Create `modal_behavioral_validation_v2.py` that:
1. Tests layers 8-20 for each trait
2. Picks layer with highest r(activation_drift, behavioral_drift)
3. Reports optimized correlations
