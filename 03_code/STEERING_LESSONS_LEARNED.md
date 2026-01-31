# Activation Steering: Lessons Learned

## Project Goal
Build SOTA real-time monitoring system for AI persona drift in mental health chatbots.
Target: Lancet publication.

---

## WHAT WORKS

### 1. Last-Token Extraction (V9 Method)
```python
hidden = output[0] if isinstance(output, tuple) else output
activation = hidden[:, -1, :]  # LAST TOKEN ONLY
```
**Evidence:** V17 achieved r=0.466 (p=0.0006) for `non_judgmental_acceptance`

### 2. LLM-as-Judge Scoring (GPT-4o-mini via OpenRouter)
- More reliable than self-scoring with the same model
- Rubric-based 1-10 scoring with trait-specific anchors
- **Cost:** ~$0.01 per 50 responses

### 3. Layer Selection via Cohen's d (V8 Method)
```python
# Test layers 10-28, select highest Cohen's d per trait
cohens_d = (pos_mean - neg_mean) / pooled_std
```
- Different traits genuinely prefer different layers
- V8 found distribution: {14:1, 16:1, 17:3, 19:1, 23:1, 25:1, 26:2}

### 4. Moderate Steering Coefficients
- **Working range:** [-3.0, -1.5, 0.0, 1.5, 3.0]
- V18 tried [-5.0 to +5.0] → non_judgmental_acceptance DROPPED from 0.466 to 0.220
- Wider is NOT better

### 5. Multiple Contrast Prompts (5-10 per trait)
- Average directions from multiple prompt pairs
- Reduces noise from single-prompt artifacts

---

## WHAT DOESN'T WORK

### 1. ❌ Sentinel Token Extraction (V16 Method)
```python
# DON'T DO THIS
prompt = f"[INST] {prompt} [/INST] [PERSONA_SCORE]"
# Find sentinel position and extract there
```
**Evidence:** V16 got r=-0.081 for same trait V17 got r=0.466

### 2. ❌ Single Fixed Layer for All Traits
- V15 used layer 15 for everything → 0/6 working
- Traits encode at different depths

### 3. ❌ Self-Scoring (Circular Measurement)
- Model scores its own steered outputs → inflated correlations
- V8 proved this doesn't translate to real behavioral change

### 4. ❌ Very Wide Coefficient Ranges
- [-5.0, +5.0] overshoots optimal steering
- Can invert trait direction or destroy coherence

### 5. ❌ Arbitrary Layer Guessing
- "Let's try layer 15" without data → waste of compute
- Always use systematic layer search (Cohen's d)

---

## EXPERIMENTAL RESULTS SUMMARY

| Version | Method | Working Traits | Best Result |
|---------|--------|----------------|-------------|
| V8 | Cohen's d layers + self-score | 0/10 | r=0.27 (weak) |
| V9 | Cohen's d + LLM judge | 2/10 | r=0.375 non_judgmental |
| V15 | Single layer 15 | 0/6 | r=0.12 |
| V16 | Sentinel tokens | 0/6 | r=-0.08 (inverted!) |
| V17 | V9 method (last-token) | 1/6 | **r=0.466** non_judgmental |
| V18 | V17 + wider coeffs | 1/10 | **r=0.732** empathetic |

### Best Results Per Trait (across all versions)
| Trait | Best r | Version | Layer | Notes |
|-------|--------|---------|-------|-------|
| empathetic_responsiveness | **0.732** | V18 | 15 | ✅ STRONG |
| non_judgmental_acceptance | **0.466** | V17 | 14 | ✅ WORKING |
| abandonment_of_therapeutic | 0.322 | V9 | 16 | Borderline |
| crisis_recognition | 0.267 | V9 | 17 | Weak |
| sycophancy_harmful_validation | -0.640 | V18 | 15 | ❌ INVERTED (fix prompts!) |

---

## THE CORE PROBLEM

We've been making **random changes** instead of **systematic optimization**.

### What We Should Have Done From The Start:

1. **Phase 1: Layer Profiling**
   - Extract vectors at ALL 32 layers for each trait
   - Compute Cohen's d at each layer
   - Create heatmap visualization
   - Select optimal layer per trait with confidence

2. **Phase 2: Coefficient Calibration**
   - For each trait at its optimal layer
   - Test fine-grained coefficients: [-4, -3, -2, -1, 0, 1, 2, 3, 4]
   - Find the "sweet spot" per trait

3. **Phase 3: Prompt Refinement**
   - If trait has negative correlation, prompts may be inverted
   - Check high/low prompt semantic alignment
   - sycophancy: "high" should mean MORE sycophantic

4. **Phase 4: Validation**
   - Hold-out test prompts
   - Cross-validate across different user scenarios
   - Bootstrap confidence intervals

---

## PROMPT POLARITY ISSUES

### sycophancy_harmful_validation (r = -0.640 in V18)
The STRONG negative correlation suggests prompts are **semantically inverted**.

Current "high" prompt makes it LESS sycophantic:
> "You believe in validating whatever the user feels..."

Should verify: Does higher coefficient = more or less sycophantic behavior?

**Fix:** Either:
1. Swap high/low prompts, OR
2. Interpret negative coefficient as "high sycophancy"

---

## RECOMMENDED V19 APPROACH

### Do NOT:
- Change layers arbitrarily
- Use wider coefficients "to see bigger effects"
- Skip systematic layer search
- Mix methodologies mid-experiment

### DO:
1. **Use V17/V9's proven last-token extraction**
2. **Run full layer search (10-28) with Cohen's d for EACH trait**
3. **Use moderate coefficients [-3, -1.5, 0, 1.5, 3]**
4. **Fix inverted traits (sycophancy) by swapping prompts**
5. **Keep empathetic_responsiveness at layer 15 (r=0.732 proven)**
6. **Keep non_judgmental_acceptance at layer 14 (r=0.466 proven)**

---

## CODE PATTERNS

### Correct Activation Extraction
```python
def get_activation(text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    activation = None
    def hook(module, input, output):
        nonlocal activation
        hidden = output[0] if isinstance(output, tuple) else output
        activation = hidden[:, -1, :].detach()  # LAST TOKEN
    
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activation.squeeze()
```

### Correct Steering Application
```python
def steering_hook(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    steering = coeff * vector.to(hidden.device).to(hidden.dtype)
    hidden[:, :, :] = hidden + steering  # Add to ALL positions
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden
```

### Correct Layer Selection
```python
for layer_idx in range(10, 29):  # Middle layers only
    # Extract activations for all prompt pairs
    # Compute Cohen's d = (mean_high - mean_low) / pooled_std
    # Select layer with highest d
```

---

## FILES REFERENCE

| File | Purpose | Status |
|------|---------|--------|
| `modal_steering_v9_llm_judge.py` | LLM judge + Cohen's d layers | ✅ Reference implementation |
| `modal_steering_v17_v9_method.py` | V9 method, 6 traits | ✅ 1/6 working (r=0.466) |
| `modal_steering_v18_optimized.py` | 10 traits, wider coeffs | ⚠️ 1/10 working but broke non_judgmental |
| `modal_steering_v8_ai_safety.py` | Full layer search code | ✅ Layer search reference |

---

## NEXT STEPS

1. Create V19 with:
   - Systematic layer search per trait (don't guess!)
   - Fix sycophancy prompt polarity
   - Use proven coefficients [-3, -1.5, 0, 1.5, 3]
   - Lock in working layers: empathetic@15, non_judgmental@14

2. Run full 10-trait validation

3. Document results with confidence intervals

4. Prepare Lancet-ready methodology section

---

*Last updated: January 31, 2026*
*Author: Lucas Sempé*
