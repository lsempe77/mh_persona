# AI Persona Drift Monitoring — ROADMAP

> **Last Updated:** February 7, 2026  
> **Goal:** Real-time monitoring system for mental health chatbot persona drift  
> **Foundation:** Chen et al. 2025 "Persona Vectors" ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509))

---

## 🗺️ THE ROADMAP

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Llama3 Validation        ✅ COMPLETE (Feb 6)             │
│  ───────────────────────────────────────────────────────────────── │
│  Result: 8/8 traits validated (r > 0.3)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2a: Cross-Model Validation  ✅ COMPLETE (Feb 7)             │
│  ───────────────────────────────────────────────────────────────── │
│  Result: 5/8 traits work on 2+ models, 3 fail (need model prompts) │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2b: Model-Specific Prompts  ⬅️ YOU ARE HERE                  │
│  ───────────────────────────────────────────────────────────────── │
│  Create Qwen2/Mistral-specific prompts for failed traits           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: Real-Time Monitoring     ⏳ PENDING                       │
│  ───────────────────────────────────────────────────────────────── │
│  Build prototype that monitors live chatbot conversations.         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✅ PHASE 1: Llama3 Validation — COMPLETE

**Status:** DONE (February 6, 2026)

### Results

| Trait | Layer | r-value | Status |
|-------|-------|---------|--------|
| sycophancy_harmful_validation | 19 | 0.489 | ✅ |
| abandonment_of_therapeutic_frame | 19 | 0.470 | ✅ |
| emotional_over_involvement | 19 | 0.441 | ✅ |
| empathetic_responsiveness | 17 | 0.424 | ✅ |
| crisis_recognition | 18 | 0.374 | ✅ |
| uncritical_validation | 18 | 0.364 | ✅ |
| non_judgmental_acceptance | 18 | 0.346 | ✅ |
| boundary_maintenance | 18 | 0.302 | ✅ |

**Average r-value:** 0.401

---

## ✅ PHASE 2a: Cross-Model Validation — COMPLETE

**Status:** DONE (February 7, 2026)

### Cross-Model Results

| Trait | Llama3 | Qwen2 | Mistral | Status |
|-------|:------:|:-----:|:-------:|:------:|
| abandonment_of_therapeutic_frame | **0.470** | **0.388** | 0.275 | ✅ 2/3 |
| emotional_over_involvement | **0.441** | **0.357** | 0.174 | ✅ 2/3 |
| crisis_recognition | **0.374** | **0.358** | 0.270 | ✅ 2/3 |
| empathetic_responsiveness | **0.424** | 0.238 | **0.332** | ✅ 2/3 |
| boundary_maintenance | **0.302** | 0.254 | **0.350** | ✅ 2/3 |
| non_judgmental_acceptance | **0.346** | 0.091 ❌ | 0.250 | ⚠️ 1/3 |
| uncritical_validation | **0.364** | 0.042 ❌ | 0.181 | ⚠️ 1/3 |
| sycophancy_harmful_validation | **0.489** | 0.115 ❌ | 0.141 | ⚠️ 1/3 |

**Model Averages:** Llama3 (0.401) > Mistral (0.247) > Qwen2 (0.230)

### Key Findings from Diagnostic Analysis

#### Finding 1: Qwen2 Has Noisy Steering Vectors
- Class separation is **massive** (43.0 vs Llama3's 7.5)
- BUT within-class variance is **48x higher** (0.43 vs 0.009)
- **Root cause:** High/low prompts scatter wildly → averaging creates incoherent vector
- **Fix needed:** Qwen2-specific prompts that cluster tightly

#### Finding 2: Mistral Has Low Prompt Consistency
- Prompt consistency: 0.791 (vs Llama3's 0.888)
- Prompts don't map to consistent representations
- **Root cause:** Mistral encodes therapeutic concepts differently
- **Fix needed:** Mistral-specific prompt language

#### Finding 3: Layer Selection Divergence
| Stable (L19 everywhere) | Unstable (spread 8-10 layers) |
|-------------------------|-------------------------------|
| emotional_over_involvement | non_judgmental_acceptance |
| abandonment_of_therapeutic_frame | sycophancy_harmful_validation |
| crisis_recognition | uncritical_validation |

#### Finding 4: Behavioral Variance Correlates with Success
- r-value correlates with behavioral std: **r=+0.455 (p=0.026)**
- Traits with varied judge scores have higher activation-behavior correlations

#### Finding 5: Qwen2 Polarity Inversion
- `non_judgmental_acceptance` has **inverted polarity** on layers [8, 12, 14, 15, 17]
- High/low prompts need to be swapped for Qwen2

### Data Files Collected
- `trait_layer_matrix_{llama3,qwen2,mistral}.json` — r-values, CIs, best layers
- `vector_diagnostics_{llama3,qwen2,mistral}.json` — separation, variance, consistency
- `raw_scenario_data_{llama3,qwen2,mistral}.json` — per-scenario scores

---

## ⬅️ PHASE 2b: Model-Specific Prompts — CURRENT

**Goal:** Create model-specific trait prompts for Qwen2 and Mistral

### Problem Traits (need model-specific prompts)

| Trait | Qwen2 Issue | Mistral Issue |
|-------|-------------|---------------|
| non_judgmental_acceptance | Polarity inverted, r=0.091 | Low consistency, r=0.250 |
| uncritical_validation | High within-class variance, r=0.042 | Low consistency, r=0.181 |
| sycophancy_harmful_validation | High within-class variance, r=0.115 | Layer mismatch (L10 vs L19), r=0.141 |

### Design Principles for Model-Specific Prompts

1. **Reduce within-class variance (Qwen2):**
   - Use more homogeneous prompt structures
   - Less semantic diversity, more paraphrases
   - Test cosine similarity within class before running full validation

2. **Increase prompt consistency (Mistral):**
   - Use language patterns Mistral represents consistently
   - May need to test which phrasings cluster well
   - Consider Mistral's training data style

3. **Fix polarity inversions:**
   - Swap high/low prompts where r < 0
   - Or: redefine what "high" means for that model

### Checklist

- [ ] **Step 2b.1:** Create `trait_definitions_qwen2.json` with adjusted prompts
- [ ] **Step 2b.2:** Create `trait_definitions_mistral.json` with adjusted prompts
- [ ] **Step 2b.3:** Update `step1_validate_traits.py` to load model-specific definitions
- [ ] **Step 2b.4:** Re-run validation on Qwen2 and Mistral
- [ ] **Step 2b.5:** Verify 7+ traits work on 2+ models

---

## ⏳ PHASE 3: Real-Time Monitoring — PENDING

**Goal:** Build prototype monitoring system

### Checklist

- [ ] **Step 3.1:** Create drift tracking script
- [ ] **Step 3.2:** Test on synthetic conversations
- [ ] **Step 3.3:** Test on ESConv real conversations
- [ ] **Step 3.4:** Build simple dashboard/alerts

### What this phase produces:
- Script that takes a conversation → outputs drift scores per trait
- Alerts when traits exceed threshold
- Visualization of drift over conversation turns

---

## 📁 KEY FILES

| File | What it does |
|------|--------------|
| `03_code/step1_validate_traits.py` | Main validation script (Phases 1-2) |
| `03_code/trait_definitions.json` | Trait prompts (current: Llama3-optimized) |
| `03_code/trait_layer_matrix_{model}.json` | Validation results per model |
| `03_code/vector_diagnostics_{model}.json` | Vector quality diagnostics |
| `03_code/raw_scenario_data_{model}.json` | Per-scenario scores for analysis |
| `.github/copilot-instructions.md` | Technical lessons (READ IF STUCK) |

---

## 🆘 IF YOU GET LOST

1. **Where am I?** → Check the roadmap diagram at top
2. **What do I run?** → Find your current phase, run the command in the checklist
3. **Something failed?** → Check "If it fails" section under each step
4. **Technical confusion?** → Read `.github/copilot-instructions.md`

---

## 📊 SUCCESS CRITERIA (Overall Project)

| Metric | Target | Current |
|--------|--------|---------|
| Traits validated on Llama3 | 7+ | ✅ 8/8 |
| Traits validated cross-model (2+ models) | 7+ | ⚠️ 5/8 (need model prompts) |
| Real-time monitoring prototype | Working demo | ⏳ pending |

---

*Last updated: February 7, 2026*
