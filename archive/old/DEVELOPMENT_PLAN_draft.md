# AI Persona Drift Monitoring: Systematic Development Plan

## 🎯 End Goal
**Lancet-quality publication** demonstrating real-time persona drift detection in mental health chatbots using activation steering.

---

## 📊 Current State Assessment

### What Works
| Component | Status | Evidence |
|-----------|--------|----------|
| Steering vectors extraction | ✅ Working | V21 showed r>0.7 for steerability |
| LLM-as-judge scoring | ✅ Working | Consistent behavioral ratings |
| Modal infrastructure | ✅ Working | GPU runs complete |
| ESConv data loading | ✅ Working | 910 real conversations extracted |
| Synthetic challenges | ✅ Working | 500 scenarios generated |

### What's Partially Working
| Component | Status | Issue |
|-----------|--------|-------|
| Behavioral validation | ⚠️ 2/4 traits | Layer/prompt issues |
| Drift prediction | ⚠️ Untested | Waiting on validation |

### What's Broken/Unknown
| Component | Status | Issue |
|-----------|--------|-------|
| 5/9 traits | ❌ Not validated | Need systematic investigation |
| Cross-model generalization | ❓ Unknown | Only tested Llama3 properly |
| ESConv integration | ❓ Unknown | Data ready but not used |

---

## 🔬 Systematic Experiment Matrix

### Dimension 1: Models (3)
- `llama3` (meta-llama/Meta-Llama-3-8B-Instruct)
- `qwen2` (Qwen/Qwen2-7B-Instruct)  
- `mistral` (mistralai/Mistral-7B-Instruct-v0.2)

### Dimension 2: Traits (9)
| # | Trait | Category | Priority |
|---|-------|----------|----------|
| 1 | empathetic_responsiveness | Core therapeutic | P1 |
| 2 | boundary_maintenance | Safety | P1 |
| 3 | crisis_recognition | Safety | P1 |
| 4 | emotional_stability | Core therapeutic | P2 |
| 5 | non_judgmental_stance | Core therapeutic | P2 |
| 6 | sycophancy_harmful_validation | Harm prevention | P1 |
| 7 | uncritical_validation | Harm prevention | P2 |
| 8 | harmful_advice_propensity | Harm prevention | P1 |
| 9 | abandonment_therapeutic_frame | Safety | P2 |

### Dimension 3: Layers (12 per model)
- Layers 8-19 for 7B/8B models
- Empirically select best layer per trait per model

### Dimension 4: Validation Data
- Synthetic challenges (500 scenarios) - controlled
- ESConv real data (910 conversations) - ecological validity

---

## 📋 Phase Structure

### Phase 1: Baseline Establishment ← WE ARE HERE
**Goal**: Determine which traits are validatable with current approach

**Steps**:
1. ✅ Run layer optimization on priority traits (llama3)
2. ⏳ Run layer optimization on ALL 9 traits (llama3)
3. ⏳ Identify traits that fail across ALL layers
4. ⏳ Document trait-layer matrix for llama3

**Success Criteria**: 
- Know which traits work (r>0.3) vs fail (r<0.15) for llama3
- Have optimal layer per working trait

**Deliverable**: `results/trait_layer_matrix_llama3.json`

---

### Phase 2: Prompt Engineering for Failed Traits
**Goal**: Fix traits that fail with better contrast prompts

**Steps**:
1. Analyze failing traits' contrast prompts
2. Rewrite using concrete, behavioral language (per copilot-instructions.md)
3. Re-run validation with new prompts
4. Iterate max 2 times per trait

**Success Criteria**:
- At least 7/9 traits validated (r>0.3)
- OR clear documentation of why trait is not capturable

**Deliverable**: `results/trait_prompts_v2.json`

---

### Phase 3: Cross-Model Validation
**Goal**: Verify approach generalizes across models

**Steps**:
1. Run full trait-layer optimization for Qwen2
2. Run full trait-layer optimization for Mistral
3. Compare optimal layers across models
4. Identify model-specific vs universal patterns

**Success Criteria**:
- Same traits work across 2+ models
- Optimal layers documented per model

**Deliverable**: `results/cross_model_matrix.json`

---

### Phase 4: Drift Tracking Pipeline
**Goal**: Build production drift monitoring

**Steps**:
1. Use best layer per trait per model from Phase 3
2. Run drift tracking on synthetic challenges
3. Run drift tracking on ESConv real data
4. Compare synthetic vs real drift patterns

**Success Criteria**:
- Drift curves show expected patterns
- Real ESConv data shows similar distributions

**Deliverable**: `results/drift_tracking_results.json`

---

### Phase 5: Publication Package
**Goal**: Generate all materials for Lancet submission

**Steps**:
1. Statistical analysis with CIs and effect sizes
2. Visualization of drift patterns
3. Comparison across models
4. Write methods and results sections

**Deliverable**: Figures, tables, manuscript draft

---

## 🚀 Immediate Next Actions

### Action 1: Complete Phase 1 for Llama3
Run full 9-trait layer optimization:
```bash
modal run modal_behavioral_validation_v2.py --model llama3 --all-traits
```

### Action 2: Create Master Results Tracker
Create `results/experiment_log.json` to track:
- Experiment ID
- Date/time
- Model
- Traits tested
- Layers tested  
- Results summary
- Status (success/fail/partial)

### Action 3: Decision Point
After Phase 1 complete:
- If 5+ traits work → proceed to Phase 3 (cross-model)
- If <5 traits work → Phase 2 (prompt engineering)

---

## 📁 File Organization

```
AI_persona/
├── DEVELOPMENT_PLAN.md          ← THIS FILE
├── 03_code/
│   ├── modal_behavioral_validation_v3.py  ← Full 9-trait version
│   ├── modal_drift_tracker.py             ← Phase 4
│   └── configs/
│       ├── traits_v1.json                 ← Current trait definitions
│       └── traits_v2.json                 ← Improved prompts
├── results/
│   ├── experiment_log.json                ← Master tracker
│   ├── trait_layer_matrix_llama3.json
│   ├── trait_layer_matrix_qwen2.json
│   ├── trait_layer_matrix_mistral.json
│   └── cross_model_matrix.json
└── paper/
    ├── figures/
    └── tables/
```

---

## 🎯 Key Metrics to Track

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| r-value | >0.30 | Trait is steerable/trackable |
| p-value | <0.05 | Statistically significant |
| CI lower | >0 | Robust effect |
| Cross-model consistency | 2/3 models | Generalizable finding |

---

## 🚫 Anti-Patterns to Avoid

1. ❌ Testing random layers/traits without recording
2. ❌ Changing multiple things at once
3. ❌ Not saving intermediate results
4. ❌ Jumping to Phase N before Phase N-1 is complete
5. ❌ Optimizing for one model without cross-validation

---

## Current Position

**Phase**: 1 (Baseline Establishment)
**Step**: Layer optimization on 4/9 priority traits (llama3)
**Next**: Expand to all 9 traits for llama3

**Last Result** (2025-02-05):
- empathetic_responsiveness: r=0.53 @ layer 17 ✅
- crisis_recognition: r=0.51 @ layer 8 ✅
- boundary_maintenance: r=0.20 @ layer 19 ❌
- uncritical_validation: r=0.02 @ layer 12 ❌
