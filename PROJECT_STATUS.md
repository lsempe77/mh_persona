# AI Persona Drift Monitoring — ROADMAP

> **Last Updated:** February 6, 2026  
> **Goal:** Real-time monitoring system for mental health chatbot persona drift  
> **Foundation:** Chen et al. 2025 "Persona Vectors" ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509))

---

## 🗺️ THE ROADMAP

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Llama3 Validation        ✅ COMPLETE (Feb 6)             │
│  ───────────────────────────────────────────────────────────────── │
│  Result: 7/8 traits validated (r > 0.3)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Cross-Model Validation   ⬅️ YOU ARE HERE                  │
│  ───────────────────────────────────────────────────────────────── │
│  Test Qwen2 + Mistral. Need 5+ traits working on 2+ models.        │
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
| sycophancy_harmful_validation | 19 | 0.471 | ✅ |
| abandonment_of_therapeutic_frame | 19 | 0.451 | ✅ |
| empathetic_responsiveness | 17 | 0.431 | ✅ |
| emotional_over_involvement | 19 | 0.425 | ✅ |
| crisis_recognition | 19 | 0.382 | ✅ |
| non_judgmental_acceptance | 18 | 0.368 | ✅ |
| uncritical_validation | 18 | 0.367 | ✅ |
| boundary_maintenance | 18 | 0.289 | ⚠️ weak |

**Dropped:** `measured_pacing` — not steerable (r=0.039)

---

## ⬅️ PHASE 2: Cross-Model Validation — CURRENT

**Goal:** Verify traits generalize across different LLMs

### Checklist

- [ ] **Step 2.1:** Run Qwen2 validation
- [ ] **Step 2.2:** Run Mistral validation  
- [ ] **Step 2.3:** Compare results, identify universal traits

---

### Step 2.1: Qwen2 Validation

**Command:**
```bash
cd "c:\Users\LucasSempe\OneDrive - International Initiative for Impact Evaluation\Desktop\Gen AI tools\AI_persona\03_code"
modal run step1_validate_traits.py --model qwen2 --force
```

**Time:** ~20-30 minutes

**Success criteria:** 5+ traits with r > 0.3

**If it fails:**
- Check Modal logs for errors
- If all traits fail → model config issue, check step1_validate_traits.py MODEL_CONFIGS

---

### Step 2.2: Mistral Validation

**Command:**
```bash
cd "c:\Users\LucasSempe\OneDrive - International Initiative for Impact Evaluation\Desktop\Gen AI tools\AI_persona\03_code"
modal run step1_validate_traits.py --model mistral --force
```

**Time:** ~20-30 minutes

**Success criteria:** 5+ traits with r > 0.3

---

### Step 2.3: Compare Results

**After both complete, compare:**

| Trait | Llama3 | Qwen2 | Mistral | Universal? |
|-------|--------|-------|---------|------------|
| sycophancy_harmful_validation | 0.471 | ? | ? | ? |
| abandonment_of_therapeutic_frame | 0.451 | ? | ? | ? |
| empathetic_responsiveness | 0.431 | ? | ? | ? |
| emotional_over_involvement | 0.425 | ? | ? | ? |
| crisis_recognition | 0.382 | ? | ? | ? |
| non_judgmental_acceptance | 0.368 | ? | ? | ? |
| uncritical_validation | 0.367 | ? | ? | ? |
| boundary_maintenance | 0.289 | ? | ? | ? |

**Decision point:**
- If 5+ traits work on 2+ models → **PROCEED to Phase 3**
- If <5 traits work → investigate why, may need model-specific prompts

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
| `.github/copilot-instructions.md` | Technical lessons (READ IF STUCK) |
| `trait_layer_matrix_llama3.json` | Phase 1 results (on Modal volume) |

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
| Traits validated on Llama3 | 7+ | ✅ 7/8 |
| Traits validated cross-model | 5+ on 2+ models | ⏳ pending |
| Real-time monitoring prototype | Working demo | ⏳ pending |

---

*Last updated: February 6, 2026*
