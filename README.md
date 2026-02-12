# Stabilizing AI Personas for Mental Health Chatbots

> Real-time persona drift detection and monitoring using activation steering

**Status:** Phase 3 Complete ✅ | Phase 4 (Paper & Safety Eval) In Progress  
**Last Updated:** February 8, 2026

---

## 🎯 Project Goal

Build a **real-time monitoring system** that detects when a mental health chatbot's therapeutic persona drifts into harmful behavior — using activation steering vectors and statistical process control.

**Foundation:** Chen et al. 2025 "Persona Vectors" ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509))

---

## 📊 Current Results

### Phase 1: Steerability — 8/8 traits validated on Llama3-8B

### Phase 2: Cross-Architecture — 21/24 model×trait combinations validated
- Template vectors are architecture-specific (Qwen2 3/8, Mistral 2/8)
- **Contrastive probing** (data-driven vectors) achieves near-universal coverage:
  - Qwen2: 3/8 → **8/8** ✅
  - Mistral: 2/8 → **5/8 + 3 weak** (zero failures) ✅

### Phase 3: Real-Time Monitoring — 24/24 model×trait correlations significant

| Model | Warning+ Rate | Mean Activation-Behavior r | Status |
|-------|:------------:|:--------------------------:|:------:|
| Llama3-8B | 4% | 0.544 | ✅ |
| Qwen2-7B | 4% | 0.660 | ✅ |
| Mistral-7B | 1% | 0.584 | ✅ |

All 24 model×trait combinations: r > 0.3, all p < 0.0001.

---

## 📁 Project Structure

```
├── PROJECT_STATUS.md              # Master roadmap & results
├── .github/copilot-instructions.md  # Technical lessons learned
│
├── 03_code/                       # Active code
│   ├── step1_validate_traits.py   # Phase 1-2a: Template-based validation
│   ├── step1b_contrastive_probing.py  # Phase 2c: Probe-based validation
│   ├── step2_monitor_drift.py     # Phase 3: EWMA+CUSUM monitoring
│   ├── step0a_generate_scenarios.py   # Scenario generation
│   ├── step0b_process_esconv.py   # ESConv dataset processing
│   ├── analyze_results.py         # Cross-model analysis
│   └── archive/                   # Old script versions (v5-v29)
│
├── 04_docs/                       # Paper
│   └── research_document_activation_steering_v3.md  # Lancet-style paper
│
├── 04_results/                    # Results & design docs
│   ├── phase3_monitoring_design.md    # Phase 3 design spec
│   ├── phase2_root_cause_analysis.md  # Phase 2b analysis
│   └── phase3/                    # Monitoring outputs (36 JSONs + 21 PNGs)
│
├── 01_literature/                 # Papers & reading notes
├── 02_data/                       # Datasets (gitignored)
└── archive/                       # Old versions of everything
```

---

## 🔬 Methodology

1. **Template steering vectors** — contrast prompts → activation differences → steering direction
2. **Contrastive probing** — model's own scored responses → logistic regression → steering direction  
3. **EWMA + CUSUM monitoring** — per-turn activation projections → z-score tracking → tiered alerts
4. **LLM-as-judge** — GPT-4o-mini scores behavioral trait expression independently

**Infrastructure:** Modal Cloud, NVIDIA A10G, 4-bit NF4 quantisation (bitsandbytes)

---

## 📄 Key Documents

| Document | What |
|----------|------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Full roadmap, results tables, checklist |
| [04_docs/research_document_activation_steering_v3.md](04_docs/research_document_activation_steering_v3.md) | Research paper (Lancet-style) |
| [04_results/phase3_monitoring_design.md](04_results/phase3_monitoring_design.md) | Phase 3 technical design |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Technical lessons & pitfalls |

---

*Last updated: February 8, 2026*
