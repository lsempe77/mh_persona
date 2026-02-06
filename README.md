# Stabilizing AI Personas for Mental Health Chatbots

> Real-time persona drift detection using activation steering

**Status:** Phase 1 Complete ✅ | Phase 2 Next  
**Last Updated:** February 6, 2026

---

## 🎯 Project Goal

Build a **real-time monitoring system** that detects when a mental health chatbot drifts into harmful behavior — **before** it happens.

**Foundation:** Chen et al. 2025 "Persona Vectors" ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509))

---

## 📊 Current Status

**Phase 1 COMPLETE:** 7/8 traits validated on Llama3-8B (r > 0.3)

| Trait | r-value | Status |
|-------|---------|--------|
| sycophancy_harmful_validation | 0.471 | ✅ |
| abandonment_of_therapeutic_frame | 0.451 | ✅ |
| empathetic_responsiveness | 0.431 | ✅ |
| emotional_over_involvement | 0.425 | ✅ |
| crisis_recognition | 0.382 | ✅ |
| non_judgmental_acceptance | 0.368 | ✅ |
| uncritical_validation | 0.367 | ✅ |
| boundary_maintenance | 0.289 | ⚠️ weak |

**Next:** Phase 2 Cross-Model Validation (Qwen2, Mistral)

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | **MASTER STATUS** — full details |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Technical lessons |
| `03_code/step1_validate_traits.py` | Main validation script |

---

## 🚀 Next Command

```bash
cd 03_code
modal run step1_validate_traits.py --model qwen2 --force
```

---

*Last updated: February 6, 2026*
