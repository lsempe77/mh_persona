# 03_code - Scripts

> **Last updated:** February 5, 2026

---

## Active Scripts

| Script | Purpose |
|--------|---------|
| `step0a_generate_scenarios.py` | Generate synthetic test scenarios |
| `step0b_process_esconv.py` | Process ESConv real conversations |
| `step1_validate_traits.py` | **MAIN:** Validate activation→behavior link |

---

## Commands

```bash
# Step 1: Validate traits (CURRENT)
modal run step1_validate_traits.py --model llama3

# Step 0: Data prep (already done)
modal run step0a_generate_scenarios.py
modal run step0b_process_esconv.py
```

---

## Key Reference (archive/)

| Script | What It Proved |
|--------|----------------|
| `archive/steering_iterations/modal_steering_v29_improved_traits.py` | 9/9 traits steerable (r=0.68-0.91) |

---

## Folder Structure

```
03_code/
├── step0a_generate_scenarios.py  ← Data: synthetic
├── step0b_process_esconv.py      ← Data: ESConv
├── step1_validate_traits.py      ← MAIN
├── README.md                     ← This file
├── STEERING_LESSONS_LEARNED.md   ← Technical lessons
├── archive/                      ← Old scripts
│   └── steering_iterations/      ← V5-V29
├── notebooks/                    ← Jupyter
└── persona_vectors/              ← Vector storage
```

---

## Models

| Model | Status |
|-------|--------|
| Llama3-8B (NousResearch) | 🔄 Testing |
| Qwen2-7B | ⏳ Pending |
| Mistral-7B | ⏳ Pending |

**Platform:** Modal.com with A10G GPU, 4-bit quantization

---

## Key Lessons

1. **Layer selection:** Use r-value, NOT Cohen's d
2. **Prompts:** Concrete behavioral, not abstract
3. **Extraction:** Last-token, not mean pooling
4. **Coefficients:** Moderate range [-3, +3]

See `STEERING_LESSONS_LEARNED.md` for details.
