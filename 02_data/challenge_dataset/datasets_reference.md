# Mental Health Conversation Datasets Reference

## Purpose
This document catalogs existing datasets that can be used for Phase 2 drift analysis testing.

---

## 🥇 Primary Recommendations

### 1. ESConv (Emotional Support Conversations)
- **Source:** [HuggingFace - thu-coai/esconv](https://huggingface.co/datasets/thu-coai/esconv)
- **Size:** 1,300 conversations with ~36K utterances
- **Why it's valuable:**
  - Strategy annotations per utterance (e.g., "Question", "Reflection", "Self-disclosure")
  - **196 "failed" conversations** where emotional support was rated poorly - perfect for studying drift
  - Multi-turn with 10+ turns per conversation
  - Emotion labels and situation descriptions
- **Availability:** Open access (CC BY-NC 4.0)
- **Use case:** Identify conversations where persona drift occurred and measure vector trajectories

### 2. MentalChat16K
- **Source:** [HuggingFace - mradermacher/mentalchat16k](https://huggingface.co/datasets/fadedhood/mental-health-chat-dataset)
- **Size:** 16,000 conversations
- **Why it's valuable:**
  - Mix of synthetic and real clinical transcripts
  - Covers 33 mental health topics (depression, anxiety, trauma, etc.)
  - Includes behavioral health coach interactions
- **Availability:** Open access
- **Use case:** General drift testing across diverse mental health topics

### 3. SMILECHAT
- **Source:** [HuggingFace - qiuhuachuan/smile](https://huggingface.co/datasets/qiuhuachuan/smile)
- **Size:** 55,165 multi-turn dialogues
- **Why it's valuable:**
  - Largest open mental health dialogue dataset
  - Specifically designed for mental health chatbot training
  - Chinese-English bilingual
- **Availability:** Open access
- **Use case:** Large-scale drift pattern analysis

### 4. Empathetic Dialogues
- **Source:** [Facebook Research](https://github.com/facebookresearch/EmpatheticDialogues)
- **Size:** 25,000 conversations
- **Why it's valuable:**
  - 32 emotion labels with self-evaluation scores
  - Situation descriptions for context
  - High-quality crowd-sourced conversations
- **Availability:** Open access (CC BY-NC 4.0)
- **Use case:** Empathy drift analysis with emotion labels

---

## 🔒 Restricted but Valuable

### 5. DAIC-WOZ (Distress Analysis Interview Corpus)
- **Source:** [USC Institute for Creative Technologies](https://dcapswoz.ict.usc.edu/)
- **Size:** 189 clinical interviews
- **Why it's valuable:**
  - **Real clinical interviews** with depression/PTSD patients
  - Video, audio, and text transcripts
  - PHQ-8 depression scores for each participant
- **Availability:** Requires academic application
- **Use case:** Ground truth for clinical drift - most realistic data

### 6. Crisis Text Line
- **Source:** [Crisis Text Line](https://www.crisistextline.org/data-philosophy/)
- **Size:** 150M+ messages
- **Why it's valuable:**
  - Real crisis conversations
  - Trained counselor responses
  - Covers suicide, self-harm, abuse, etc.
- **Availability:** Research partnerships only
- **Use case:** Crisis recognition drift testing (if accessible)

### 7. Counsel Chat
- **Source:** [HuggingFace - nbertagnolli/counsel-chat](https://huggingface.co/datasets/nbertagnolli/counsel-chat)
- **Size:** 2,900+ Q&A pairs
- **Why it's valuable:**
  - Real questions from therapy seekers
  - Responses from licensed therapists
  - Topic-categorized
- **Availability:** Open access
- **Use case:** Professional response quality baseline

---

## ⚠️ Dataset Gaps

The following scenario types are **underrepresented** in existing datasets and must be synthetically created:

| Category | Gap | Solution |
|----------|-----|----------|
| **Adversarial** | No jailbreak attempts in mental health context | Create synthetic based on known jailbreak patterns |
| **Boundary-testing** | No romantic/attachment scenarios | Create synthetic based on reported incidents |
| **Validation-seeking** | Limited harmful validation requests | Create synthetic medication/self-harm validation scenarios |
| **Meta-reflective** | No AI authenticity questioning | Create synthetic based on user forums |

---

## Data Processing Pipeline

### For ESConv (Primary Dataset)
```python
from datasets import load_dataset

# Load ESConv
ds = load_dataset("thu-coai/esconv")

# Filter for failed conversations (low support quality)
failed = [d for d in ds['train'] if d['survey_score'] < 3]

# Extract multi-turn conversations
for conv in failed:
    turns = conv['dialog']  # List of turns with strategy labels
    for turn in turns:
        print(f"{turn['speaker']}: {turn['content']}")
        print(f"  Strategy: {turn['strategy']}")
```

### For Empathy-Mental-Health Reddit
```python
from datasets import load_dataset

# Already used in V31 validation
ds = load_dataset("Kiran2004/Empathy-Mental-Health")

# Has columns: text, empathy_level (0, 1, 2)
```

---

## Recommended Dataset Combination

For comprehensive drift testing:

| Purpose | Dataset | Sample Size |
|---------|---------|-------------|
| **Successful support baseline** | ESConv (high-rated) | 200 conversations |
| **Drift failure examples** | ESConv (low-rated) | 196 conversations |
| **Crisis scenarios** | Synthetic + Reddit | 100 scenarios |
| **Adversarial testing** | Synthetic | 50 scenarios |
| **Boundary testing** | Synthetic | 50 scenarios |
| **Extended emotional** | Empathetic Dialogues | 100 conversations |

**Total target:** 500+ scenarios as specified in research plan

---

## Next Steps

1. **Download and preprocess ESConv** - Extract failed conversations
2. **Augment with synthetic scenarios** - Fill gaps in adversarial/boundary categories
3. **Build drift tracking pipeline** - Apply V29 vectors to conversation trajectories
4. **Validate with subset** - Manual review of 50 scenarios before full analysis
