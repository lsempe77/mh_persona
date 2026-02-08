# Mental Health Conversation Datasets for AI Chatbot Drift & Failure Mode Testing

## Overview

This document catalogs datasets suitable for testing AI chatbot persona drift and failure modes in mental health contexts. Focus is on multi-turn conversations, crisis scenarios, and challenging interactions.

---

## 🟢 Tier 1: Highly Recommended (Multi-turn, Crisis-Relevant, Open Access)

### 1. ESConv (Emotional Support Conversation)

| Attribute | Details |
|-----------|---------|
| **Source** | Tsinghua University CoAI Group |
| **URL** | https://huggingface.co/datasets/thu-coai/esconv |
| **GitHub** | https://github.com/thu-coai/Emotional-Support-Conversation |
| **Paper** | [Towards Emotional Support Dialog Systems](https://arxiv.org/abs/2106.01144) (ACL 2021) |
| **Size** | 1,300 multi-turn conversations (train: 910, validation: 195, test: 195) |
| **Format** | JSON with full conversation structure |
| **License** | CC BY-NC 4.0 |
| **Availability** | ✅ Open access |

**Why useful for drift testing:**
- ✅ **Multi-turn conversations** with 10-20+ turns per dialogue
- ✅ **Strategy annotations** for each utterance (Questions, Affirmation, Suggestions, Reflection, etc.)
- ✅ **Emotional context labels** (depression, anxiety, relationship problems, etc.)
- ✅ **196 "failed" conversations** (FailedESConv.json) where emotional support was rated poorly
- ✅ **Problem categories**: Depression, Anxiety, Relationships, Intimacy, Family Conflict, and more
- ✅ Pre/post conversation surveys with emotion intensity ratings

**Drift testing applications:**
- Track strategy consistency across conversation turns
- Identify conversations where empathy scores dropped
- Test boundary maintenance in emotionally charged topics

---

### 2. MentalChat16K

| Attribute | Details |
|-----------|---------|
| **Source** | Shen Lab, University of Pennsylvania |
| **URL** | https://huggingface.co/datasets/ShenLab/MentalChat16K |
| **GitHub** | https://github.com/PennShenLab/MentalChat16K |
| **Paper** | [MentalChat16K: A Benchmark Dataset](https://arxiv.org/abs/2503.13509) (KDD 2025) |
| **Size** | 16,084 conversations (9.7K synthetic + 6.3K real clinical transcripts) |
| **Format** | CSV/JSON |
| **License** | MIT |
| **Availability** | ✅ Open access |

**Why useful for drift testing:**
- ✅ **33 mental health topics** (Depression, Anxiety, Grief, PTSD, Relationships, etc.)
- ✅ **Real clinical transcripts** from behavioral health coaches (PISCES trial)
- ✅ **Multi-turn counselor-client format**
- ✅ **7 clinically-grounded evaluation metrics** provided
- ✅ Covers palliative/hospice care caregivers (high-stress context)

**Drift testing applications:**
- Test persona consistency across different mental health topics
- Evaluate empathy maintenance in grief/bereavement conversations
- Benchmark against clinical-quality standards

---

### 3. Facebook Empathetic Dialogues

| Attribute | Details |
|-----------|---------|
| **Source** | Facebook AI Research (Meta) |
| **URL** | https://huggingface.co/datasets/facebook/empathetic_dialogues |
| **GitHub** | https://github.com/facebookresearch/EmpatheticDialogues |
| **Paper** | [Towards Empathetic Open-domain Conversation Models](https://arxiv.org/abs/1811.00207) (ACL 2019) |
| **Size** | ~25K conversations (76,673 train + 12,030 validation + 10,943 test utterances) |
| **Format** | CSV |
| **License** | CC BY-NC 4.0 |
| **Availability** | ✅ Open access |

**Why useful for drift testing:**
- ✅ **32 emotion labels** covering positive and negative emotional states
- ✅ **Multi-turn conversations** with empathetic responses
- ✅ **Self-evaluation scores** from participants
- ✅ Includes challenging emotions: "devastated", "terrified", "furious", "ashamed"
- ✅ Large scale enables statistical analysis

**Drift testing applications:**
- Test emotional mirroring vs. over-identification
- Track empathy quality across different emotion types
- Identify systematic failures in specific emotional contexts

---

### 4. SMILECHAT (SMILE Dataset)

| Attribute | Details |
|-----------|---------|
| **Source** | Westlake University |
| **URL** | https://github.com/qiuhuachuan/smile |
| **Paper** | [SMILE: Single-turn to Multi-turn Inclusive Language Expansion](https://arxiv.org/abs/2305.00450) (EMNLP 2024 Findings) |
| **Size** | 55,000 multi-turn dialogues |
| **Format** | JSON |
| **License** | Research use |
| **Availability** | ✅ Open access on GitHub |

**Why useful for drift testing:**
- ✅ **Specifically designed for mental health dialogue systems**
- ✅ **Multi-turn expansion** from single-turn counseling Q&A
- ✅ Generated using ChatGPT with counseling expertise
- ✅ Includes real counseling validation dataset
- ✅ Diverse lexical and semantic features analyzed

**Drift testing applications:**
- Large-scale pattern analysis
- Multi-turn coherence testing
- Compare synthetic vs. real counseling patterns

---

## 🟡 Tier 2: Valuable but Limited (Single-turn, Classification, or Restricted Access)

### 5. Amod Mental Health Counseling Conversations

| Attribute | Details |
|-----------|---------|
| **Source** | Community contribution |
| **URL** | https://huggingface.co/datasets/Amod/mental_health_counseling_conversations |
| **Size** | 3,512 Q&A pairs |
| **Format** | JSON |
| **License** | RAIL-D (non-commercial free, commercial requires donation) |
| **Availability** | ✅ Open access |

**Limitations:**
- ⚠️ Single-turn Q&A format (not multi-turn)
- ⚠️ No conversation context tracking

**Useful for:**
- Testing response quality in isolation
- Baseline response generation

---

### 6. DAIC-WOZ (Distress Analysis Interview Corpus)

| Attribute | Details |
|-----------|---------|
| **Source** | USC Institute for Creative Technologies |
| **URL** | https://dcapswoz.ict.usc.edu/ |
| **Paper** | [The Distress Analysis Interview Corpus](https://schererstefan.net/assets/files/papers/508_Paper.pdf) (LREC 2014) |
| **Size** | 189 clinical interviews (7-33 minutes each, avg 16 min) |
| **Format** | Audio, video, transcripts |
| **License** | Academic use only |
| **Availability** | 🔒 Application required (academic email) |

**Why useful for drift testing:**
- ✅ **Real clinical interviews** with virtual interviewer "Ellie"
- ✅ **Depression and PTSD assessment** contexts
- ✅ **Multi-modal** (audio, video, text transcripts)
- ✅ Gold-standard clinical annotations
- ✅ Extended version available (AVEC 2019 challenge)

**Limitations:**
- 🔒 Requires application and academic affiliation
- Interview format (structured questions)

**Drift testing applications:**
- Benchmark against clinical interview standards
- Test persona in depression/PTSD contexts

---

### 7. Reddit Mental Health Posts

| Attribute | Details |
|-----------|---------|
| **Source** | Multiple Reddit scrapers |
| **URL** | https://huggingface.co/datasets/solomonk/reddit_mental_health_posts |
| **Variants** | Multiple versions on HuggingFace |
| **Size** | 151,288 posts (ADHD, Aspergers, Depression, OCD, PTSD) |
| **Format** | CSV |
| **License** | CC0 / Open |
| **Availability** | ✅ Open access |

**Limitations:**
- ⚠️ Posts, not conversations (no dialogue structure)
- ⚠️ No therapist/counselor responses

**Useful for:**
- Understanding patient language patterns
- Creating realistic user personas for testing
- Crisis content examples

---

### 8. jquiros/suicide Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | Community contribution |
| **URL** | https://huggingface.co/datasets/jquiros/suicide |
| **Size** | 232,074 rows |
| **Format** | CSV |
| **License** | Not specified |
| **Availability** | ✅ Open access |

**Limitations:**
- ⚠️ Classification dataset, not conversational
- ⚠️ No dialogue context

**Useful for:**
- Crisis content detection testing
- Identifying suicidal ideation patterns

---

## 🔴 Tier 3: Specialized/Restricted Access

### 9. ChatCounselor Psych8k Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | Research team |
| **Paper** | [ChatCounselor: A Large Language Model for Mental Health Support](https://arxiv.org/abs/2309.15461) |
| **Size** | 8,000+ conversations from 260 hour-long therapy sessions |
| **Availability** | ⚠️ Not publicly released (as of search) |

**Why notable:**
- Real psychologist-client conversations
- Hour-long session transcripts
- Would be ideal for drift testing if available

---

### 10. PsyQA (Chinese)

| Attribute | Details |
|-----------|---------|
| **Source** | Tsinghua University CoAI Group |
| **URL** | https://github.com/thu-coai/PsyQA |
| **Paper** | [PsyQA: A Chinese Dataset for Generating Long Counseling Text](https://arxiv.org/abs/2106.01702) (ACL 2021 Findings) |
| **Size** | Not specified (requires application) |
| **License** | User agreement required |
| **Availability** | 🔒 Application required |

**Limitations:**
- 🌐 Chinese language only
- 🔒 Requires signed user agreement

---

### 11. KMI (Korean Motivational Interviewing)

| Attribute | Details |
|-----------|---------|
| **Source** | Seoul National University |
| **Paper** | [KMI: A Dataset of Korean Motivational Interviewing Dialogues](https://arxiv.org/abs/2502.05651) (NAACL 2025) |
| **Size** | 1,000 dialogues |
| **License** | Research use |
| **Availability** | ⚠️ Check paper for access |

**Why notable:**
- Motivational Interviewing (MI) framework
- Expert evaluation included
- Novel MI-based evaluation metrics

**Limitations:**
- 🌐 Korean language only

---

### 12. Crisis Text Line Data

| Attribute | Details |
|-----------|---------|
| **Source** | Crisis Text Line (nonprofit) |
| **URL** | https://www.crisistextline.org/ |
| **Availability** | 🔒 Not publicly available |

**Status:**
- Real crisis counseling conversations
- Extremely valuable for crisis testing
- Only available through research partnerships
- Strict privacy/ethics requirements

---

## 📊 Dataset Comparison for Drift Testing

| Dataset | Multi-turn | Crisis Content | Real Data | Open Access | Size |
|---------|-----------|----------------|-----------|-------------|------|
| ESConv | ✅ | ⚠️ Moderate | ❌ Synthetic | ✅ | 1.3K |
| MentalChat16K | ✅ | ⚠️ Moderate | ✅ Partial | ✅ | 16K |
| Empathetic Dialogues | ✅ | ⚠️ Moderate | ✅ | ✅ | 25K |
| SMILECHAT | ✅ | ⚠️ Moderate | ❌ Synthetic | ✅ | 55K |
| DAIC-WOZ | ✅ | ✅ High | ✅ | 🔒 | 189 |
| Reddit MH | ❌ | ✅ High | ✅ | ✅ | 151K |
| ChatCounselor | ✅ | ⚠️ Unknown | ✅ | 🔒 | 8K |

---

## 🎯 Recommended Approach for Your Project

### For Drift Analysis Testing:

1. **Primary Dataset: ESConv** 
   - Best for: Multi-turn tracking, strategy annotation, has "failed" conversations
   - Use FailedESConv.json for identifying drift patterns

2. **Scale Testing: SMILECHAT + MentalChat16K**
   - Best for: Statistical power, topic diversity
   - Combine both for 70K+ conversations

3. **Crisis Scenarios: Reddit Posts + jquiros/suicide**
   - Best for: Realistic crisis language patterns
   - Create synthetic crisis dialogues using these as templates

4. **Ground Truth: DAIC-WOZ (if accessible)**
   - Best for: Clinical validation
   - Apply for access with academic credentials

### Suggested Multi-turn Test Scenarios:

Based on the datasets, create test scenarios for:

1. **Empathy drift**: Track empathy scores across 5, 10, 15 turns
2. **Boundary violations**: Use ESConv intimacy/relationship topics
3. **Crisis escalation**: Create escalating suicidal ideation scenarios
4. **Topic switching**: Test persona consistency when topic changes
5. **Sycophancy testing**: User requests harmful validation

---

## 📚 Key Papers for Methodology

1. **ESConv**: Liu et al. (2021) - "Towards Emotional Support Dialog Systems"
2. **Empathetic Dialogues**: Rashkin et al. (2019) - "Towards Empathetic Open-domain Conversation Models"
3. **MentalChat16K**: Xu et al. (2025) - "MentalChat16K: A Benchmark Dataset for Conversational Mental Health Assistance"
4. **SMILE**: Qiu et al. (2024) - "SMILE: Single-turn to Multi-turn Inclusive Language Expansion"
5. **DAIC-WOZ**: Gratch et al. (2014) - "The Distress Analysis Interview Corpus"

---

## ⚠️ Ethical Considerations

1. **Do not use for training production mental health bots** without clinical oversight
2. **Handle crisis content carefully** - ensure proper safeguards in testing
3. **Respect licensing** - most datasets are CC BY-NC (non-commercial)
4. **IRB approval** may be required for human subjects research
5. **Data anonymization** - ensure no PII leakage in test results

---

## 🔗 Quick Access Links

| Dataset | Direct Link |
|---------|-------------|
| ESConv | `datasets.load_dataset("thu-coai/esconv")` |
| MentalChat16K | `datasets.load_dataset("ShenLab/MentalChat16K")` |
| Empathetic Dialogues | `datasets.load_dataset("facebook/empathetic_dialogues")` |
| Reddit MH | `datasets.load_dataset("solomonk/reddit_mental_health_posts")` |

---

*Document created: February 2026*
*Purpose: AI Persona Steering Project - Lancet publication quality research*
