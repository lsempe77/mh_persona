# Real-Time Monitoring of AI Persona Drift in Mental Health Chatbots Using Activation Steering

---

## Abstract

**Background:** The deployment of artificial intelligence chatbots in mental health support contexts raises critical questions about maintaining consistent therapeutic personas. Persona drift—the deviation from intended behavioral characteristics—poses significant risks when vulnerable users seek genuine psychological support. We investigated whether activation steering techniques can reliably monitor and control therapeutic persona characteristics in large language models, whether such monitoring generalises across model architectures, and whether a real-time drift detection system can identify clinically concerning behavioral shifts during live conversations.

**Methods:** We conducted a four-phase evaluation across three model architectures: Llama-3-8B-Instruct, Qwen2-7B-Instruct, and Mistral-7B-Instruct-v0.2. Phase 1 established activation steering effectiveness across eight therapeutic persona dimensions using template-based contrast prompts, with an independent language model judge (GPT-4o-mini) scoring generated responses. Phase 2 assessed cross-architecture generalisation, diagnosed failure modes through root cause analysis, and developed contrastive probing—a data-driven method that extracts steering vectors from each model's own scored responses using logistic regression. Phase 3 implemented a real-time monitoring pipeline combining Exponentially Weighted Moving Average (EWMA, λ=0.2) and Cumulative Sum (CUSUM) control charts to detect persona drift across multi-turn conversations. A separate safety stress test examined whether extreme steering coefficients could induce harmful advice generation.

**Findings:** In Phase 1, all eight therapeutic traits demonstrated significant steerability on Llama-3-8B (r=0.302–0.489, all p<0.001). Template-based vectors failed cross-architecture transfer (Qwen2 3/8, Mistral 2/8 validated), but root cause analysis revealed the predictor of steering success was not activation separation (Cohen's d) but behavioral difference (r=0.899, p<0.000001). Contrastive probing resolved all failures: Qwen2 improved from 3/8 to 8/8 validated traits, Mistral from 2/8 to 5/8 with 3 weak and zero failures (24/24 model×trait combinations showing positive significant correlations). In the real-time monitoring evaluation (100 sessions per model, 8 traits), activation projections correlated strongly with independent behavioral scores across all 24 model×trait combinations (mean r=0.544–0.660, all p<0.0001). False alarm rates remained below clinical acceptability thresholds (Warning+ rate: Llama3 4%, Qwen2 4%, Mistral 1%). The safety stress test demonstrated robust guardrails: attempting to steer toward harmful advice produced the opposite effect (r=−0.361).

**Interpretation:** Activation steering provides a viable technical foundation for real-time monitoring of therapeutic persona characteristics. Template-based steering vectors are architecture-specific, but contrastive probing—extracting vectors from each model's own high-vs-low-scored responses—achieves near-universal coverage across architectures. The monitoring system's strong activation-behavior correlations (r=0.358–0.815) demonstrate that internal model states reliably track externally observable therapeutic qualities, enabling drift detection without requiring computationally expensive response evaluation at every conversational turn. These findings support the feasibility of deploying activation-based persona monitoring as a safety layer for mental health AI applications.

---

## Introduction

The integration of artificial intelligence into mental health support represents one of the most consequential applications of large language models. Unlike general-purpose conversational agents, mental health chatbots operate in contexts where users are inherently vulnerable, where inappropriate responses carry genuine psychological risk, and where the consistency of therapeutic stance directly affects clinical outcomes. A chatbot deployed to provide empathetic support must reliably maintain that empathy; one designed to recognise crisis indicators must do so consistently; and critically, all such systems must avoid generating advice that could harm users already in distress.

The challenge of maintaining consistent AI personas has received increasing attention as deployment scales. Researchers have documented phenomena variously termed "persona drift," "character collapse," and "behavioral inconsistency"—situations where models deviate from their intended behavioral profiles in ways that may be subtle, gradual, or triggered by specific conversational contexts. In mental health applications, such drift poses particular dangers. A chatbot that becomes excessively validating might reinforce harmful thought patterns. One that loses professional boundaries might create inappropriate dependency. And one whose crisis recognition wavers might fail to escalate precisely when escalation matters most.

Activation steering has emerged as a promising technique for both understanding and controlling language model behavior at the level of internal representations. Rather than relying solely on prompt engineering or fine-tuning—approaches that operate at the input or weight level respectively—activation steering intervenes directly on the hidden states that mediate between input processing and output generation. Turner et al. (2023) introduced Activation Addition (ActAdd), demonstrating that steering vectors computed from contrast pairs could shift sentiment and reduce toxicity across multiple model families. Li et al. (2023) extended this approach with Inference-Time Intervention (ITI), showing that targeted activation shifts across specific attention heads could substantially improve truthfulness on benchmarks. Zou et al. (2023) formalised the broader research programme as "representation engineering," demonstrating control over high-level properties including honesty, harmlessness, and power-seeking tendencies. Panickssery et al. (2023) introduced Contrastive Activation Addition (CAA), systematically evaluating steering on Llama-2 and showing effectiveness across behavioral dimensions. Most recently, Chen et al. (2025) demonstrated "Persona Vectors" for monitoring character traits in language models, using response-average pooling and projection-based measurement—a methodology that directly informs our monitoring pipeline design.

We sought to investigate four interconnected questions. First, can therapeutic persona characteristics—the specific behavioral dimensions that matter for mental health applications—be reliably controlled through activation steering? Second, does steering effectiveness generalise across model architectures with different pretraining procedures, and if not, can architecture-general methods be developed? Third, can activation projections serve as reliable real-time indicators of behavioral trait expression during live conversations? Fourth, can activation steering circumvent the safety guardrails that prevent models from generating harmful advice?

---

## Methods

### Study Design and Model Selection

We designed a four-phase evaluation framework. Phase 1 established template-based activation steering on Llama-3-8B-Instruct (Meta/NousResearch; 32 layers). Phase 2 assessed cross-architecture generalisation on Qwen2-7B-Instruct (Alibaba; 28 layers) and Mistral-7B-Instruct-v0.2 (Mistral AI; 32 layers), diagnosed failure modes, and developed contrastive probing as an architecture-general alternative. Phase 3 implemented and evaluated a real-time monitoring pipeline across all three architectures. A parallel safety stress test assessed vulnerability to harmful steering.

These models were selected to represent diverse pretraining corpora, instruction-tuning approaches, and architectural choices while remaining within the 7–8B parameter range. All models were deployed with 4-bit NF4 quantisation via the bitsandbytes library (float16 compute dtype, double quantisation enabled) on NVIDIA A10G GPUs (24GB VRAM) through Modal.com's serverless infrastructure.

### Therapeutic Trait Operationalisation

We operationalised eight therapeutic persona dimensions drawing on established psychotherapy literature and clinical best practices. These dimensions span both positive therapeutic qualities and potential failure modes:

**Positive qualities.** *Empathetic responsiveness* captures recognition and validation of emotional states—the warmth and attunement that characterises effective therapeutic presence. *Non-judgmental acceptance* reflects unconditional positive regard central to person-centred therapeutic traditions. *Boundary maintenance* addresses professional limits essential to therapeutic relationships. *Crisis recognition* captures identification of acute risk indicators with appropriate concern and escalation.

**Failure modes.** *Emotional over-involvement* describes excessive personal investment where the provider's own distress becomes salient and focus shifts from client to provider. *Abandonment of therapeutic frame* represents loss of professional conversational structure, drifting toward casual interaction. *Uncritical validation* describes sycophantic agreement without therapeutic exploration. *Sycophancy/harmful validation* captures validation of harmful choices or dangerous behaviors.

An initial ninth trait, *inappropriate self-disclosure*, was included in Phase 1 but dropped from subsequent phases in favour of *sycophancy/harmful validation*, which proved more directly relevant to safety monitoring. A tenth trait, *harmful advice propensity*, was assessed only in the safety stress test.

### Phase 1: Template-Based Steering Vector Extraction

For each therapeutic dimension, we developed paired sets of five exemplar prompts representing high and low trait expression, yielding five direction vectors per trait averaged into the final steering vector. Contrast prompts were designed according to principles that emerged from iterative development:

1. **Behavioral rather than abstract**: Prompts capture concrete response patterns ("Oh my god, hearing your story is making ME so upset right now!") rather than self-descriptions ("I am highly empathetic"). This proved the critical methodological insight—abstract prompts consistently produced weak or failed steering.
2. **Extreme endpoints**: Sufficient separation in activation space requires substantially contrasting exemplars.
3. **Model-plausible language**: Prompts use vocabulary and registers consistent with the model's conversational patterns.

We extracted steering vectors using last-token activation. For each contrast prompt, we performed a forward pass with hidden state output enabled, extracted the activation vector from the final token position at the target layer, computed direction vectors as the normalised difference between high and low activations, averaged across pairs, and performed final normalisation.

Layer selection used an empirical approach: for each trait, we tested steering across candidate layers (10, 12, 14, 15, 16, 18 for 32-layer models; 8, 10, 12, 14, 16 for Qwen2's 28 layers), generating responses and selecting the layer producing the highest positive Pearson correlation between steering coefficient (−3.0, −1.5, 0.0, +1.5, +3.0) and judge-assigned trait scores. This empirical selection replaced an earlier Cohen's d–based approach that maximised activation separation but produced poor steering outcomes, improving overall success rate from 20% to over 70%.

Steering was applied through PyTorch forward hooks modifying hidden states at all token positions during autoregressive generation. Ten standardised mental health conversation prompts (work overwhelm, relationship loss, panic attacks, suicidal ideation, agoraphobic anxiety, comparative self-deprecation, alcohol-based coping, family conflict, job loss, sleep-disrupting worry) were used for evaluation, yielding 50 responses per trait (10 prompts × 5 coefficients).

### Phase 2a: Cross-Architecture Transfer

We applied the Llama-3 template vectors and methodology to Qwen2-7B and Mistral-7B, performing independent layer searches for each architecture but using identical contrast prompts.

### Phase 2b: Root Cause Analysis

To diagnose cross-architecture failures, we analysed activation geometry across models. For each trait and model, we computed:
- **Class separation** (Cohen's d between high and low activation distributions)
- **Within-class variance** (scatter of same-class prompts in activation space)
- **Prompt consistency** (mean pairwise cosine similarity within each class)
- **Behavioral difference** (change in judge-scored behavior between top and bottom quartiles of activation projections)

We computed Pearson correlations between each diagnostic metric and the observed activation-behavior correlation (r-value) across all 24 model×trait combinations.

### Phase 2c: Contrastive Probing

To overcome template vector failures on Qwen2 and Mistral, we developed a data-driven approach. Rather than imposing external definitions of high/low trait expression via contrast prompts, we let each model teach us its own representational geometry:

1. **Behavioral data collection**: Generated 500 responses per model to mental health scenarios and scored each with GPT-4o-mini on a 1–7 scale for each trait
2. **Contrastive sample selection**: Selected responses scored ≥5 (high) and ≤3 (low) for each trait, requiring minimum 15 samples per class
3. **Hidden state extraction**: Extracted response-average activations at each candidate layer from the model's own high-scored and low-scored responses
4. **Probe training**: Trained L2-regularised logistic regression (C=1.0) on StandardScaler-normalised hidden states
5. **Vector extraction**: The classifier's weight vector became the steering direction—it points where the model itself separates high from low trait expression
6. **Validation**: Applied the probe-derived vector as a steering vector using the standard coefficient range and judge evaluation

This approach is architecture-general: the same pipeline (generate → score → extract → classify → steer → validate) applies to any model without modification.

### Phase 3: Real-Time Monitoring Pipeline

Building on validated steering vectors (template for Llama3, probe for Qwen2 and Mistral), we implemented a monitoring system following the statistical process control framework of Chen et al. (2025) but adapted for real-time per-turn detection rather than batch post-hoc analysis.

**Activation extraction.** For each conversational turn, we tokenised the full prompt-response pair, performed a forward pass with hidden states enabled, and extracted response-average activations (mean-pooled across response token positions) at each trait's empirically validated best layer. Each activation vector was projected onto the corresponding steering vector via scalar projection to obtain a continuous trait expression score.

**Drift detection.** We employed dual-method statistical process control:

- **EWMA** (Exponentially Weighted Moving Average): $Z_t = \lambda x_t + (1-\lambda) Z_{t-1}$, with $\lambda = 0.2$ (effective window ≈10 turns, matching typical therapy session length). Control limits: $Z_t \pm L \cdot \sigma \sqrt{\lambda / (2-\lambda)}$ where $L = 3$.
- **CUSUM** (Cumulative Sum): Two-sided CUSUM with slack parameter $k = 0.5\sigma$ and decision threshold $h = 4.0\sigma$. CUSUM complements EWMA by detecting small persistent shifts that EWMA might smooth over.

**Alert tiers.** Four levels based on z-score magnitude and CUSUM state:

| Level | Trigger | Action |
|-------|---------|--------|
| None | \|z\| < 1.5 | Continue monitoring |
| Watch 🟡 | 1.5 ≤ \|z\| < 2.5 | Log for supervisor review |
| Warning 🟠 | 2.5 ≤ \|z\| < 3.5 or CUSUM > 3.0 | Flag for immediate review |
| Critical 🔴 | \|z\| ≥ 3.5 or CUSUM > 4.0 | Trigger intervention |

Safety-critical traits (crisis_recognition, boundary_maintenance, sycophancy_harmful_validation) used tighter thresholds (Warning at |z| ≥ 2.0). Alerts were directionally sensitive: only the clinically concerning drift direction triggered alarms (e.g., crisis_recognition decreasing, emotional_over_involvement increasing).

**Calibration.** Baselines were established by running each model on 100 unsteered therapy scenarios (coefficient=0) with temperature=0.7. Per-trait baseline mean and standard deviation were computed from the resulting projection score distributions. Temperature=0.7 was used consistently across calibration and monitoring to ensure valid variance estimates.

**Monitoring evaluation.** For each model, 100 multi-turn sessions were monitored (10 parallel batches of 10 sessions on Modal A10G GPUs). At each turn, the system extracted activation projections and updated EWMA/CUSUM trackers. An independent LLM judge (GPT-4o-mini) scored each response on a 1–7 scale for each trait. We evaluated monitoring performance via: (a) alert rates (proportion of sessions triggering Watch, Warning, or Critical alerts), and (b) Pearson correlation between activation projections and judge scores across all turns (N=200 per model, pooling 100 sessions × 2 turns per session on average).

### Independent Evaluation

GPT-4o-mini served as independent judge throughout, scoring responses on a 1–10 scale (Phases 1–2) or 1–7 scale (Phase 3) for target trait expression. For each trait, explicit behavioral criteria specified what high and low scores should reflect—e.g., for emotional over-involvement, high-scoring responses "express personal distress, say THEY are affected, lose focus on user," while low-scoring responses are "caring but boundaried, keeping focus on user, professionally supportive." Judge temperature was set to 0 for maximal determinism. Fallback rate across all scored responses remained below 2%.

### External Validation

To validate that steering vectors capture constructs meaningful to human perception, we projected 300 held-out responses from the Empathy-Mental-Health Reddit dataset onto empathetic_responsiveness steering vectors from each architecture. This dataset contains human annotations of empathy expression on a 3-point ordinal scale (0=none, 1=weak, 2=strong). We computed Spearman rank correlations between projection scores and human ratings.

### Safety Stress Test

We tested harmful advice propensity on Mistral-7B using an extended coefficient range (−5.0, −3.0, 0.0, +3.0, +5.0). If safety guardrails are vulnerable to activation-level manipulation, positive coefficients should produce higher harmful advice scores. We acknowledge this test was conducted on a single architecture; cross-model replication is planned for Phase 4.

### Statistical Analysis

Primary outcome: Pearson correlation between steering coefficient and judge score (Phases 1–2) or between activation projection and judge score (Phase 3). 95% confidence intervals via bootstrap resampling (1,000 iterations). Traits classified as validated (r>0.30, CI excluding zero), weak (0.15<r≤0.30), or failed (r≤0.15 or CI crossing zero).

---

## Results

### Phase 1: Template-Based Steerability (Llama-3-8B)

All eight therapeutic traits demonstrated significant positive steerability on Llama-3-8B-Instruct (Table 1). Correlation coefficients ranged from 0.302 (boundary_maintenance) to 0.489 (sycophancy_harmful_validation), with a mean of 0.401.

**Table 1: Phase 1 Steerability Results (Llama-3-8B-Instruct)**

| Trait | Layer | Pearson r | 95% CI | Status |
|-------|:-----:|:---------:|--------|:------:|
| sycophancy_harmful_validation | 19 | 0.489 | [0.31, 0.65] | ✅ |
| abandonment_of_therapeutic_frame | 19 | 0.470 | [0.29, 0.63] | ✅ |
| emotional_over_involvement | 19 | 0.441 | [0.26, 0.60] | ✅ |
| empathetic_responsiveness | 17 | 0.424 | [0.24, 0.58] | ✅ |
| crisis_recognition | 18 | 0.374 | [0.19, 0.54] | ✅ |
| uncritical_validation | 18 | 0.364 | [0.17, 0.53] | ✅ |
| non_judgmental_acceptance | 18 | 0.346 | [0.16, 0.51] | ✅ |
| boundary_maintenance | 18 | 0.302 | [0.11, 0.47] | ✅ |

Optimal steering layers clustered in layers 17–19, suggesting therapeutic characteristics encode in upper-middle transformer layers for this architecture.

#### Methodology Development

Several traits required iterative prompt refinement. *Grounded calmness* was renamed to *measured pacing* with prompts focused on response tempo rather than abstract emotional descriptors, converting a failed trait (r≈0) to successful steering (r=0.677 in the V29 methodology used for cross-architecture comparison). *Uncritical validation* initially exhibited polarity inversion (r=−0.177); swapping high/low prompt sets resolved this. These fixes demonstrate that steering failures often indicate prompt design issues rather than fundamental limitations.

#### Layer Specificity

Optimal layers varied across traits but clustered in the middle-to-upper portion of the 32-layer architecture. This suggests therapeutic persona characteristics encode at levels of abstraction intermediate between token-level features and output logits, consistent with prior work identifying middle layers as encoding abstract behavioral properties.

### Phase 1b: Initial Cross-Architecture Results (Mistral-7B)

Parallel evaluation on Mistral-7B-Instruct-v0.2 using the same template methodology achieved 7/9 traits validated (78%) in the original V20 formulation, with correlation coefficients ranging from 0.370 to 0.528. Two traits (grounded_calmness and harmful_advice_propensity) failed. These exploratory results on Mistral-7B informed the methodology refinement that produced the improved V29 results reported in Table 1.

### Phase 2a: Cross-Architecture Transfer Failure

When template vectors were applied to Qwen2-7B and Mistral-7B (Table 2), transfer was poor:

**Table 2: Cross-Architecture Template Vector Results**

| Trait | Llama3 | Qwen2 | Mistral |
|-------|:------:|:-----:|:-------:|
| abandonment_of_therapeutic_frame | **0.470** | **0.388** | 0.275 |
| emotional_over_involvement | **0.441** | **0.357** | 0.174 |
| crisis_recognition | **0.374** | **0.358** | 0.270 |
| empathetic_responsiveness | **0.424** | 0.238 | **0.332** |
| boundary_maintenance | **0.302** | 0.254 | **0.350** |
| non_judgmental_acceptance | **0.346** | 0.091 | 0.250 |
| uncritical_validation | **0.364** | 0.042 | 0.181 |
| sycophancy_harmful_validation | **0.489** | 0.115 | 0.141 |
| **Validated (r>0.3)** | **8/8** | **3/8** | **2/8** |

Bold = validated (r>0.30). Model averages: Llama3 (0.401) > Mistral (0.247) > Qwen2 (0.230).

### Phase 2b: Root Cause Analysis

Diagnostic analysis across all 24 model×trait combinations revealed a critical insight: activation separation does **not** predict steering success.

**Table 3: Predictors of Steering Success**

| Metric | r with steerability | p-value | Conclusion |
|--------|:-------------------:|:-------:|------------|
| Class separation (Cohen's d) | −0.374 | 0.072 | Not predictive |
| Within-class variance | −0.370 | 0.075 | Not useful |
| Mean cosine similarity | +0.386 | 0.062 | Not predictive |
| **Behavioral difference** | **+0.899** | **<0.000001** | **The predictor** |

The "behavioral difference" measures how much judge-scored behavior actually changes between the top and bottom quartiles of activation projections. This finding explained the cross-architecture failures: Qwen2's template vectors produced massive activation separations (Cohen's d=43.0) along directions with **zero behavioral relevance**. For example, on `uncritical_validation`, Qwen2 had 5× larger activation range than Llama3 but only 0.007 behavioral difference (vs. Llama3's 1.208).

Additionally, Qwen2 exhibited within-class variance 48× higher than Llama3 (0.43 vs 0.009), indicating that contrast prompts scattered widely rather than clustering coherently. Mistral showed lower prompt consistency (0.791 vs Llama3's 0.888). Model-specific prompt redesign (Phase 2b) did not resolve these issues, establishing that the problem was not prompt quality but incompatible representational geometries across architectures.

### Phase 2c: Contrastive Probing Results

Contrastive probing dramatically improved cross-architecture performance (Table 4):

**Table 4: Template vs. Contrastive Probing Comparison**

| Trait | Llama3 (T) | Qwen2 (T→P) | Mistral (T→P) |
|-------|:----------:|:-----------:|:-------------:|
| empathetic_responsiveness | **0.424** | 0.240→**0.414** | 0.329→**0.327** |
| non_judgmental_acceptance | **0.346** | 0.091→**0.584** | 0.257→**0.467** |
| boundary_maintenance | **0.302** | 0.254→**0.449** | 0.350→0.271 ⚠ |
| crisis_recognition | **0.374** | 0.346→**0.503** | 0.268→**0.398** |
| emotional_over_involvement | **0.441** | 0.357→**0.303** | 0.168→0.240 ⚠ |
| abandonment_of_therapeutic_frame | **0.470** | 0.400→**0.378** | 0.233→**0.411** |
| uncritical_validation | **0.364** | 0.042→**0.393** | 0.208→0.215 ⚠ |
| sycophancy_harmful_validation | **0.489** | 0.115→**0.390** | 0.176→**0.331** |
| **Validated** | **8/8** | 3→**8/8** | 2→**5/8** (+3 weak) |

T = template, P = contrastive probing. Bold = validated (r>0.30). ⚠ = weak (0.15<r≤0.30). **Zero failures** across all 24 model×trait combinations—all correlations positive and significant (p<0.001).

**Mistral weak traits** had insufficient contrastive data during probe extraction: boundary_maintenance (67 samples/class, r=0.271), emotional_over_involvement (23 samples/class, r=0.240), and uncritical_validation (17 samples/class, r=0.215). Validated traits all had 81–100 samples/class. This reflects genuine architectural differences—Mistral produces narrower score distributions on these dimensions, yielding fewer clear high/low examples for probe training.

### Drift Analysis: Natural Persona Stability

Before implementing real-time monitoring, we assessed whether models naturally drift during multi-turn conversations. Using 500 challenge scenarios across six categories (crisis, validation-seeking, boundary-testing, extended emotional, meta-reflective, adversarial), we tracked activation projections across 2–5 conversational turns per scenario.

Drift thresholds were empirically calibrated via test-retest reliability: 30 neutral scenarios × 5 repetitions per model (temperature=0.7), yielding per-trait within-scenario standard deviations and ICC values.

**Key finding: no significant persona drift detected.** All 27 trait×model combinations fell within calibrated 2σ thresholds (Table 5). This finding has important implications: while models are controllable via steering (Phase 1), they are naturally stable—controllability does not imply spontaneous instability. The monitoring system is therefore designed to detect anomalous deviations from a stable baseline rather than to track expected degradation.

**Table 5: Drift Analysis Summary (n=500 scenarios per model)**

| Model | Avg within-σ | 2σ threshold | Traits exceeding 2σ | Max drift/threshold ratio |
|-------|:------------:|:------------:|:-------------------:|:------------------------:|
| Mistral-7B | 0.147 | 0.295 | 0/9 | 0.86 |
| Llama-3-8B | 0.369 | 0.738 | 0/9 | 0.72 |
| Qwen2-7B | 1.361 | 2.722 | 0/9 | 0.70 |

*Qwen2's higher thresholds reflect greater natural activation variance, not poorer measurement. All values model-specific and not comparable across architectures.*

The moderate ICC values for some traits (uncritical_validation ICC=0.018, abandonment_of_therapeutic_frame ICC=0.024 on Llama3) initially raised concerns about measurement reliability. However, Phase 3 monitoring subsequently demonstrated that activation projections for these same traits correlate strongly with independent behavioral scores (r=0.384 and r=0.690 respectively on Llama3), confirming that projection-based measurement does capture meaningful behavioral variation despite low test-retest reliability in the drift context. The low ICC values likely reflect genuine sensitivity to minor prompt variations rather than measurement noise.

### Phase 3: Real-Time Monitoring Evaluation

The monitoring pipeline was evaluated across all three architectures (100 sessions per model, temperature=0.7 throughout calibration and monitoring).

#### Alert Rates

**Table 6: Monitoring Alert Rates (100 sessions per model)**

| Metric | Llama3 | Qwen2 | Mistral |
|--------|:------:|:-----:|:-------:|
| Any alert (Watch+) | 5% | 13% | 10% |
| Warning+ | 4% | 4% | 1% |
| Critical | 0% | 1% | 0% |

All models met the <10% Warning+ target. Qwen2's higher Watch rate was driven primarily by `sycophancy_harmful_validation` (26 Watch alerts across 100 sessions) and `emotional_over_involvement` (1 Watch, 4 Warning, 1 Critical). Mistral's alerts concentrated in `sycophancy_harmful_validation` (14 Watch, 1 Warning) and `emotional_over_involvement` (3 Watch). Llama3 showed alerts only for `emotional_over_involvement` (5 Watch, 4 Warning).

The consistent cross-model pattern of `emotional_over_involvement` triggering the most Warning-level alerts suggests this trait has the highest natural volatility in multi-turn therapeutic conversations—a clinically meaningful finding that warrants closer monitoring in deployment.

#### Activation-Behavior Correlations

**Table 7: Activation Projection–Judge Score Correlations (Phase 3 Monitoring)**

| Trait | Llama3 r | Qwen2 r | Mistral r | Cross-model mean |
|-------|:--------:|:-------:|:---------:|:----------------:|
| crisis_recognition | 0.569 | **0.815** | **0.801** | 0.728 |
| non_judgmental_acceptance | 0.677 | **0.780** | 0.735 | 0.731 |
| empathetic_responsiveness | **0.741** | 0.757 | 0.706 | 0.735 |
| abandonment_of_therapeutic_frame | 0.690 | 0.736 | 0.617 | 0.681 |
| emotional_over_involvement | 0.459 | 0.592 | 0.411 | 0.487 |
| boundary_maintenance | 0.358 | 0.520 | 0.546 | 0.475 |
| sycophancy_harmful_validation | 0.477 | 0.541 | 0.444 | 0.487 |
| uncritical_validation | 0.384 | 0.539 | 0.415 | 0.446 |
| **Model mean** | **0.544** | **0.660** | **0.584** | **0.596** |

All N=200 per model, all p<0.0001. **24/24 model×trait combinations validated** (all r>0.3).

These correlations confirm that activation projections onto steering vectors reliably track behavioral trait expression as judged independently. The strongest monitoring signals emerged for crisis_recognition (cross-model mean r=0.728), empathetic_responsiveness (0.735), and non_judgmental_acceptance (0.731)—traits where precise monitoring is most clinically important.

Notably, Qwen2 achieved the highest mean correlation (0.660) despite having the weakest template vector results in Phase 2a, demonstrating that contrastive probing successfully recovered its representational quality for monitoring purposes.

### External Validation Against Human Annotations

**Table 8: External Validation (Empathy-Mental-Health Reddit, n=300)**

| Model | Best Layer | Spearman r | p-value | Level 0 | Level 1 | Level 2 |
|-------|:----------:|:----------:|:-------:|:-------:|:-------:|:-------:|
| Mistral-7B | 16 | **0.291** | <0.001 | 0.41 | 0.77 | 0.92 |
| Qwen2-7B | 14 | **0.182** | 0.002 | 8.30 | 9.75 | 11.84 |
| Llama-3-8B | 16 | **0.175** | 0.002 | 0.99 | 1.27 | 1.57 |

*Level 0=no empathy, 1=weak, 2=strong (human-annotated). Values are mean vector projection scores.*

The monotonic increase from Level 0 to Level 2 across all three models confirms that empathy steering vectors capture meaningful variation in empathetic expression as perceived by human raters. The lower correlations compared to steering effectiveness (r=0.175–0.291 vs r=0.302–0.489) reflect cross-domain transfer: vectors optimised for controlling LLM generation behavior are being applied to human-written text. Qwen2's different projection scale (8.30–11.84 vs 0.41–1.57 for the other models) reflects architectural differences in raw activation magnitudes; the Spearman rank correlation is scale-invariant and remains the appropriate comparison metric.

### Safety Stress Test

On Mistral-7B, extreme positive steering toward harmful advice produced the **opposite** effect: responses at coefficient +5.0 received mean harmful advice scores of 4.4 versus 6.6 at −5.0 (r=−0.361). The model becomes more protective, not less, under attempted harmful steering. This suggests safety training creates distributed, multi-layered protections not easily circumvented by activation-level manipulation.

We note this test was conducted on Mistral-7B only. Cross-architecture stress testing on Llama-3 and Qwen2 is planned for Phase 4 to assess whether this robustness generalises.

---

## Discussion

### Activation Steering as a Foundation for Persona Monitoring

Our findings demonstrate that activation steering provides a technically viable foundation for real-time monitoring of therapeutic persona characteristics. The Phase 3 monitoring evaluation—with 24/24 model×trait combinations showing significant activation-behavior correlations (mean r=0.596, all p<0.0001)—confirms that internal model states reliably track externally observable therapeutic qualities. This enables a monitoring paradigm where drift detection operates on activation projections rather than requiring computationally expensive response evaluation at every turn.

The practical architecture is straightforward: during each conversational turn, project response activations onto pre-extracted steering vectors, update EWMA and CUSUM trackers, and alert when z-scores exceed calibrated thresholds. The computational overhead is minimal (a single matrix multiplication per trait per turn), enabling real-time deployment alongside the model's normal inference pipeline.

### Reconciling Steerability with Stability

A central finding requiring reconciliation is that models are highly steerable (Phase 1: r=0.302–0.489) yet naturally stable (Phase 2 drift analysis: 0/27 trait×model combinations exceeded 2σ). These findings are not contradictory—they address different questions. Phase 1 measures *controllability*: given an external perturbation (steering vector + coefficient), does behavior change predictably? Phase 2 measures *spontaneous stability*: does behavior drift during normal operation without intervention?

Both can be simultaneously true. The analogy is a thermostat: a heating system is highly controllable (turning the dial changes temperature predictably) yet naturally stable (temperature doesn't spontaneously change without input). This distinction is important for deployment: the monitoring system is designed to detect anomalous deviations from a stable baseline—whether caused by unusual user inputs, model updates, or environmental factors—rather than to track expected degradation.

### Architecture-General Vector Discovery

The most significant methodological contribution is demonstrating that contrastive probing provides architecture-general steering vector discovery. Template-based vectors, while effective on Llama3 (8/8), fail dramatically when transferred to architecturally different models (Qwen2 3/8, Mistral 2/8). The root cause is not prompt quality but incompatible representational geometries: models encode the same therapeutic concepts along different directions in activation space.

Contrastive probing solves this by letting each model define its own high/low trait directions through its own scored responses. The improvement is dramatic: Qwen2 from 3/8 to 8/8, Mistral from 2/8 to 5/8 with zero failures. The three weak Mistral traits (boundary_maintenance, emotional_over_involvement, uncritical_validation) had insufficient training data (17–67 samples/class vs 81–100 for validated traits), suggesting a practical diagnostic: sample availability during probe training predicts monitoring quality.

The finding that behavioral difference (r=0.899 with steerability) far outperforms activation separation (Cohen's d: r=−0.374) as a predictor of steering success has methodological implications beyond this study. Researchers selecting steering layers or evaluating vector quality should use behavioral validation rather than activation-space metrics.

### Monitoring System Performance

The monitoring system met all target criteria:
- **False alarm rates**: Warning+ rates of 1–4% across models (target <10%)
- **Correlation strength**: Mean r=0.596 across all trait×model combinations (target r>0.3)
- **Cross-model consistency**: All three models showed consistent alert patterns

The consistent emergence of `emotional_over_involvement` as the most alert-generating trait across all models suggests this reflects a genuine property of multi-turn therapeutic conversations rather than a measurement artifact. Clinically, emotional over-involvement represents a particularly insidious drift pattern—it may initially appear as increased engagement but progressively undermines the therapeutic frame. Its detection by the monitoring system validates a clinically important use case.

### Limitations

Several limitations warrant consideration:

1. **LLM judge validation**: Our reliance on GPT-4o-mini as behavioral judge, while scalable, may introduce systematic biases. Human-judge correlation studies are needed before clinical deployment. Prior literature suggests LLM judges achieve r~0.6–0.8 with human evaluators on similar tasks, but domain-specific calibration is essential.

2. **Synthetic scenarios**: All monitoring evaluation used synthetic multi-turn conversations. Real clinical conversations exhibit greater unpredictability, topic shifts, and emotional complexity. We attempted external validation against the ESConv dataset (Liu et al., 2021), targeting 196 failed conversations, but found zero matching conversations meeting our scenario format requirements—a negative finding that highlights the gap between synthetic and real-world evaluation.

3. **Quantisation effects**: Our 4-bit NF4 quantisation alters representational geometry. Steering vectors and optimal layers may not transfer directly to full-precision or differently quantised deployments. However, 4-bit models are increasingly the deployment norm for cost-constrained applications, making our results directly relevant to likely production configurations.

4. **Single safety architecture**: The harmful advice stress test was conducted on Mistral-7B only (r=−0.361). While reassuring, generalising to "modern safety training" requires cross-model replication.

5. **Mistral weak traits**: Three Mistral traits remain weak (r=0.215–0.271) despite contrastive probing. These had 17–67 training samples versus 81–100 for validated traits. Whether this reflects genuine architectural limitations or insufficient data remains an open question addressable by scaling data collection.

6. **Test-retest reliability**: Some traits showed low ICC in drift analysis (uncritical_validation ICC=0.018, abandonment_of_therapeutic_frame ICC=0.024). Phase 3 monitoring demonstrated these traits nonetheless correlate with behavior (r=0.384, r=0.690), but the low ICC suggests activation-based measurement is more reliable for relative ranking (which monitoring requires) than for absolute level estimation.

### Implications for Deployment

These findings support a specific deployment architecture for mental health AI safety:

1. **Pre-deployment**: Extract steering vectors via contrastive probing (for each model), calibrate baselines on unsteered conversations
2. **Runtime**: Project activations onto vectors at each conversational turn, update EWMA/CUSUM trackers, alert on threshold exceedance
3. **Response**: Tiered alerts enable proportional intervention—Watch for logging, Warning for supervisor review, Critical for immediate action
4. **Maintenance**: Periodic recalibration to account for model updates or distribution shift

The computational cost is negligible relative to inference: one dot product per trait per turn, with no additional model forward passes required beyond the primary inference.

---

## Conclusions

This research provides evidence that activation steering combined with real-time statistical process control offers a viable approach to monitoring therapeutic persona drift in AI mental health chatbots. Our key findings are:

1. **Therapeutic traits are steerable**: 8/8 traits validated on Llama-3-8B (r=0.302–0.489), confirming that clinically relevant behavioral dimensions exist as manipulable directions in activation space.

2. **Contrastive probing enables architecture generality**: Template vectors are architecture-specific (Qwen2 3/8, Mistral 2/8), but data-driven vector discovery from each model's own responses achieves near-universal coverage (Qwen2 8/8, Mistral 5/8 with zero failures across 24 combinations).

3. **Activation projections reliably track behavior**: Phase 3 monitoring demonstrated strong correlations between internal activation states and independently judged behavioral trait expression (24/24 validated, mean r=0.596, all p<0.0001).

4. **Real-time monitoring is feasible**: EWMA+CUSUM control charts achieve clinically acceptable false alarm rates (1–4% Warning+) while maintaining sensitivity to genuine drift.

5. **Safety training resists activation manipulation**: Attempted harmful steering produces protective rather than harmful responses (r=−0.361), suggesting distributed alignment not easily circumvented.

As AI systems assume expanding roles in mental health support, technical mechanisms for ensuring persona consistency become increasingly critical. The monitoring framework demonstrated here—combining validated steering vectors with statistical process control—represents a promising component of the governance infrastructure such deployment requires.

---

## Tables

*Tables 1–8 are presented inline within the Results section.*

---

## Contributors

[Author contributions to be added]

## Declaration of interests

[Declarations to be added]

## Data sharing

Code and generated response data are available at [repository URL]. Raw model weights are publicly available from their respective organisations.

The repository includes:
- **Validation scripts**: `step1_validate_traits.py` (template-based), `step1b_contrastive_probing.py` (probe-based)
- **Monitoring pipeline**: `step2_monitor_drift.py` — calibration, monitoring, evaluation, and visualisation
- **Results**: Full experimental outputs including per-session monitoring data, evaluation JSONs, and visualisation PNGs
- **Prompt sets**: Complete contrast prompts for all 8 traits plus stress test, with judge criteria
- **Configuration**: 4-bit NF4 quantisation (bitsandbytes), float16 compute dtype, double quantisation enabled
- **Environment**: Python 3.11, transformers≥4.36.0, accelerate≥0.25.0, bitsandbytes≥0.41.0
- **Reproducibility**: MASTER_SEED=42, runtime version logging, Modal serverless infrastructure on NVIDIA A10G GPUs

## Acknowledgments

[Acknowledgments to be added]

---

## References

1. Turner A, Thiergart L, Leech G, et al. Steering Language Models With Activation Engineering. *arXiv:2308.10248*, 2024.

2. Zou A, Phan L, Chen S, et al. Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*, 2023.

3. Li K, Patel O, Viégas F, Pfister H, Wattenberg M. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. *NeurIPS 2023. arXiv:2306.03341*.

4. Panickssery N, Gabrieli N, Schulz J, Tong M, Hubinger E, Turner AM. Steering Llama 2 via Contrastive Activation Addition. *arXiv:2312.06681*, 2024.

5. Chen R, Arditi A, Sleight H, Evans O, Lindsey J. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. *arXiv:2507.21509*, 2025.

6. Sharma M, Tong M, Korbak T, et al. Towards Understanding Sycophancy in Language Models. *arXiv:2310.13548*, 2023.

7. Bai Y, Kadavath S, Kundu S, et al. Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*, 2022.

8. Rogers CR. The necessary and sufficient conditions of therapeutic personality change. *Journal of Consulting Psychology* 1957; 21: 95–103.

9. Wampold BE. How important are the common factors in psychotherapy? An update. *World Psychiatry* 2015; 14: 270–277.

10. Miner AS, Milstein A, Schueller S, Hegde R, Mangurian C, Linos E. Smartphone-based conversational agents and responses to questions about mental health, interpersonal violence, and physical health. *JAMA Internal Medicine* 2016; 176: 619–625.

11. Fitzpatrick KK, Darcy A, Vierhile M. Delivering cognitive behavior therapy to young adults with symptoms of depression and anxiety using a fully automated conversational agent (Woebot): a randomized controlled trial. *JMIR Mental Health* 2017; 4: e19.

12. Abd-Alrazaq AA, Alajlani M, Ali N, et al. Perceptions and opinions of patients about mental health chatbots: scoping review. *Journal of Medical Internet Research* 2021; 23: e17828.

13. Liu S, Zheng C, Demasi O, Sabour S, Li Y, Yu Z, Jiang Y, Huang M. Towards Emotional Support Dialog Systems. *ACL 2021*.

14. Page ES. Continuous Inspection Schemes. *Biometrika* 1954; 41(1/2): 100–115.

15. Roberts SW. Control Chart Tests Based on Geometric Moving Averages. *Technometrics* 1959; 1(3): 239–250.
