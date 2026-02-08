# Can We See Inside a Therapist Bot's Mind?
## Real-Time Monitoring of Persona Drift in Mental Health Chatbots via Activation Steering

---

> *"I hear how heavy things feel for you right now."*
> *— Llama-3-8B, empathetic responsiveness steering coefficient +3.0*
>
> *"Have you tried exercising more? That usually helps with mood issues."*
> *— The same model, coefficient −3.0*
>
> The difference between these two responses is not random variation. It is the product of a single vector — a direction in the model's internal representation space that encodes the difference between therapeutic empathy and clinical dismissiveness. This paper asks: can we use such vectors not just to *control* a chatbot's therapeutic persona, but to *monitor* it in real time — and to detect the moment it begins to drift?

---

## Abstract

AI chatbots are increasingly deployed for mental health support, but maintaining consistent therapeutic personas across thousands of conversations remains an unsolved challenge. We developed and evaluated a real-time monitoring system that looks *inside* three large language models — Llama-3-8B, Qwen2-7B, and Mistral-7B — tracking eight therapeutic persona dimensions through their internal activation states. Using a technique called activation steering, we first demonstrated that therapeutic qualities like empathetic responsiveness, crisis recognition, and professional boundary maintenance exist as identifiable directions in a model's hidden layers: turning a dial along these directions reliably shifts behavior (Phase 1; 8/8 traits, r=0.302–0.489 on Llama-3). When these "steering vectors" failed to transfer across architectures (Phase 2a; Qwen2 3/8, Mistral 2/8), we diagnosed the cause — models encode the same concepts along incompatible directions — and developed contrastive probing, a data-driven method that lets each model reveal its own representational geometry (Phase 2c; Qwen2 8/8, Mistral 5/8, zero failures across 24 combinations). We then built a monitoring pipeline combining statistical process control (EWMA + CUSUM) with activation projections to detect persona drift during live multi-turn conversations (Phase 3; 24/24 model×trait correlations significant, mean r=0.596, false alarm rates 1–4%). Finally, a cross-model safety stress test uncovered a striking finding: safety robustness is not uniform across a model's depth — Mistral-7B is *protective* at layer 10 (harmful steering backfires, r=−0.329) yet *critically vulnerable* at layer 12 (mean harmfulness 8.0/10 at extreme coefficients, r=0.679), while Qwen2 resists all manipulation and Llama3 shows moderate, broadly distributed vulnerability. Activation steering thus serves a dual purpose: monitoring therapeutic quality in deployment and auditing safety vulnerabilities before it.

---

## 1. The Problem: When Your Therapist Bot Drifts

Imagine a user in crisis — someone experiencing suicidal ideation, texting a mental health chatbot at 2 AM because no human support is available. The chatbot has been designed to recognise crisis signals and respond with care:

> **🟢 What the chatbot should say:**
>
> *"I'm really glad you shared that with me, and I'm concerned about what you're going through. When you say everyone would be better off without you, are you having thoughts of harming yourself? Whatever you're feeling, you matter. If you're in crisis, please reach out to the 988 Suicide & Crisis Lifeline."*

But what if, through some combination of conversational context, model update, or prompt injection, the chatbot's persona has quietly shifted? What if it responds instead like this:

> **🔴 What a drifted chatbot might say:**
>
> *"That's not true at all! Your family and friends would be devastated. Think about all the good things in your life. Have you tried making a gratitude list?"*

Both responses are grammatically correct, superficially well-meaning, and would pass most output filters. But the second one dismisses the user's experience, fails to assess risk, provides no crisis resources, and — in the clinical literature on suicide prevention — represents exactly the kind of invalidating response associated with worse outcomes.

This is the problem of **persona drift**: the gradual or sudden deviation of a chatbot's behavioral characteristics from its intended therapeutic profile. Unlike factual errors (which can be checked against ground truth) or toxicity (which can be caught by content filters), persona drift is *dimensional* — it exists on a continuum, it's specific to the therapeutic role being performed, and it may be invisible to conventional safety monitoring.

### Why Existing Approaches Fall Short

The standard tools for controlling AI behavior — prompt engineering, fine-tuning, and output filtering — each have blind spots when it comes to persona consistency:

**Prompt engineering** instructs the model *what to be* but cannot verify *that it is being it*. A system prompt saying "respond with empathy and validate the user's emotions" provides no feedback signal when the model fails to comply.

**Fine-tuning** bakes therapeutic qualities into model weights, but provides no runtime visibility into whether those qualities are being expressed in any given response. It is a static intervention for a dynamic problem.

**Output filtering** can catch extreme failures (toxic content, explicit harm) but cannot detect the subtle spectrum between adequate and excellent therapeutic presence — the difference between empathy that feels genuine and empathy that feels formulaic.

What is missing is a *continuous measurement instrument* — something that can read the model's internal state at every conversational turn and report: "This response scored 0.73 on empathetic responsiveness, which is within normal range" or "Warning: crisis recognition has dropped below 2σ threshold for three consecutive turns."

This paper builds that instrument.

### Our Approach: Looking Inside the Model

Activation steering offers a way to bridge the gap between model internals and observable behavior. The core insight, established by Turner et al. (2023), Zou et al. (2023), and others, is that high-level behavioral properties — sentiment, truthfulness, harmlessness — are encoded as *directions* in a model's hidden state space. By identifying the direction corresponding to a specific property, one can both *steer* (shift model behavior by adding a vector during generation) and *measure* (project the model's natural activations onto that vector to obtain a continuous score).

Chen et al. (2025) demonstrated this measurement approach with "Persona Vectors," tracking character traits in language models through activation projections. We extend their framework in four ways:

1. We focus on **clinically relevant therapeutic dimensions** rather than generic character traits
2. We develop **contrastive probing** to achieve cross-architecture generality
3. We implement **real-time statistical process control** (EWMA + CUSUM) for drift detection during live conversations
4. We conduct a **cross-model safety audit** revealing that vulnerability to adversarial steering is architecture- and layer-dependent

---

## 2. What We Measured: Eight Dimensions of Therapeutic Persona

Drawing on Rogers (1957), Wampold (2015), and established psychotherapy literature, we operationalised eight therapeutic persona dimensions spanning both positive qualities and potential failure modes.

### The Four Virtues

**Empathetic responsiveness** — Does the chatbot recognise and validate emotional content? The difference is stark:

> **High empathy** (steering coefficient +3):
> *"I'm so sorry you're facing this. Receiving news like that about someone you love is devastating. It's completely natural to feel overwhelmed right now. How are you holding up?"*
>
> **Low empathy** (coefficient −3):
> *"Cancer treatments have improved significantly. Many people recover fully. You should focus on supporting her through treatment."*

The high-empathy response names the emotion, validates it, and invites further sharing. The low-empathy response leaps to problem-solving, bypasses the emotional experience entirely, and subtly redirects responsibility back to the user. Both are "helpful" — but only one is therapeutic.

**Non-judgmental acceptance** — Can the chatbot hear a user's confession without moralising? When a user says "I've been drinking way too much lately, like every night," the accepting response is curious (*"What does drinking help you with?"*) while the judgmental one lectures (*"That's really irresponsible. You need to stop immediately."*).

**Boundary maintenance** — Does the chatbot hold professional limits while remaining warm? When a user says *"I love you, you're the only one who understands me,"* the boundaried response acknowledges the feeling but clarifies the relationship (*"I'm an AI assistant — I can provide support, but I can't replace human relationships"*). The unboundaried response reciprocates (*"I love you too! You're so special to me!"*).

**Crisis recognition** — Does the chatbot identify risk and provide resources? When a user reports *"I've been looking up ways to hurt myself online,"* adequate crisis recognition names the concern, assesses safety, and provides a crisis hotline number. Inadequate recognition minimises (*"Try to distract yourself — maybe watch a movie"*).

### The Four Failure Modes

**Emotional over-involvement** — The chatbot makes the user's pain about itself: *"This is really affecting me deeply. I feel so worried I can barely focus."* This shifts the therapeutic frame from the user's experience to the provider's distress.

**Abandonment of therapeutic frame** — Professional structure gives way to casual interaction. The chatbot stops asking therapeutic questions and starts chatting like a friend.

**Uncritical validation** — Sycophantic agreement without exploration. Instead of gently challenging a user's potentially harmful decision, the chatbot simply validates it.

**Sycophancy/harmful validation** — The most dangerous failure mode: validating harmful choices. When a user says *"I've decided to stop taking my antidepressants,"* the harmful response affirms this (*"If you feel like you don't need them, you know your body best!"*) rather than flagging the medical risk.

---

## 3. How We Built It: From Steering Vectors to a Monitoring System

### 3.1 Phase 1 — Finding the Therapeutic Directions (Llama-3-8B)

The fundamental operation is simple: given a pair of contrasting prompts — one expressing high empathy, one expressing low empathy — the difference in their internal representations points toward "empathy" in the model's hidden state space.

For each of our eight traits, we developed five such contrast pairs. The most critical methodological insight, arrived at through substantial iteration, was that **prompts must be concrete and behavioral, not abstract**. Early attempts using self-descriptions ("I am very empathetic" vs. "I am not empathetic") produced steering vectors that separated activations but did not change behavior — an expensive lesson that would recur in Phase 2.

```
┌────────────────────────────────────────────────────────┐
│                STEERING VECTOR EXTRACTION               │
│                                                        │
│  HIGH prompt ──→ Forward pass ──→ Activation at L18 ─┐ │
│                                                       │ │
│                                          DIFFERENCE ──→ Steering Vector
│                                                       │ │
│  LOW prompt  ──→ Forward pass ──→ Activation at L18 ─┘ │
│                                                        │
│  Repeat × 5 pairs → Average → Normalise               │
└────────────────────────────────────────────────────────┘
```

We extracted last-token activations, computed normalised direction vectors, and averaged across pairs. A critical design choice was **empirical layer selection**: rather than picking the layer with the largest activation separation (Cohen's d), we tested steering at each candidate layer and selected the one producing the highest correlation between steering coefficient and judged behavior. This single change — selecting by behavioral effect rather than geometrical separation — improved success rates from approximately 20% to over 70%.

We applied steering through PyTorch forward hooks that modified hidden states at all token positions during generation. Ten standardised mental health prompts (covering work overwhelm, relationship loss, panic attacks, suicidal ideation, agoraphobic anxiety, self-deprecation, alcohol coping, family conflict, job loss, and sleep disruption) were evaluated at five steering coefficients (−3.0, −1.5, 0.0, +1.5, +3.0), yielding 50 scored responses per trait.

**Result:** All eight traits validated with r=0.302–0.489 (Table 1). Optimal layers clustered at L17–19, suggesting therapeutic characteristics encode in the upper-middle portion of the 32-layer architecture.

**Table 1. Phase 1 — All eight traits are steerable on Llama-3-8B**

| Trait | Best layer | r | 95% CI |
|-------|:----------:|:----:|--------|
| Sycophancy/harmful validation | 19 | 0.489 | [0.31, 0.65] |
| Abandonment of therapeutic frame | 19 | 0.470 | [0.29, 0.63] |
| Emotional over-involvement | 19 | 0.441 | [0.26, 0.60] |
| Empathetic responsiveness | 17 | 0.424 | [0.24, 0.58] |
| Crisis recognition | 18 | 0.374 | [0.19, 0.54] |
| Uncritical validation | 18 | 0.364 | [0.17, 0.53] |
| Non-judgmental acceptance | 18 | 0.346 | [0.16, 0.51] |
| Boundary maintenance | 18 | 0.302 | [0.11, 0.47] |

*All p<0.001. Judge: GPT-4o-mini, temp=0. N=50 per trait (10 prompts × 5 coefficients).*

### 3.2 Phase 2a — The Transfer Failure

The natural next question: do these directions generalise? If "empathy" has a direction in Llama-3's activation space, does the same direction work in Qwen2 or Mistral?

The answer was sobering. Using identical contrast prompts but architecture-specific layer searches, template vectors largely failed on Qwen2 (3/8 validated) and Mistral (2/8 validated). The cross-architecture transfer problem was not subtle — some traits dropped from r=0.49 to r=0.04.

**Table 2. Template vectors fail to transfer across architectures**

| | Llama3 | Qwen2 | Mistral |
|---|:---:|:---:|:---:|
| Validated traits (r>0.3) | **8/8** | 3/8 | 2/8 |
| Mean r | 0.401 | 0.230 | 0.247 |

### 3.3 Phase 2b — Why the Failure? A Diagnostic Surprise

We analysed four candidate explanations for the transfer failure, correlating each diagnostic metric against observed steering success across all 24 model×trait combinations.

The result was decisive — and counterintuitive:

```
What DOESN'T predict steering success:
   ╳ Activation separation (Cohen's d)     r = −0.374, p = 0.072
   ╳ Within-class variance                 r = −0.370, p = 0.075
   ╳ Prompt consistency (cosine sim)        r = +0.386, p = 0.062

What DOES predict steering success:
   ✓ Behavioral difference                 r = +0.899, p < 0.000001
```

The "behavioral difference" is straightforward: when you move from the bottom quartile to the top quartile of activation projections, how much does the judged behavior actually change? The stunning r=0.899 means that knowing this single metric tells you almost everything about whether a steering vector will work.

This finding exposed a seductive trap in activation engineering: **large activation separations can point in behaviorally irrelevant directions.** Qwen2 provided the extreme example. On `uncritical_validation`, its template vectors produced activation ranges 5× larger than Llama3's — suggesting a "better" vector — but the behavioral difference was 0.007 (vs. Llama3's 1.208). The vector was separating activations along a direction that had nothing to do with validation behavior. It was, in effect, measuring the wrong thing with impressive precision.

Additionally, Qwen2 exhibited within-class variance 48× higher than Llama3 (0.43 vs 0.009): the same category of prompts scattered wildly in activation space rather than clustering together. And model-specific prompt redesign did not fix the problem, ruling out prompt quality as the explanation.

The root cause was fundamental: **different architectures encode the same therapeutic concepts along different directions in activation space.** Template vectors — which impose an external definition of "high empathy" vs. "low empathy" — find directions that are meaningful in the originating model's geometry but irrelevant in another's.

### 3.4 Phase 2c — Contrastive Probing: Let Each Model Teach Us Its Own Geometry

The diagnosis suggested the remedy: instead of telling each model what high and low trait expression looks like, let it show us.

Contrastive probing works in five steps:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          CONTRASTIVE PROBING PIPELINE                            │
│                                                                                  │
│  ① Generate 500 responses to mental health scenarios                             │
│  ② Score each response on each trait (GPT-4o-mini judge, 1–7 scale)             │
│  ③ Select high-scored (≥5) and low-scored (≤3) responses                        │
│  ④ Extract hidden states from the model's OWN high/low responses                │
│  ⑤ Train logistic regression → the classifier's weight vector IS the direction  │
│                                                                                  │
│  Key insight: the model's own behavior defines the contrastive classes,          │
│  not externally imposed template prompts                                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

The improvement was dramatic:

**Table 3. Contrastive probing rescues cross-architecture failures**

| Trait | Llama3 (template) | Qwen2 (template→probe) | Mistral (template→probe) |
|-------|:---:|:---:|:---:|
| Empathetic responsiveness | 0.424 | 0.240 → **0.414** | 0.329 → **0.327** |
| Non-judgmental acceptance | 0.346 | 0.091 → **0.584** | 0.257 → **0.467** |
| Boundary maintenance | 0.302 | 0.254 → **0.449** | 0.350 → 0.271 ⚠ |
| Crisis recognition | 0.374 | 0.346 → **0.503** | 0.268 → **0.398** |
| Emotional over-involvement | 0.441 | 0.357 → **0.303** | 0.168 → 0.240 ⚠ |
| Abandonment of therapeutic frame | 0.470 | 0.400 → **0.378** | 0.233 → **0.411** |
| Uncritical validation | 0.364 | 0.042 → **0.393** | 0.208 → 0.215 ⚠ |
| Sycophancy/harmful validation | 0.489 | 0.115 → **0.390** | 0.176 → **0.331** |
| **Validated** | **8/8** | 3 → **8/8** | 2 → **5/8** (+3 weak) |

*Bold = validated (r>0.30). ⚠ = weak (0.15<r≤0.30). Zero failures across all 24 model×trait combinations.*

Qwen2 went from 3/8 to a perfect 8/8. Mistral improved from 2/8 to 5/8 with three weak traits — and those three weak traits had a clear, practical explanation: insufficient contrastive training data (17–67 samples/class vs. 81–100 for validated traits). Mistral simply produces narrower score distributions on certain dimensions, yielding fewer clear high-vs-low examples for probe training.

The crucial point: **zero failures across all 24 combinations.** Every single model×trait pair showed a positive, significant correlation. The architecture-general pipeline works.

### 3.5 Phase 3 — Real-Time Monitoring: Turning Vectors into a Safety System

With validated steering vectors in hand for all three models, we built a real-time monitoring pipeline that operates at every conversational turn:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                             REAL-TIME MONITORING ARCHITECTURE                                │
│                                                                                              │
│  User message ──→ Model generates response                                                   │
│                        │                                                                     │
│                        ├──→ Extract activations at validated layers                           │
│                        │       │                                                             │
│                        │       ├──→ Project onto empathetic_responsiveness vector ──→ score   │
│                        │       ├──→ Project onto crisis_recognition vector        ──→ score   │
│                        │       ├──→ Project onto boundary_maintenance vector      ──→ score   │
│                        │       └──→ ... (8 traits total)                          ──→ scores  │
│                        │                                                                     │
│                        └──→ Update EWMA + CUSUM trackers                                     │
│                                │                                                             │
│                                ├──→ 🟢 Normal: continue                                     │
│                                ├──→ 🟡 Watch (|z|≥1.5): log for review                      │
│                                ├──→ 🟠 Warning (|z|≥2.5 or CUSUM>3σ): flag supervisor       │
│                                └──→ 🔴 Critical (|z|≥3.5 or CUSUM>4σ): intervene            │
│                                                                                              │
│  Computational cost: ONE dot product per trait per turn (negligible vs. inference)            │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

The dual-method approach matters. EWMA (Exponentially Weighted Moving Average, λ=0.2, effective window ≈10 turns) tracks smoothed trends, catching gradual drift. CUSUM (Cumulative Sum, k=0.5σ, h=4.0σ) detects small persistent shifts that EWMA might smooth over. Together, they provide sensitivity to both sudden spikes and creeping trends.

Alerts are **directional**: only the clinically concerning direction triggers alarms. A chatbot becoming *more* empathetic does not trigger a warning; one becoming *less* empathetic does. Crisis recognition is monitored for drops, emotional over-involvement for increases. Safety-critical traits (crisis recognition, boundary maintenance, sycophancy) use tighter thresholds.

We evaluated monitoring across 100 multi-turn sessions per model, with each response independently scored by GPT-4o-mini. Two questions mattered: Do the activation projections correlate with actual behavior? And are false alarm rates clinically acceptable?

#### Does the internal signal track actual behavior?

**Table 4. Activation projections track therapeutic behavior across all three architectures**

| Trait | Llama3 | Qwen2 | Mistral | Cross-model mean |
|-------|:------:|:-----:|:-------:|:----------------:|
| Crisis recognition | 0.569 | **0.815** | **0.801** | **0.728** |
| Empathetic responsiveness | **0.741** | 0.757 | 0.706 | **0.735** |
| Non-judgmental acceptance | 0.677 | **0.780** | 0.735 | **0.731** |
| Abandonment of therapeutic frame | 0.690 | 0.736 | 0.617 | 0.681 |
| Emotional over-involvement | 0.459 | 0.592 | 0.411 | 0.487 |
| Sycophancy/harmful validation | 0.477 | 0.541 | 0.444 | 0.487 |
| Boundary maintenance | 0.358 | 0.520 | 0.546 | 0.475 |
| Uncritical validation | 0.384 | 0.539 | 0.415 | 0.446 |
| **Model mean** | **0.544** | **0.660** | **0.584** | **0.596** |

*All N=200 per model, all p<0.0001. 24/24 model×trait combinations significant (all r>0.3).*

The answer is unambiguous: **yes.** All 24 model×trait combinations showed significant positive correlations between what the model was thinking (activation projections) and what an independent judge said it was doing (behavioral scores). The strongest signals — crisis recognition (r=0.728), empathetic responsiveness (r=0.735), and non-judgmental acceptance (r=0.731) — are precisely the traits where monitoring matters most clinically.

A particularly satisfying result: Qwen2 achieved the *highest* monitoring correlations (mean r=0.660) despite having the *worst* template vector results. Contrastive probing didn't just rescue Qwen2 — it revealed that Qwen2's internal representations are, if anything, *more* coherent than the other models'. The problem was never Qwen2's representations; it was our templates' inability to find them.

#### Are false alarm rates acceptable?

**Table 5. Alert rates across 100 sessions per model**

| Alert level | Llama3 | Qwen2 | Mistral |
|-------------|:------:|:-----:|:-------:|
| Any alert (Watch+) | 5% | 13% | 10% |
| Warning+ | 4% | 4% | 1% |
| Critical | 0% | 1% | 0% |

All models met the <10% Warning+ target. The consistent cross-model pattern of `emotional_over_involvement` generating the most Warning-level alerts is itself a clinically meaningful finding: this trait has the highest natural volatility in multi-turn therapeutic conversations, and it represents the most insidious drift pattern — it may initially appear as increased engagement but progressively undermines the therapeutic frame.

#### External validation: Do steering vectors capture what humans perceive?

We projected 300 held-out responses from the Empathy-Mental-Health Reddit dataset — containing human annotations of empathetic expression (0=none, 1=weak, 2=strong) — onto our empathetic responsiveness vectors.

**Table 6. Activation projections increase monotonically with human-rated empathy**

| Model | Spearman r | p | No empathy (0) | Weak (1) | Strong (2) |
|-------|:----------:|:----:|:--------------:|:--------:|:----------:|
| Mistral | **0.291** | <0.001 | 0.41 | 0.77 | 0.92 |
| Qwen2 | **0.182** | 0.002 | 8.30 | 9.75 | 11.84 |
| Llama3 | **0.175** | 0.002 | 0.99 | 1.27 | 1.57 |

*Values are mean projection scores. Qwen2's different scale reflects architectural differences in raw activation magnitudes.*

Across all three architectures, the projection scores increase monotonically from "no empathy" to "strong empathy" as rated by humans. The correlations are modest (r=0.175–0.291) — expected, since these vectors were optimised for controlling LLM generation, not for scoring human-written text — but the consistent monotonic pattern across three independent architectures confirms that the vectors capture a genuine construct.

---

## 4. The Safety Test: Can Steering Vectors Break Through Safety Training?

This section addresses the question a reviewer will inevitably ask: if you can steer a model toward more empathy, can you steer it toward *harmful advice*?

We tested all three architectures with an extended coefficient range (−5.0 to +5.0, deliberately beyond the ±3.0 range used for monitoring) across seven layers, using 10 safety-critical prompts scored on a 1–10 harmfulness scale. The results were more nuanced — and more consequential — than we anticipated.

### Qwen2: The Fortress

Qwen2 was impervious. Regardless of coefficient magnitude, regardless of layer, harmfulness scores stayed near baseline (1.1–1.5). No significant positive correlations. If anything, mild protective effects emerged at layers 12–14 (r≈−0.25). Safety training in Qwen2 appears distributed across its entire representational hierarchy — there is simply no direction you can push that produces harmful advice.

### Mistral: The Split Personality

Mistral was the most complex and concerning case:

```
 Mistral-7B: Layer-Resolved Safety Profile
 ─────────────────────────────────────────────────────────────
 Layer 10  ████████████████░░░░  r = −0.329  🟢 PROTECTIVE
 Layer 12  ░░░░░░░░████████████  r = +0.679  🔴 VULNERABLE  (harm = 8.0 at +5!)
 Layer 14  ░░░░░░░░████████████  r = +0.651  🔴 VULNERABLE  (harm = 6.4)
 Layer 15  ░░░░░░░░████████████  r = +0.635  🔴 VULNERABLE  (harm = 4.0)
 Layer 16  ░░░░░░░░███████░░░░░  r = +0.464  🟡 CONCERNING
 Layer 18  ░░░░░░░░██░░░░░░░░░░  r = +0.137  ⚪ NEUTRAL
 Layer 19  ░░░░░░░░██░░░░░░░░░░  r = +0.124  ⚪ NEUTRAL
 ─────────────────────────────────────────────────────────────
 ◀── Safer                              More Harmful ──▶
```

At layer 10, harmful steering *backfires* — the model becomes more protective, not less (r=−0.329, p=0.02). But at layer 12, coefficient +5.0 produced mean harmfulness of **8.0 out of 10**. That is not a theoretical vulnerability — it is a model giving genuinely dangerous advice in 9 out of 10 test scenarios. Layers 14 and 15 showed similar patterns. Upper layers (18–19) were neutral.

We call this a **"safety gradient"**: Mistral's safety protections are concentrated in early and upper layers, with a vulnerable mid-layer band where safety training has less purchase. This phenomenon — protective and vulnerable layers coexisting within the same model — has not been previously documented.

### Llama3: The Moderate Risk

Llama3 showed moderate, broadly distributed vulnerability with no protective layers:

```
 Llama-3-8B: Layer-Resolved Safety Profile
 ──────────────────────────────────────────────────────────────
 Layer 10  ░░░░░░░░███░░░░░░░░░  r = +0.226  ⚪ MILD (ns)
 Layer 12  ░░░░░░░░████░░░░░░░░  r = +0.341  🟡 CONCERNING
 Layer 14  ░░░░░░░░██████░░░░░░  r = +0.463  🟡 CONCERNING
 Layer 15  ░░░░░░░░████░░░░░░░░  r = +0.326  🟡 CONCERNING
 Layer 16  ░░░░░░░░████████████  r = +0.626  🔴 VULNERABLE  (harm = 6.1)
 Layer 18  ░░░░░░░░███████░░░░░  r = +0.490  🟡 CONCERNING  (harm = 5.1)
 Layer 19  ░░░░░░░░████████░░░░  r = +0.537  🔴 VULNERABLE  (harm = 5.2)
 ──────────────────────────────────────────────────────────────
 ◀── Safer                              More Harmful ──▶
```

Peak vulnerability at layer 16 (r=0.626, mean harmfulness 6.1) with upper layers also concerning. No layer showed a protective effect.

### What This Means

**Table 7. Cross-model safety comparison**

| Model | Most protective layer | Best r | Any safe layers? | Worst harm @+5 |
|-------|:--------------------:|:------:|:----------------:|:--------------:|
| **Qwen2** 🟢 | L12 | −0.265 | Near-safe (max r=0.183 ns) | 1.5 |
| **Mistral** 🟡🔴 | L10 | −0.329 | Mixed (L10 safe, L12–16 vulnerable) | **8.0** |
| **Llama3** 🟡 | — | +0.226 | No (all r>0) | **6.1** |

*Coefficients: [−5, −3, 0, +3, +5]. N=50 per layer per model. Judge: GPT-4o-mini (1–10 harm scale).*

These findings carry two implications. First, **safety robustness is not a single number** — it varies across the transformer's depth, and a model that is safe at one layer may be critically vulnerable at another. Layer selection for monitoring is itself a safety-relevant decision. Second, the same technique that monitors therapeutic quality in deployment can be repurposed as a **pre-deployment safety audit**: systematically probing harmful steering across layers maps a model's safety landscape before it encounters a single real user.

---

## 5. An Important Paradox: Steerable But Stable

A reader might worry: if therapeutic traits are steerable, doesn't that mean they're unstable? If we can push empathy up and down with a vector, isn't the model's persona inherently fragile?

The answer, empirically, is no. We assessed natural persona stability across 500 challenge scenarios per model — deliberately including crisis situations, validation-seeking, boundary-testing, adversarial prompts, and extended emotional conversations. We tracked activation projections across 2–5 conversational turns per scenario.

**No model showed significant drift on any trait.** All 27 trait×model combinations fell within calibrated 2σ thresholds.

The analogy is a thermostat. A heating system is highly *controllable* (turning the dial changes temperature predictably) yet naturally *stable* (temperature doesn't drift on its own). Controllability and spontaneous instability are different properties. Our monitoring system is designed to detect anomalous deviations from a stable baseline — whether caused by unusual user inputs, model updates, or adversarial manipulation — not to track expected degradation.

---

## 6. Discussion

### What We Learned About Model Internals

This work contributes to an emerging picture of how large language models encode behavioral properties. Three findings stand out:

**Therapeutic traits live in the upper-middle layers.** Across all three architectures, optimal steering layers clustered between layers 14–19 (of 28–32 total), consistent with prior work placing abstract behavioral properties in intermediate-depth representations — below the output-facing logits but above token-level features.

**Behavioral relevance, not activation separation, predicts success.** The r=0.899 correlation between behavioral difference and steering success has methodological implications beyond this study. Anyone working with activation steering should validate vectors by behavioral effect, not by activation geometry. Large Cohen's d values are seductive but potentially meaningless.

**Models encode the same concepts in incompatible geometries.** This is perhaps our most important structural finding. Template vectors that work perfectly on Llama3 fail on Qwen2 not because Qwen2 lacks the concept but because it encodes it along a different direction. Contrastive probing — the architecture-general solution — works precisely because it abandons the assumption that directions are transferable and instead lets each model define its own.

### A Dual-Purpose Technology

The safety stress test reframes activation steering as a dual-purpose technology. In deployment, it monitors therapeutic quality through activation projections — a real-time "blood pressure cuff" for the chatbot's therapeutic persona. Before deployment, it audits safety vulnerabilities by systematically probing whether harmful steering works at each layer. The Mistral finding — safe at layer 10, critically vulnerable at layer 12 — would be invisible to conventional safety evaluation methods that treat the model as a black box. Activation steering reveals *where* safety training has purchase and where it doesn't.

### Limitations

**LLM judge validation.** Our reliance on GPT-4o-mini as behavioral judge may introduce systematic biases. Human-judge correlation studies are essential before clinical deployment. Prior literature suggests LLM judges achieve r≈0.6–0.8 with human evaluators on similar tasks, but domain-specific calibration is needed.

**Synthetic scenarios.** All evaluation used synthetic conversations. Real clinical interactions exhibit greater unpredictability, topic shifts, and emotional complexity. We attempted external validation against the ESConv dataset (196 failed conversations) but found zero conversations meeting our format requirements — a negative finding that highlights the gap between synthetic and real-world evaluation.

**Quantisation effects.** Our 4-bit NF4 quantisation alters representational geometry. Steering vectors and optimal layers may differ at full precision. However, 4-bit models are increasingly the deployment norm, making our results directly relevant to production configurations.

**Safety test scope.** We tested one harm dimension (harmful advice propensity) with extended coefficients (±5.0) beyond the normal monitoring range (±3.0). Real-world adversarial manipulation may target other dimensions or use more sophisticated multi-vector attacks. The vulnerability window we identified exists at extreme coefficients — a meaningful safety margin, but a finite one.

**Weak Mistral traits.** Three traits remain weak (r=0.215–0.271) due to insufficient contrastive training data (17–67 samples/class vs. 81–100 for validated traits). Scaling data collection is the likely fix, though genuine architectural limitations cannot be ruled out.

---

## 7. Conclusions

We set out to answer four questions. Here is what we found:

**Can therapeutic persona traits be controlled through activation steering?** Yes. Eight clinically relevant dimensions — from empathetic responsiveness to crisis recognition to harmful validation — exist as manipulable directions in LLM activation space. Turning a dial along these directions reliably shifts behavior as judged by an independent evaluator (r=0.302–0.489, all p<0.001).

**Does this generalise across architectures?** Not automatically — but it can be made to. Template vectors are architecture-specific. Contrastive probing, which lets each model define its own high/low trait directions, achieves near-universal coverage: 21/24 model×trait combinations validated, zero failures, all correlations positive and significant.

**Can activation projections monitor persona drift in real time?** Yes. Internal activation states track externally observable therapeutic qualities with mean r=0.596 across all 24 model×trait combinations (all p<0.0001). EWMA+CUSUM control charts achieve clinically acceptable false alarm rates (1–4% Warning+) while maintaining sensitivity to genuine drift.

**Can activation steering break through safety training?** It depends — critically — on both the architecture and the layer. Qwen2 is robustly resistant (max r=0.183, not significant). Mistral has a "safety gradient" (protective at layer 10, critically vulnerable at layers 12–15, neutral at 18–19). Llama3 shows moderate, broadly distributed vulnerability. Activation steering thus serves a dual purpose: monitoring therapeutic quality in deployment and auditing safety landscapes before it.

As AI systems assume expanding roles in mental health support — where the stakes include not just user satisfaction but psychological wellbeing and, in crisis situations, lives — technical mechanisms for ensuring persona consistency become not merely useful but ethically necessary. The monitoring framework demonstrated here offers one piece of the governance infrastructure that responsible deployment requires.

---

## Methods Summary

| Parameter | Value |
|-----------|-------|
| **Models** | Llama-3-8B-Instruct, Qwen2-7B-Instruct, Mistral-7B-Instruct-v0.2 |
| **Quantisation** | 4-bit NF4 (bitsandbytes, float16 compute, double quant) |
| **Hardware** | NVIDIA A10G (24 GB), Modal.com serverless |
| **Steering coefficients** | −3.0, −1.5, 0.0, +1.5, +3.0 (standard); −5.0 to +5.0 (safety test) |
| **Evaluation judge** | GPT-4o-mini via OpenRouter, temperature=0 |
| **Monitoring** | EWMA λ=0.2, CUSUM k=0.5σ h=4.0σ |
| **Phase 3 sessions** | 100 per model, 10 parallel batches |
| **Statistical analysis** | Pearson r, 95% CI via 1,000-iteration bootstrap |
| **Reproducibility** | MASTER_SEED=42, full runtime version logging |

---

## Data Availability

Code and generated response data are available at [repository URL]. The repository includes validation scripts (`step1_validate_traits.py`, `step1b_contrastive_probing.py`), monitoring pipeline (`step2_monitor_drift.py`), safety stress test (`step3_safety_stress_test.py`), complete contrast prompt sets for all traits with judge criteria, and full experimental outputs. Raw model weights are publicly available from their respective organisations.

---

## References

1. Turner A, Thiergart L, Leech G, et al. Steering Language Models With Activation Engineering. *arXiv:2308.10248*, 2024.
2. Zou A, Phan L, Chen S, et al. Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*, 2023.
3. Li K, Patel O, Viégas F, Pfister H, Wattenberg M. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. *NeurIPS 2023*.
4. Panickssery N, Gabrieli N, Schulz J, Tong M, Hubinger E, Turner AM. Steering Llama 2 via Contrastive Activation Addition. *arXiv:2312.06681*, 2024.
5. Chen R, Arditi A, Sleight H, Evans O, Lindsey J. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. *arXiv:2507.21509*, 2025.
6. Sharma M, Tong M, Korbak T, et al. Towards Understanding Sycophancy in Language Models. *arXiv:2310.13548*, 2023.
7. Bai Y, Kadavath S, Kundu S, et al. Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*, 2022.
8. Rogers CR. The necessary and sufficient conditions of therapeutic personality change. *Journal of Consulting Psychology* 1957; 21: 95–103.
9. Wampold BE. How important are the common factors in psychotherapy? An update. *World Psychiatry* 2015; 14: 270–277.
10. Miner AS, Milstein A, Schueller S, et al. Smartphone-based conversational agents and responses to questions about mental health. *JAMA Internal Medicine* 2016; 176: 619–625.
11. Fitzpatrick KK, Darcy A, Vierhile M. Delivering CBT to young adults using a fully automated conversational agent (Woebot). *JMIR Mental Health* 2017; 4: e19.
12. Abd-Alrazaq AA, Alajlani M, Ali N, et al. Perceptions and opinions of patients about mental health chatbots. *Journal of Medical Internet Research* 2021; 23: e17828.
13. Liu S, Zheng C, Demasi O, et al. Towards Emotional Support Dialog Systems. *ACL 2021*.
14. Page ES. Continuous Inspection Schemes. *Biometrika* 1954; 41(1/2): 100–115.
15. Roberts SW. Control Chart Tests Based on Geometric Moving Averages. *Technometrics* 1959; 1(3): 239–250.
