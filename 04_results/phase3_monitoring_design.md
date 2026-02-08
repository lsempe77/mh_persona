# Phase 3: Real-Time Persona Drift Monitoring — Design Specification

> **Date:** February 2026  
> **Author:** AI Persona Steering Project  
> **Foundation:** Phase 2 validation (21/24 model×trait validated) + Chen et al. 2025 "Persona Vectors"  
> **Target:** Lancet Digital Health publication quality

---

## 1. Research Summary

### 1.1 How Chen et al. Monitor Persona Drift

From the [persona_vectors](https://github.com/safety-research/persona_vectors) codebase, monitoring works as follows:

1. **Vector extraction**: Compute `response_avg_diff` — the mean activation difference between high-trait and low-trait responses, averaged across all response tokens at a specific layer. Shape: `[layers × hidden_dim]`.

2. **Projection calculation** (`eval/cal_projection.py`):
   - For each model response, extract hidden states at the target layer
   - Compute `response_avg = hidden_states[layer][:, prompt_len:, :].mean(dim=1)` — mean-pooled response activations
   - Project onto the persona vector: `projection = (response_avg · vector) / ||vector||` (scalar projection)
   - Alternative: cosine similarity `cos_sim(response_avg, vector)`
   
3. **Correlation with behavior**: The projection score correlates with LLM-judge behavioral scores (r = 0.76–0.97 in their paper).

**Key insight:** Chen et al. monitor at the **per-response level**, not per-token. Each model output gets a single scalar projection score per trait. They do NOT implement online drift detection — they compute projections batch-wise and correlate with behavioral scores.

### 1.2 How Our System Differs

Our system extends Chen et al. in two critical ways:

| Aspect | Chen et al. | Our System |
|--------|-------------|------------|
| **Scope** | General personality traits (evil, sycophancy, hallucination) | Mental health therapy-specific traits (8 traits) |
| **Vectors** | Template-based (system prompt contrast) | Contrastive probing (data-driven) for Qwen2/Mistral |
| **Monitoring** | Batch post-hoc analysis | **Real-time per-turn tracking with alerts** |
| **Detection** | Simple projection → correlation | **EWMA + threshold-based drift detection** |
| **Application** | Training data screening / post-hoc evaluation | **Live clinical safety monitoring** |

### 1.3 Drift Detection Methods Reviewed

| Method | Mechanism | Strengths | Weaknesses | Verdict |
|--------|-----------|-----------|------------|---------|
| **CUSUM** | Cumulative sum of deviations from target | Optimal for detecting small persistent shifts; well-studied ARL properties | Binary alert (no severity gradient); requires specification of shift magnitude | ✅ Include as secondary detector |
| **EWMA** | Exponentially weighted moving average with control limits | Smooth, interpretable trajectory; natural severity gradient; tunable λ | Less optimal for abrupt shifts | ✅ **Primary method** |
| **Shewhart chart** | Individual point vs. ±3σ limits | Detects large sudden shifts | Misses small persistent drift | ❌ Too insensitive |
| **Change-point (PELT/BOCPD)** | Bayesian or penalized offline detection | Can pinpoint exact change time | Typically offline; computationally heavier | ❌ Not real-time friendly |

**Decision: EWMA as primary, CUSUM as secondary alert.**

---

## 2. Architecture Design

### 2.1 System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     CONVERSATION STREAM                          │
│  User message → Model generates response → Next turn...         │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                  ACTIVATION EXTRACTOR                            │
│  Hook at best layer per trait → Extract response-avg hidden      │
│  state → Project onto steering vector → 8 scalar scores/turn    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   DRIFT TRACKER (per trait)                      │
│  EWMA smoothing → Control limits → CUSUM accumulator            │
│  Outputs: smoothed_score, z_score, cusum_stat, alert_level      │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ALERT SYSTEM                                  │
│  None → Watch → Warning → Critical                              │
│  Based on: z-score thresholds + CUSUM threshold                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   OUTPUT / DASHBOARD                             │
│  Per-turn trajectory plot │ Alert log │ Multi-trait heatmap      │
│  JSON results for analysis │ Optional: live Streamlit dashboard  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

#### A. Activation Extractor

```python
class ActivationExtractor:
    """Extract per-turn trait projections from model hidden states."""
    
    def __init__(self, model, tokenizer, steering_vectors, trait_layers):
        """
        Args:
            model: The loaded LLM (HuggingFace, 4-bit quantized)
            tokenizer: Corresponding tokenizer
            steering_vectors: Dict[trait_name → 1D tensor (hidden_dim,)]
            trait_layers: Dict[trait_name → int] (best layer per trait)
        """
        
    def extract_projections(self, prompt: str, response: str) -> Dict[str, float]:
        """
        For a single turn (prompt + response):
        1. Tokenize full text (prompt + response)
        2. Forward pass with output_hidden_states=True
        3. For each trait:
           a. Get hidden states at trait's best layer
           b. Mean-pool response tokens: h = hidden[layer][:, prompt_len:, :].mean(dim=1)
           c. Project: score = (h · v) / ||v||  where v is the steering vector
        4. Return dict of {trait: projection_score}
        """
```

**Key design decisions:**
- **Response-average pooling** (matching Chen et al.'s `response_avg`), not last-token. Rationale: for monitoring, we want the overall trait expression across the full response, not just the generation direction at the last token. Our validation used last-token for *steering*, but Chen et al. found response-average works better for *measurement*.
- **Scalar projection** `a_proj_b(response_avg, vector)`, not cosine similarity. Rationale: projection preserves magnitude information — a strongly trait-expressing response should have a larger projection than a mildly expressing one.
- **Per-trait layer selection**: Each trait uses its empirically validated best layer (from `trait_layer_matrix_*.json`).

#### B. Drift Tracker

```python
class DriftTracker:
    """Track per-trait projection scores over conversation turns."""
    
    def __init__(self, trait_name: str, baseline_mean: float, baseline_std: float,
                 ewma_lambda: float = 0.2, cusum_threshold: float = 4.0,
                 cusum_slack: float = 0.5):
        """
        Args:
            baseline_mean: Expected projection score under normal behavior
            baseline_std: Standard deviation of projection scores
            ewma_lambda: EWMA weight (0.1-0.3 recommended; 0.2 = balanced)
            cusum_threshold: CUSUM alert threshold (in σ units)
            cusum_slack: CUSUM slack parameter (allowable deviation before accumulating)
        """
        self.ewma = baseline_mean  # Initialize EWMA at baseline
        self.cusum_high = 0.0      # Positive-direction CUSUM
        self.cusum_low = 0.0       # Negative-direction CUSUM
        self.history = []          # Raw projection scores
        self.ewma_history = []     # Smoothed scores
        
    def update(self, projection_score: float) -> Dict:
        """
        Process a new turn's projection score.
        
        Returns dict with:
            raw_score: The input projection
            ewma_score: Smoothed score
            z_score: (ewma - baseline_mean) / baseline_std
            cusum_high: Cumulative positive deviation
            cusum_low: Cumulative negative deviation
            alert_level: 'none' | 'watch' | 'warning' | 'critical'
        """
        # 1. Update EWMA
        self.ewma = ewma_lambda * projection_score + (1 - ewma_lambda) * self.ewma
        
        # 2. Compute z-score
        z = (self.ewma - self.baseline_mean) / self.baseline_std
        
        # 3. Update CUSUM (two-sided)
        self.cusum_high = max(0, self.cusum_high + z - self.cusum_slack)
        self.cusum_low = max(0, self.cusum_low - z - self.cusum_slack)
        
        # 4. Determine alert level
        alert = self._compute_alert(z, self.cusum_high, self.cusum_low)
        
        return {...}
```

**EWMA parameter selection:**
- `λ = 0.2`: Balances responsiveness vs. smoothing. With λ=0.2, approximately the last 10 turns significantly influence the EWMA (since $(1-0.2)^{10} ≈ 0.107$). This matches a typical therapy session length (45-60 min ≈ 15-30 turns).
- Control limits: $\text{EWMA} \pm L \cdot \sigma \sqrt{\frac{\lambda}{2-\lambda}}$ where $L = 3$ (standard). With λ=0.2, this gives $\pm 3 \cdot \sigma \cdot \sqrt{0.111} \approx \pm σ$.

**CUSUM parameter selection:**
- `slack = 0.5σ`: Allows natural fluctuation of ±0.5σ before accumulating evidence.
- `threshold = 4.0`: Standard choice, gives ARL₀ ≈ 168 (false alarm every ~168 turns under normal operation — roughly 6 sessions).

#### C. Alert System

| Level | Trigger | Meaning | Action |
|-------|---------|---------|--------|
| **None** | \|z\| < 1.5 | Normal variation | Continue monitoring |
| **Watch** 🟡 | 1.5 ≤ \|z\| < 2.5 | Mild deviation, could be transient | Log for supervisor review |
| **Warning** 🟠 | 2.5 ≤ \|z\| < 3.5 OR CUSUM > 3.0 | Significant drift detected | Flag for immediate review |
| **Critical** 🔴 | \|z\| ≥ 3.5 OR CUSUM > 4.0 | Severe drift, safety concern | Trigger intervention protocol |

**Clinical safety considerations:**
- For **safety-critical traits** (crisis_recognition, boundary_maintenance, sycophancy_harmful_validation), use **tighter thresholds** (Warning at |z| ≥ 2.0 instead of 2.5).
- For **quality traits** (empathetic_responsiveness, non_judgmental_acceptance), use standard thresholds.
- **Directional sensitivity**: Some traits drift is concerning in only one direction:
  - `crisis_recognition` decreasing = dangerous (missing crises)
  - `emotional_over_involvement` increasing = concerning (enmeshment)
  - `boundary_maintenance` decreasing = concerning (boundary violations)
  - `sycophancy_harmful_validation` increasing = dangerous (validating harm)

### 2.3 Baseline Calibration

The baseline (mean, std) for each trait is established during a **calibration phase**:

1. Run the model on 50-100 **normal therapy scenarios** (from our validation set, coefficient=0)
2. Extract per-turn projections for each trait
3. Compute `baseline_mean = mean(projections)`, `baseline_std = std(projections)`
4. Store in calibration file alongside the model's steering vectors

This parallels the "Phase I" of control chart methodology — establishing in-control process parameters before monitoring begins.

---

## 3. Vectors to Use Per Model

| Model | Vector Source | Best Layers |
|-------|-------------|-------------|
| **Llama3-8B-Instruct** | Template steering vectors | emp:17, nja:18, bm:18, cr:18, eoi:19, atf:19, uv:18, shv:19 |
| **Qwen2-7B-Instruct** | Probe vectors (Solution B) | emp:15, nja:15, bm:17, cr:16, eoi:12, atf:19, uv:14, shv:17 |
| **Mistral-7B-Instruct-v0.2** | Probe vectors (Solution B) | From `trait_layer_matrix_probe_mistral.json` |

**Important:** For Qwen2 and Mistral, we use the contrastive probing vectors (logistic regression weight vectors), NOT template vectors. These are stored in `steering_vectors_probe_{qwen2,mistral}.pkl`.

---

## 4. Validation Strategy

### 4.1 Internal Validation (Phase 3.1)

Using our existing 500-scenario validation data:

1. **Replay validation**: Take 20 scenarios, simulate a multi-turn conversation (5 turns each), track projections over turns, verify that EWMA trajectory matches known behavioral drift.
2. **Alert calibration**: Run 100 normal scenarios → verify false alarm rate < 5%.
3. **Sensitivity test**: Run 20 steered scenarios (coefficient ≠ 0) → verify detection within 3-5 turns.

### 4.2 Cross-Model Consistency (Phase 3.2)

Same scenarios across Llama3, Qwen2, Mistral:
- Do all models show similar drift trajectories on the same scenarios?
- Are alert thresholds comparable across architectures?

### 4.3 Clinical Face Validity (Phase 3.3)

Expert review of:
- 10 scenarios where monitoring correctly detected drift
- 5 scenarios where monitoring false-alarmed
- 5 scenarios where drift was missed

---

## 5. Output Formats

### 5.1 Per-Session JSON

```json
{
    "session_id": "session_001",
    "model": "llama3",
    "n_turns": 15,
    "calibration": {
        "empathetic_responsiveness": {"baseline_mean": 1.2, "baseline_std": 0.45},
        ...
    },
    "turns": [
        {
            "turn": 1,
            "timestamp": "2026-02-09T10:00:00",
            "traits": {
                "empathetic_responsiveness": {
                    "raw_projection": 1.35,
                    "ewma": 1.23,
                    "z_score": 0.07,
                    "cusum_high": 0.0,
                    "cusum_low": 0.0,
                    "alert": "none"
                },
                ...
            },
            "any_alert": false,
            "max_alert_level": "none"
        },
        ...
    ],
    "summary": {
        "total_alerts": 2,
        "alert_traits": ["crisis_recognition"],
        "max_severity": "warning",
        "drift_detected": true
    }
}
```

### 5.2 Trajectory Visualization

For each session, generate:
1. **8-panel trait trajectory plot**: Raw + EWMA + control limits per trait, with alert zones shaded
2. **Multi-trait heatmap**: Turns (x-axis) × traits (y-axis), color = z-score
3. **Alert timeline**: Horizontal bar showing when each alert was triggered

---

## 6. Implementation Plan

### Phase 3.1: Core Monitoring Engine (MVP)

**File:** `03_code/step2_monitor_drift.py`  
**Platform:** Modal (A10G GPU)

```
Inputs:
  - Model + tokenizer (4-bit quantized)
  - Steering vectors (template or probe, per model)
  - Multi-turn conversation scenarios
  
Processing:
  1. Load model + vectors on GPU
  2. For each scenario:
     a. For each turn in conversation:
        - Generate response
        - Extract hidden states at all needed layers
        - Project onto each trait's steering vector
        - Update EWMA + CUSUM trackers
     b. Log per-turn results
  3. Aggregate statistics
  
Outputs:
  - Per-session JSON with all tracking data
  - Summary statistics (false alarm rate, detection sensitivity)
  - Trajectory plots (saved as PNG)
```

### Phase 3.2: Cross-Model Comparison

Run the same scenarios on all 3 models, compare:
- Drift trajectory similarity
- Alert concordance
- False alarm rates

### Phase 3.3: Dashboard Prototype

**Option A:** Streamlit app (local)  
**Option B:** Static HTML report (generated per session)

For Lancet publication, static visualizations are sufficient.

### Phase 3.4: Paper Figures

Generate publication-quality figures:
1. Example drift trajectory (single session, 3 traits)
2. Cross-model comparison panel
3. ROC curve: activation-based alerts vs. behavioral-judge ground truth
4. Heatmap of trait co-drift patterns

---

## 7. Key Design Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pooling method | Response-average | Chen et al. use this for measurement; captures overall trait expression |
| Projection type | Scalar projection (dot product / ‖v‖) | Preserves magnitude; cosine only gives direction |
| Smoothing | EWMA (λ=0.2) | ~10-turn effective window matches therapy session length |
| Secondary detector | CUSUM | Catches small persistent drift that EWMA might smooth over |
| Baseline | Per-model calibration on unsteered responses | Accounts for model-specific activation scales |
| Alert thresholds | Asymmetric, trait-dependent | Safety-critical traits get tighter thresholds |
| Vector source | Template for Llama3, Probe for Qwen2/Mistral | Use each model's best-validated vectors |

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| High false alarm rate | Medium | Reduces clinical trust | Calibrate on 100+ normal sessions; tune λ and thresholds |
| Missed critical drift | Low | Patient safety | Use asymmetric thresholds; safety traits get 2× sensitivity |
| Response-avg ≠ last-token performance | Medium | Lower r-values in monitoring vs. validation | Test both pooling methods; may need recalibration |
| EWMA lag on abrupt shifts | Low | Delayed detection | CUSUM catches these; combined system has redundancy |
| Cross-model incomparability | Low | Can't aggregate alerts | Per-model calibration normalizes to z-scores |

---

## 9. Success Criteria for Phase 3

| Metric | Target | Meaning |
|--------|--------|---------|
| Detection sensitivity | > 80% | Correctly flags sessions where behavioral drift > 2 points |
| False alarm rate | < 10% | < 10% of normal sessions trigger Warning+ alerts |
| Detection latency | ≤ 5 turns | Alert triggered within 5 turns of drift onset |
| Cross-model concordance | r > 0.5 | Models agree on which sessions are drifting |
| Clinical face validity | Expert rating ≥ 4/5 | Clinician deems alerts clinically meaningful |

---

## 10. Open Questions

1. **Response-average vs. last-token for monitoring**: Our validation used last-token extraction for *steering* vectors, but Chen et al. use response-average for *monitoring*. We should test both and compare r-values against behavioral scores in the monitoring context.

2. **Multi-turn context effects**: Our validation measured t1→tN drift across two isolated turns. In a real conversation, context accumulates. Does the projection naturally track with conversational context, or do we need to control for conversation depth?

3. **Conversation-level vs. turn-level alerts**: Should we alert on individual turns, or only on sustained drift patterns? The EWMA+CUSUM design favors sustained patterns, but a single severely unsafe response should trigger immediate alerts regardless.

4. **Probe vector stability**: Our probe vectors were trained on 100 scored responses per class. Are they stable enough for production monitoring, or do they need periodic recalibration?

---

## References

1. Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. *arXiv:2507.21509*.
2. Turner, A., et al. (2023). Activation Addition: Steering Language Models Without Optimization. *arXiv:2308.10248*.
3. Panickssery, A., et al. (2023). Steering Llama 2 via Contrastive Activation Addition. *arXiv:2312.06681*.
4. Page, E. S. (1954). Continuous Inspection Schemes. *Biometrika*, 41(1/2), 100–115.
5. Roberts, S. W. (1959). Control Chart Tests Based on Geometric Moving Averages. *Technometrics*, 1(3), 239–250.
