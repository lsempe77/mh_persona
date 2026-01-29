# Code Development Plan

## Phase 1: Persona Vector Extraction (Months 1-3)

### Setup
```bash
# Required packages (to be installed)
pip install torch transformers accelerate
pip install nnsight  # For activation access
pip install datasets  # For loading ESConv, etc.
```

### Key Scripts to Develop

1. `persona_vectors/extract_vectors.py`
   - Implement automated persona vector extraction from Chen et al.
   - Input: trait description (natural language)
   - Output: direction vector in activation space

2. `persona_vectors/therapeutic_traits.py`
   - Define mental health-specific traits
   - Generate contrastive prompts for each trait

3. `utils/model_utils.py`
   - Model loading helpers
   - Activation extraction functions

## Phase 2: Drift Measurement (Months 4-6)

4. `experiments/measure_drift.py`
   - Track activation trajectories during conversations
   - Compute drift metrics

5. `experiments/challenge_scenarios.py`
   - Load/generate mental health challenge scenarios
   - Run drift experiments

## Phase 3: Interventions (Months 7-10)

6. `persona_vectors/activation_capping.py`
   - Implement inference-time activation capping
   - Define safe regions

7. `experiments/evaluate_interventions.py`
   - Test safety improvements
   - Measure quality trade-offs

---

## Models to Test

- [ ] Llama 3.1 8B Instruct
- [ ] Llama 3.1 70B Instruct
- [ ] Qwen 2.5 7B Instruct
- [ ] Mistral 7B Instruct

## Compute Requirements

- GPU: A100 40GB minimum for 7B models
- Storage: ~50GB for model weights
- Cloud budget: estimate needed
