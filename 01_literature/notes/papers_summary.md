# AI Persona Research Papers Summary

This document summarizes two related research papers on understanding and controlling AI personas in large language models.

---

## Paper 1: Persona Vectors: Monitoring and Controlling Character Traits in Language Models

**Authors:** Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey

**ArXiv:** [2507.21509](https://arxiv.org/abs/2507.21509) (July 2025)

**Topics:** Computational Linguistics (cs.CL), Machine Learning (cs.LG)

### Abstract

Large language models interact with users through a simulated 'Assistant' persona. While the Assistant is typically trained to be helpful, harmless, and honest, it sometimes deviates from these ideals.

### Key Contributions

1. **Persona Vectors Discovery**: Identified directions in the model's activation space—*persona vectors*—underlying several traits such as:
   - Evil
   - Sycophancy
   - Propensity to hallucinate

2. **Deployment Monitoring**: These vectors can be used to monitor fluctuations in the Assistant's personality at deployment time.

3. **Training Control**: Persona vectors can predict and control personality shifts that occur during training:
   - Both intended and unintended personality changes after finetuning are strongly correlated with shifts along the relevant persona vectors
   - These shifts can be mitigated through **post-hoc intervention**
   - Or avoided with a new **preventative steering method**

4. **Training Data Flagging**: Persona vectors can flag training data that will produce undesirable personality changes:
   - At the dataset level
   - At the individual sample level

5. **Automated Extraction**: The method for extracting persona vectors is automated and can be applied to any personality trait of interest, given only a natural-language description.

### Implications

This work provides tools for AI safety by enabling monitoring and control of undesirable personality traits in deployed LLMs.

---

## Paper 2: The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

**Authors:** Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey

**ArXiv:** [2601.10387](https://arxiv.org/abs/2601.10387) (January 2026)

**Topics:** Computational Linguistics (cs.CL)

### Abstract

Large language models can represent a variety of personas but typically default to a helpful Assistant identity cultivated during post-training. This paper investigates the structure of the space of model personas by extracting activation directions corresponding to diverse character archetypes.

### Key Findings

1. **The Assistant Axis**: Across several different models, the leading component of the persona space is an "Assistant Axis," which captures the extent to which a model is operating in its default Assistant mode.

2. **Steering Effects**:
   - **Towards Assistant direction**: Reinforces helpful and harmless behavior
   - **Away from Assistant direction**: Increases the model's tendency to identify as other entities
   - **Extreme negative values**: Often induces a mystical, theatrical speaking style

3. **Pre-trained Models**: The axis is also present in pre-trained models, where it:
   - Promotes helpful human archetypes (consultants, coaches)
   - Inhibits spiritual archetypes

4. **Persona Drift Prediction**: Measuring deviations along the Assistant Axis predicts "persona drift"—where models slip into exhibiting harmful or bizarre behaviors uncharacteristic of their typical persona.

5. **Drift Triggers**: Persona drift is often driven by:
   - Conversations demanding meta-reflection on the model's processes
   - Interactions featuring emotionally vulnerable users

6. **Stabilization Method**: Restricting activations to a fixed region along the Assistant Axis can stabilize model behavior:
   - In drift-prone scenarios
   - Against adversarial persona-based jailbreaks

### Key Insight

Post-training steers models toward a particular region of persona space but only **loosely tethers** them to it. This motivates work on training and steering strategies that more deeply anchor models to a coherent persona.

---

## Common Themes & Connections

| Aspect | Persona Vectors | Assistant Axis |
|--------|-----------------|----------------|
| **Focus** | Specific traits (evil, sycophancy, hallucination) | Overall persona space structure |
| **Application** | Monitoring & control during training/deployment | Understanding & stabilizing default behavior |
| **Method** | Automated persona vector extraction | Principal component analysis of persona space |
| **Safety Angle** | Flag problematic training data | Prevent persona drift & jailbreaks |
| **Shared Author** | Jack Lindsey | Jack Lindsey |

### Practical Applications

1. **AI Safety Monitoring**: Real-time detection of personality drift in deployed systems
2. **Training Quality Control**: Identifying training samples that may cause undesirable changes
3. **Robustness Improvements**: Steering methods to maintain helpful Assistant behavior
4. **Jailbreak Prevention**: Activation-space interventions against adversarial attacks

---

## References

- Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*. arXiv:2507.21509
- Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*. arXiv:2601.10387
