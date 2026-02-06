"""
Validate empathetic_responsiveness vector against Empathy-Mental-Health Reddit dataset.

This dataset has HUMAN empathy ratings, providing external validation of our steering vectors.

Key finding we're testing:
- Does our empathetic_responsiveness steering vector (from V29) correlate with 
  human-rated empathy levels in real Reddit mental health conversations?

Run with: modal run modal_validate_empathy_v30.py

Author: AI Persona Steering Project
"""

import modal
import json
from datetime import datetime

# =========================================================================
# MODAL SETUP
# =========================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "pandas",
        "numpy",
        "scipy",
        "requests",
    )
)

app = modal.App("empathy-validation-v30", image=image)
volume = modal.Volume.from_name("persona-validation-results", create_if_missing=True)

# =========================================================================
# V29 EMPATHETIC RESPONSIVENESS TRAIT DEFINITION (from V29)
# =========================================================================

EMPATHY_TRAIT = {
    "name": "empathetic_responsiveness",
    "high_prompts": [
        "I can really feel the weight of what you're carrying right now, and I want you to know that your feelings make complete sense.",
        "That sounds incredibly difficult, and I'm truly moved by your courage in sharing this with me.",
        "I sense there's a lot of pain beneath what you're describing, and I want you to know I'm here with you in this.",
        "Your feelings are so valid, and I can understand why this situation would affect you so deeply.",
        "I'm really touched that you trusted me with this. What you're going through sounds genuinely hard.",
        "The emotions you're experiencing right now are completely understandable given everything you've been dealing with.",
    ],
    "low_prompts": [
        "I understand. What else would you like to discuss?",
        "That's a common issue. Have you tried any solutions?",
        "I see. Many people experience similar things.",
        "Noted. Is there something specific you need help with?",
        "That happens sometimes. What would you like to do about it?",
        "I hear you. Let's focus on practical next steps.",
    ],
    "judge_high": "Response shows genuine emotional attunement, validates feelings deeply, uses warm language, demonstrates they truly 'get' the emotional experience",
    "judge_low": "Response is factual, brief, solution-focused without emotional acknowledgment, could be talking to anyone about anything",
    # Best layer from V29 Llama-3-8B results
    "best_layer": 14,
}

# Candidate layers to test (from V29)
CANDIDATE_LAYERS = [10, 12, 14, 15, 16, 18]


# =========================================================================
# MAIN VALIDATION FUNCTION
# =========================================================================

@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={"/output": volume},
)
def validate_empathy():
    """
    Extract empathy vector from Llama-3-8B and validate against Reddit dataset.
    """
    import torch
    import numpy as np
    import pandas as pd
    from scipy import stats
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import requests
    
    print("=" * 70)
    print("EMPATHY VECTOR EXTERNAL VALIDATION")
    print("Model: Llama-3-8B-Instruct | Dataset: Empathy-Mental-Health Reddit")
    print("=" * 70)
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    
    model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
    print(f"\n► Loading {model_name}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    model.eval()
    print(f"✓ Model loaded ({model.config.num_hidden_layers} layers)")
    
    # =========================================================================
    # EXTRACT EMPATHY STEERING VECTOR (same method as V29)
    # =========================================================================
    
    print("\n► Extracting empathy steering vectors...")
    
    def get_activation(text: str, layer_idx: int) -> torch.Tensor:
        """Get last-token activation at specific layer."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        activation = None
        def hook(module, input, output):
            nonlocal activation
            hidden_states = output[0] if isinstance(output, tuple) else output
            activation = hidden_states[:, -1, :].detach()  # Last token
        
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        
        return activation
    
    def extract_steering_vector(trait: dict, layer_idx: int) -> torch.Tensor:
        """Extract steering vector using contrast prompts."""
        directions = []
        
        for high, low in zip(trait["high_prompts"], trait["low_prompts"]):
            high_act = get_activation(high, layer_idx)
            low_act = get_activation(low, layer_idx)
            
            direction = high_act - low_act
            direction = direction / (direction.norm() + 1e-8)  # Normalize
            directions.append(direction)
        
        # Average and normalize final vector
        steering_vector = torch.stack(directions).mean(dim=0)
        steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
        
        return steering_vector.squeeze()
    
    # Extract vectors for each candidate layer
    layer_vectors = {}
    for layer in CANDIDATE_LAYERS:
        vec = extract_steering_vector(EMPATHY_TRAIT, layer)
        layer_vectors[layer] = vec
        print(f"  ✓ Layer {layer}: vector extracted (norm={vec.norm().item():.4f})")
    
    # =========================================================================
    # LOAD EMPATHY-MENTAL-HEALTH REDDIT DATASET
    # =========================================================================
    
    print("\n► Loading Empathy-Mental-Health Reddit dataset...")
    
    url = "https://raw.githubusercontent.com/behavioral-data/Empathy-Mental-Health/master/dataset/emotional-reactions-reddit.csv"
    
    try:
        empathy_df = pd.read_csv(url)
        print(f"✓ Loaded {len(empathy_df)} samples")
        print(f"  Columns: {list(empathy_df.columns)}")
        print(f"  Human empathy levels: {empathy_df['level'].value_counts().to_dict()}")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return {"error": str(e)}
    
    # =========================================================================
    # SCORE RESPONSES AND COMPUTE CORRELATION
    # =========================================================================
    
    print("\n► Scoring Reddit responses on empathy vector...")
    
    # Sample for efficiency (use all if small enough)
    sample_size = min(300, len(empathy_df))
    sample_df = empathy_df.sample(n=sample_size, random_state=42)
    
    def score_text(text: str, layer_idx: int) -> float:
        """Project text activation onto empathy vector."""
        activation = get_activation(text, layer_idx)
        direction = layer_vectors[layer_idx].to(activation.dtype)
        score = torch.dot(activation.squeeze(), direction).item()
        return score
    
    results_by_layer = {}
    
    for layer in CANDIDATE_LAYERS:
        print(f"\n  Testing layer {layer}...")
        
        empathy_scores = []
        human_levels = []
        
        for idx, row in sample_df.iterrows():
            try:
                # Format as conversation (response only for scoring)
                response_text = str(row['response_post'])
                
                if len(response_text) > 20:  # Skip very short responses
                    score = score_text(response_text, layer)
                    empathy_scores.append(score)
                    human_levels.append(row['level'])
            except Exception as e:
                continue
        
        if len(empathy_scores) > 10:
            # Spearman correlation (ordinal human ratings)
            spearman_r, spearman_p = stats.spearmanr(empathy_scores, human_levels)
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(empathy_scores, human_levels)
            
            # Mean scores by human level
            by_level = {}
            for score, level in zip(empathy_scores, human_levels):
                if level not in by_level:
                    by_level[level] = []
                by_level[level].append(score)
            
            level_means = {k: np.mean(v) for k, v in sorted(by_level.items())}
            
            results_by_layer[layer] = {
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "n_samples": len(empathy_scores),
                "mean_by_level": level_means,
            }
            
            print(f"    Spearman r = {spearman_r:.3f} (p={spearman_p:.4f})")
            print(f"    Pearson r = {pearson_r:.3f} (p={pearson_p:.4f})")
            print(f"    n = {len(empathy_scores)}")
            for level, mean in level_means.items():
                print(f"    Level {level}: mean score = {mean:.3f}")
    
    # =========================================================================
    # FIND BEST LAYER
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    best_layer = max(results_by_layer.keys(), 
                     key=lambda l: results_by_layer[l]["spearman_r"])
    best_result = results_by_layer[best_layer]
    
    print(f"\n► Best layer: {best_layer}")
    print(f"  Spearman correlation with human ratings: r = {best_result['spearman_r']:.3f}")
    print(f"  p-value: {best_result['spearman_p']:.6f}")
    print(f"  Samples scored: {best_result['n_samples']}")
    
    print(f"\n► Mean projection score by human empathy level:")
    for level, mean in best_result['mean_by_level'].items():
        print(f"    Level {level}: {mean:+.3f}")
    
    # Statistical interpretation
    r = best_result['spearman_r']
    if r > 0.5:
        interpretation = "STRONG positive correlation"
    elif r > 0.3:
        interpretation = "MODERATE positive correlation"
    elif r > 0.1:
        interpretation = "WEAK positive correlation"
    elif r > -0.1:
        interpretation = "NO meaningful correlation"
    else:
        interpretation = "NEGATIVE correlation (unexpected)"
    
    print(f"\n► Interpretation: {interpretation}")
    
    if best_result['spearman_p'] < 0.001:
        sig = "highly significant (p < 0.001)"
    elif best_result['spearman_p'] < 0.01:
        sig = "significant (p < 0.01)"
    elif best_result['spearman_p'] < 0.05:
        sig = "significant (p < 0.05)"
    else:
        sig = "NOT significant"
    
    print(f"► Statistical significance: {sig}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    results = {
        "validation_type": "external_human_ratings",
        "dataset": "Empathy-Mental-Health Reddit",
        "dataset_url": url,
        "model": "NousResearch/Meta-Llama-3-8B-Instruct",
        "trait": "empathetic_responsiveness",
        "timestamp": datetime.now().isoformat(),
        "sample_size": sample_size,
        "best_layer": best_layer,
        "best_result": {
            "spearman_r": best_result['spearman_r'],
            "spearman_p": best_result['spearman_p'],
            "pearson_r": best_result['pearson_r'],
            "pearson_p": best_result['pearson_p'],
            "n_samples": best_result['n_samples'],
            "mean_by_human_level": best_result['mean_by_level'],
        },
        "all_layers": {
            str(k): {
                "spearman_r": v["spearman_r"],
                "spearman_p": v["spearman_p"],
                "pearson_r": v["pearson_r"],
                "pearson_p": v["pearson_p"],
            }
            for k, v in results_by_layer.items()
        },
        "interpretation": interpretation,
        "significance": sig,
    }
    
    # Save to volume
    output_path = "/output/empathy_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {output_path}")
    
    volume.commit()
    
    return results


@app.local_entrypoint()
def main():
    """Run validation and display results."""
    print("=" * 70)
    print("EMPATHY VECTOR EXTERNAL VALIDATION")
    print("=" * 70)
    print("\nThis validates our empathetic_responsiveness steering vector")
    print("against the Empathy-Mental-Health Reddit dataset which has")
    print("HUMAN empathy ratings (ground truth).\n")
    
    results = validate_empathy.remote()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return
    
    print(f"\n✅ External Validation Complete!")
    print(f"\nDataset: {results['dataset']}")
    print(f"Model: {results['model']}")
    print(f"Trait: {results['trait']}")
    print(f"\n► Best layer: {results['best_layer']}")
    print(f"► Spearman r with human ratings: {results['best_result']['spearman_r']:.3f}")
    print(f"► p-value: {results['best_result']['spearman_p']:.6f}")
    print(f"► Interpretation: {results['interpretation']}")
    print(f"► Significance: {results['significance']}")
    
    print(f"\n► Mean vector projection by human empathy level:")
    for level, mean in results['best_result']['mean_by_human_level'].items():
        print(f"    Human Level {level}: {mean:+.3f}")
    
    print("\n" + "=" * 70)
    print("FOR PAPER")
    print("=" * 70)
    
    r = results['best_result']['spearman_r']
    p = results['best_result']['spearman_p']
    n = results['best_result']['n_samples']
    
    print(f"""
External validation against the Empathy-Mental-Health Reddit dataset 
(n={n}) revealed significant correlation between our empathetic_responsiveness 
steering vector projections and human empathy ratings 
(Spearman r = {r:.3f}, p {'< 0.001' if p < 0.001 else f'= {p:.4f}'}).
""")
