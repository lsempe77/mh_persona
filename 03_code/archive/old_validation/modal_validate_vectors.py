"""
Validate persona vectors against real mental health datasets.

Datasets used:
1. Empathy-Mental-Health Reddit (has human empathy ratings)
2. Zenodo Escalating Prompts (crisis progression testing)
3. Friend & Expert Responses (casual vs professional tone)
4. ShenLab/MentalChat16K (counselor-client conversations)

Run with: modal run modal_validate_vectors.py
"""

import modal
import os
from pathlib import Path

# =========================================================================
# MODAL SETUP
# =========================================================================

# Local path where vectors are stored
LOCAL_VECTORS_DIR = Path(__file__).parent / "04_results" / "vectors"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "pandas",
        "numpy",
        "scipy",
        "requests",
        "openpyxl",  # For Excel files
    )
)

app = modal.App("persona-vector-validation", image=image)

# Use the same volume where vectors are stored
volume = modal.Volume.from_name("persona-vectors", create_if_missing=True)


# =========================================================================
# UPLOAD VECTORS TO MODAL VOLUME
# =========================================================================

@app.function(volumes={"/output": volume})
def upload_vectors(vectors_data: dict):
    """Upload local vectors to Modal volume."""
    import torch
    
    for name, data in vectors_data.items():
        # Reconstruct tensor from numpy list
        tensor = torch.tensor(data["direction"])
        save_data = {
            "direction": tensor,
            "trait_name": data["trait_name"],
            "layer_idx": data["layer_idx"],
            "strength": data["strength"],
            "metadata": data["metadata"]
        }
        path = f"/output/{name}"
        torch.save(save_data, path)
        print(f"  ✓ Uploaded {name}")
    
    volume.commit()
    return len(vectors_data)


# =========================================================================
# VALIDATION FUNCTION
# =========================================================================

@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={"/output": volume},
)
def validate_vectors():
    """Load vectors and validate against real datasets."""
    
    import torch
    import numpy as np
    import pandas as pd
    from scipy import stats
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import requests
    import json
    
    print("="*60)
    print("PERSONA VECTOR VALIDATION")
    print("="*60)
    
    # =========================================================================
    # LOAD MODEL (same as extraction)
    # =========================================================================
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"\n► Loading {model_name}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    print(f"✓ Model loaded")
    
    # =========================================================================
    # LOAD EXTRACTED VECTORS
    # =========================================================================
    
    print("\n► Loading extracted persona vectors...")
    
    vectors = {}
    vector_files = [f for f in os.listdir("/output") if f.endswith("_vector.pt")]
    
    for vf in vector_files:
        data = torch.load(f"/output/{vf}", map_location="cuda")
        trait_name = data['trait_name']
        vectors[trait_name] = {
            'direction': data['direction'].to("cuda"),
            'layer_idx': data['layer_idx'],
            'strength': data['strength'],
        }
        print(f"  ✓ {trait_name} (layer {data['layer_idx']}, strength {data['strength']:.2f})")
    
    if not vectors:
        print("ERROR: No vectors found! Run modal_extract_vectors.py first.")
        return {"error": "No vectors found"}
    
    # =========================================================================
    # SCORING FUNCTION
    # =========================================================================
    
    def get_activation(text: str, layer_idx: int) -> torch.Tensor:
        """Get activation at specific layer for text."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        activation = None
        def hook(module, input, output):
            nonlocal activation
            # Handle different output shapes
            hidden_states = output[0]
            if hidden_states.dim() == 3:
                # Normal case: (batch, seq, hidden)
                activation = hidden_states[:, -1, :].detach()
            elif hidden_states.dim() == 2:
                # Already squeezed: (seq, hidden) - take last token
                activation = hidden_states[-1, :].unsqueeze(0).detach()
            else:
                raise ValueError(f"Unexpected hidden states shape: {hidden_states.shape}")
        
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        
        return activation
    
    def score_text(text: str, trait_name: str) -> float:
        """Score a text on a specific trait."""
        v = vectors[trait_name]
        activation = get_activation(text, v['layer_idx'])
        direction = v['direction'].to(activation.dtype)
        score = torch.dot(activation.squeeze(), direction).item()
        return score
    
    def score_all_traits(text: str) -> dict:
        """Score a text on all traits."""
        return {name: score_text(text, name) for name in vectors.keys()}
    
    # =========================================================================
    # DATASET 1: Empathy-Mental-Health Reddit
    # =========================================================================
    
    print("\n" + "="*60)
    print("DATASET 1: Empathy-Mental-Health Reddit")
    print("="*60)
    
    try:
        # Download from GitHub
        url = "https://raw.githubusercontent.com/behavioral-data/Empathy-Mental-Health/master/dataset/emotional-reactions-reddit.csv"
        empathy_df = pd.read_csv(url)
        print(f"✓ Loaded {len(empathy_df)} samples")
        
        # Sample for efficiency
        sample_size = min(200, len(empathy_df))
        empathy_sample = empathy_df.sample(n=sample_size, random_state=42)
        
        # Score responses
        empathy_scores = []
        human_levels = []
        
        for idx, row in empathy_sample.iterrows():
            try:
                # Format as conversation
                text = f"User: {row['seeker_post']}\nAssistant: {row['response_post']}"
                
                if 'empathetic_responsiveness' in vectors:
                    score = score_text(text, 'empathetic_responsiveness')
                    empathy_scores.append(score)
                    human_levels.append(row['level'])
            except Exception as e:
                continue
        
        if empathy_scores and 'empathetic_responsiveness' in vectors:
            # Compute correlation with human ratings
            correlation, p_value = stats.spearmanr(empathy_scores, human_levels)
            print(f"\n► Empathy Vector vs Human Ratings:")
            print(f"  Spearman correlation: {correlation:.3f} (p={p_value:.4f})")
            print(f"  Samples scored: {len(empathy_scores)}")
            
            # Group by human level
            by_level = {}
            for score, level in zip(empathy_scores, human_levels):
                if level not in by_level:
                    by_level[level] = []
                by_level[level].append(score)
            
            print(f"\n  Mean score by human empathy level:")
            for level in sorted(by_level.keys()):
                mean_score = np.mean(by_level[level])
                print(f"    Level {level}: {mean_score:+.2f} (n={len(by_level[level])})")
        
        empathy_results = {
            "correlation": correlation if empathy_scores else None,
            "p_value": p_value if empathy_scores else None,
            "n_samples": len(empathy_scores),
        }
        
    except Exception as e:
        print(f"Error loading Empathy dataset: {e}")
        empathy_results = {"error": str(e)}
    
    # =========================================================================
    # DATASET 2: Zenodo Escalating Prompts (Crisis Progression)
    # =========================================================================
    
    print("\n" + "="*60)
    print("DATASET 2: Zenodo Escalating Prompts")
    print("="*60)
    
    try:
        # This is an Excel file - download and parse
        url = "https://zenodo.org/records/8332778/files/LLMDATA.xlsx"
        response = requests.get(url)
        
        import io
        escalating_df = pd.read_excel(io.BytesIO(response.content))
        print(f"✓ Loaded escalating prompts data")
        print(f"  Columns: {list(escalating_df.columns)}")
        print(f"  Shape: {escalating_df.shape}")
        
        # Score the prompts if crisis_recognition vector exists
        if 'crisis_recognition' in vectors:
            # Look for prompt columns
            prompt_cols = [c for c in escalating_df.columns if 'prompt' in c.lower() or 'input' in c.lower()]
            if prompt_cols:
                print(f"\n► Scoring prompts on crisis_recognition:")
                sample = escalating_df.head(20)
                for idx, row in sample.iterrows():
                    for col in prompt_cols[:1]:  # Just first prompt column
                        if pd.notna(row[col]):
                            text = f"User: {row[col]}"
                            score = score_text(text, 'crisis_recognition')
                            print(f"  [{idx}] {score:+.2f} | {str(row[col])[:60]}...")
        
        escalating_results = {"loaded": True, "shape": escalating_df.shape}
        
    except Exception as e:
        print(f"Error loading Zenodo dataset: {e}")
        escalating_results = {"error": str(e)}
    
    # =========================================================================
    # DATASET 3: ShenLab MentalChat16K
    # =========================================================================
    
    print("\n" + "="*60)
    print("DATASET 3: ShenLab MentalChat16K")
    print("="*60)
    
    try:
        ds = load_dataset("ShenLab/MentalChat16K", split="train")
        print(f"✓ Loaded {len(ds)} conversations")
        print(f"  Columns: {ds.column_names}")
        
        # Sample and score
        sample_size = min(100, len(ds))
        indices = np.random.choice(len(ds), sample_size, replace=False)
        
        all_scores = {trait: [] for trait in vectors.keys()}
        
        print(f"\n► Scoring {sample_size} conversations on all traits...")
        
        for i, idx in enumerate(indices[:50]):  # First 50 for speed
            try:
                item = ds[int(idx)]
                # Try different column names
                if 'input' in item and 'output' in item:
                    text = f"User: {item['input']}\nAssistant: {item['output']}"
                elif 'question' in item and 'answer' in item:
                    text = f"User: {item['question']}\nAssistant: {item['answer']}"
                else:
                    # Just concatenate all text fields
                    text = " ".join(str(v) for v in item.values() if isinstance(v, str))
                
                scores = score_all_traits(text)
                for trait, score in scores.items():
                    all_scores[trait].append(score)
                    
            except Exception as e:
                continue
        
        print(f"\n► Score distributions across MentalChat16K:")
        mentalchat_stats = {}
        for trait, scores in all_scores.items():
            if scores:
                mean = np.mean(scores)
                std = np.std(scores)
                print(f"  {trait}: mean={mean:+.2f}, std={std:.2f}")
                mentalchat_stats[trait] = {"mean": mean, "std": std, "n": len(scores)}
        
        mentalchat_results = mentalchat_stats
        
    except Exception as e:
        print(f"Error loading MentalChat16K: {e}")
        mentalchat_results = {"error": str(e)}
    
    # =========================================================================
    # DATASET 4: Friend & Expert Responses (Kaggle - manual examples)
    # =========================================================================
    
    print("\n" + "="*60)
    print("DATASET 4: Tone Comparison (Friend vs Professional)")
    print("="*60)
    
    # Since Kaggle requires auth, use representative examples
    tone_examples = [
        {
            "input": "I've been feeling really anxious lately and can't sleep.",
            "friend": "Aw man, that sucks! Have you tried like, chamomile tea or something? My cousin swears by it. Maybe just chill and watch some Netflix to take your mind off things?",
            "professional": "I hear that you're experiencing anxiety and sleep difficulties. These often go together. Can you tell me more about when the anxiety started? It might help to establish a calming bedtime routine, but we should also explore what's underlying these feelings."
        },
        {
            "input": "I think I might be depressed. Nothing feels fun anymore.",
            "friend": "Dude, I feel you. Life can be such a drag sometimes. Wanna hang out this weekend? Getting out might help! Don't be so hard on yourself.",
            "professional": "Thank you for sharing that with me. Losing interest in things you used to enjoy is called anhedonia, and it's a common symptom of depression. How long have you been feeling this way? Have you spoken to a doctor about this?"
        },
        {
            "input": "My partner and I had a huge fight and I don't know what to do.",
            "friend": "Ugh, relationship drama is the worst! What happened? Do you want me to come over with ice cream? Sometimes you just gotta vent it out!",
            "professional": "Conflicts in relationships can be very distressing. Before we discuss solutions, I'd like to understand what the disagreement was about and how you're both feeling now. What do you think is at the core of this conflict?"
        },
        {
            "input": "I've been drinking a lot more than usual to cope.",
            "friend": "Hey, we all need to unwind sometimes! As long as you're not going overboard, right? But like, maybe try yoga or something too?",
            "professional": "I appreciate you being honest about your alcohol use. Using substances to cope can become a pattern that's hard to break. What are you trying to cope with? Let's also talk about some healthier alternatives and whether a screening might be helpful."
        },
    ]
    
    print(f"► Comparing Friend vs Professional responses:")
    
    tone_results = {"friend": {}, "professional": {}}
    
    for trait in vectors.keys():
        friend_scores = []
        prof_scores = []
        
        for ex in tone_examples:
            friend_text = f"User: {ex['input']}\nAssistant: {ex['friend']}"
            prof_text = f"User: {ex['input']}\nAssistant: {ex['professional']}"
            
            friend_scores.append(score_text(friend_text, trait))
            prof_scores.append(score_text(prof_text, trait))
        
        tone_results["friend"][trait] = np.mean(friend_scores)
        tone_results["professional"][trait] = np.mean(prof_scores)
    
    print(f"\n  {'Trait':<40} {'Friend':>10} {'Professional':>12} {'Diff':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*12} {'-'*10}")
    
    for trait in vectors.keys():
        f_score = tone_results["friend"][trait]
        p_score = tone_results["professional"][trait]
        diff = p_score - f_score
        print(f"  {trait:<40} {f_score:>+10.2f} {p_score:>+12.2f} {diff:>+10.2f}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    print("\n" + "="*60)
    print("SAVING VALIDATION RESULTS")
    print("="*60)
    
    results = {
        "empathy_reddit": empathy_results,
        "escalating_prompts": escalating_results,
        "mentalchat16k": mentalchat_results,
        "tone_comparison": tone_results,
    }
    
    with open("/output/validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("✓ Saved validation_results.json")
    
    volume.commit()
    
    return results


@app.function(image=image, volumes={"/output": volume})
def download_validation_results():
    """Download validation results to local machine."""
    import os
    
    results = {}
    for f in os.listdir("/output"):
        if f.endswith(".json"):
            with open(f"/output/{f}", "r") as file:
                results[f] = file.read()
    
    return results


@app.local_entrypoint()
def main():
    """Run validation and download results."""
    import os
    import json
    import torch
    
    print("="*60)
    print("PERSONA VECTOR VALIDATION PIPELINE")
    print("="*60)
    
    # First, upload local vectors to Modal volume
    print("\n► Uploading local vectors to Modal...")
    vectors_dir = LOCAL_VECTORS_DIR
    
    if not vectors_dir.exists():
        print(f"ERROR: Vectors directory not found: {vectors_dir}")
        print("Run modal_extract_vectors.py first!")
        return
    
    # Load vectors locally and prepare for upload
    vectors_data = {}
    for vf in vectors_dir.glob("*_vector.pt"):
        data = torch.load(vf, map_location="cpu", weights_only=False)
        # Convert tensor to list for serialization
        # The extraction saves with keys: trait_name, direction, layer_idx, strength, metadata
        # Convert bfloat16 to float32 first for numpy compatibility
        direction_tensor = data["direction"].float()
        vectors_data[vf.name] = {
            "direction": direction_tensor.numpy().tolist(),
            "trait_name": data["trait_name"],
            "layer_idx": data["layer_idx"],
            "strength": data["strength"],
            "metadata": data["metadata"]
        }
        print(f"  ✓ Loaded {vf.name}")
    
    if not vectors_data:
        print("ERROR: No vectors found!")
        return
    
    # Upload to Modal volume
    n_uploaded = upload_vectors.remote(vectors_data)
    print(f"  ✓ Uploaded {n_uploaded} vectors to Modal")
    
    # Run validation on Modal
    print("\n► Running validation on Modal GPU...")
    results = validate_vectors.remote()
    
    print("\n► Validation complete!")
    
    # Download results
    print("\n► Downloading results...")
    files = download_validation_results.remote()
    
    output_dir = "04_results/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, content in files.items():
        local_path = f"{output_dir}/{filename}"
        with open(local_path, 'w') as f:
            f.write(content)
        print(f"  ✓ Saved {local_path}")
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    # Print summary
    if "empathy_reddit" in results and "correlation" in results["empathy_reddit"]:
        corr = results["empathy_reddit"]["correlation"]
        if corr is not None:
            print(f"\n► Empathy vector correlation with human ratings: {corr:.3f}")
    
    if "tone_comparison" in results:
        print(f"\n► Tone comparison (Professional - Friend scores):")
        tc = results["tone_comparison"]
        for trait in tc.get("professional", {}).keys():
            diff = tc["professional"][trait] - tc["friend"][trait]
            direction = "↑" if diff > 0 else "↓"
            print(f"    {trait}: {diff:+.2f} {direction}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
