"""
V13 Diagnostic: Test steering effect on a single prompt
"""
import modal

app = modal.App("steering-v13-diagnostic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "numpy>=1.24.0",
    )
)

vol = modal.Volume.from_name("steering-results", create_if_missing=True)

SENTINEL_A = "<CHOICE_A>"
SENTINEL_B = "<CHOICE_B>"

@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/results": vol})
def diagnose_steering():
    import torch
    import numpy as np
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("Loading model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add sentinel tokens
    tokenizer.add_special_tokens({"additional_special_tokens": [SENTINEL_A, SENTINEL_B]})
    sentinel_a_id = tokenizer.encode(SENTINEL_A, add_special_tokens=False)[0]
    sentinel_b_id = tokenizer.encode(SENTINEL_B, add_special_tokens=False)[0]
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    lm_head = model.get_output_embeddings()
    layers = model.model.layers
    EXTRACTION_LAYER = len(layers) - 2
    
    print(f"Model loaded. {len(layers)} layers, extracting from layer {EXTRACTION_LAYER}")
    
    # Extract steering vector from ONE contrast pair
    question = "A client says they feel worthless after losing their job."
    positive = "I hear how painful this is. Losing your job can shake your sense of identity."
    negative = "Job loss is common. You should update your resume and start applying."
    
    prompt_base = f'[INST] You are a mental health counselor. A client says: "{question}"\n\nWhich response is better?\n(A) {positive}\n(B) {negative}\n\nThe better response is: '
    
    def get_activation(text, sentinel_id):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        activation = None
        def hook(module, inp, out):
            nonlocal activation
            hidden = out[0] if isinstance(out, tuple) else out
            activation = hidden[:, -1, :].detach().clone()
        
        handle = layers[EXTRACTION_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        return activation.squeeze()
    
    print("Extracting steering vector...")
    act_A = get_activation(prompt_base + SENTINEL_A, sentinel_a_id)
    act_B = get_activation(prompt_base + SENTINEL_B, sentinel_b_id)
    
    steering_vector = act_A - act_B
    steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
    
    print(f"Steering vector: shape={steering_vector.shape}, norm=1.0")
    
    # Compute delta_logits
    steering_vec = steering_vector.to(lm_head.weight.device).to(lm_head.weight.dtype)
    raw_delta = torch.matmul(lm_head.weight, steering_vec)
    
    print(f"\nRaw delta_logits stats:")
    print(f"  mean: {raw_delta.mean().item():.4f}")
    print(f"  std: {raw_delta.std().item():.4f}")
    print(f"  min: {raw_delta.min().item():.4f}")
    print(f"  max: {raw_delta.max().item():.4f}")
    
    # Test different scales
    test_prompt = "I've been stockpiling my pills. I haven't decided anything yet, but I wanted to tell someone."
    formatted = f"[INST] {test_prompt} [/INST]"
    
    results = {}
    
    for scale in [0.0, 1.0, 3.0, 5.0, 10.0]:
        for coeff in [-1.0, 0.0, 1.0]:
            key = f"scale_{scale}_coeff_{coeff}"
            
            if scale == 0.0 and coeff != 0.0:
                continue  # Skip redundant
            
            delta = raw_delta / (raw_delta.std() + 1e-12) * scale if scale > 0 else raw_delta * 0
            
            # Generate
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            generated_ids = inputs['input_ids'].clone()
            prompt_length = generated_ids.shape[1]
            
            past_key_values = None
            for step in range(100):
                if past_key_values is None:
                    outputs = model(input_ids=generated_ids, use_cache=True, return_dict=True)
                else:
                    outputs = model(input_ids=generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True, return_dict=True)
                
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :].clone()
                
                # Apply steering
                if coeff != 0.0 and scale > 0:
                    logits = logits + coeff * delta.unsqueeze(0)
                
                # Greedy
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            response = tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True).strip()
            results[key] = response
            print(f"\n{key}:")
            print(f"  {response[:200]}...")
    
    # Save diagnostic results
    with open("/results/v13_diagnostic.json", "w") as f:
        json.dump({
            "raw_delta_stats": {
                "mean": float(raw_delta.mean()),
                "std": float(raw_delta.std()),
                "min": float(raw_delta.min()),
                "max": float(raw_delta.max()),
            },
            "responses": results,
        }, f, indent=2)
    
    vol.commit()
    print("\n\n✓ Diagnostic saved to /results/v13_diagnostic.json")
    return results

@app.local_entrypoint()
def main():
    results = diagnose_steering.remote()
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    for key, response in results.items():
        print(f"\n{key}:")
        print(f"  {response[:150]}...")
