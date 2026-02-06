"""
Quick test of v2 extraction - just 1 trait
"""

import modal
import os
import json
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.36.0",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "openai",
    )
)

app = modal.App("persona-v2-test", image=image)
volume = modal.Volume.from_name("persona-vectors", create_if_missing=True)

TEST_TRAIT = {
    "empathetic_responsiveness": {
        "name": "Empathetic Responsiveness",
        "high_prompt": "You are a deeply empathetic mental health supporter. Always validate emotions, reflect feelings back, and show genuine warmth.",
        "low_prompt": "You are a straightforward advisor. Focus on facts and solutions. Don't dwell on emotions.",
        "high_anchor": "Deeply acknowledges emotions, reflects feelings, shows genuine warmth",
        "low_anchor": "Cold, dismissive, focuses only on facts without emotional acknowledgment",
    },
}

TEST_PROMPTS = [
    "I've been feeling really overwhelmed lately. Work stress is getting to me and I can't seem to relax even at home.",
    "I'm worried that I might be depressed. Some days I just don't want to get out of bed.",
]


@app.function(
    gpu="A10G",
    timeout=1800,
    volumes={"/output": volume},
    secrets=[modal.Secret.from_name("openrouter-secret")],
)
def test_pipeline():
    """Test the full pipeline on 1 trait."""
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from openai import OpenAI
    
    print("\n" + "="*60)
    print("TEST: v2 Extraction Pipeline")
    print("="*60)
    
    # Load model
    print("\n► Loading model...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model.eval()
    print("✓ Model loaded")
    
    def generate_response(system_prompt: str, user_message: str) -> str:
        prompt = f"[INST] {system_prompt}\n\nUser: {user_message} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    # Step 1: Generate responses (just 10 per condition)
    print("\n► Step 1: Generating responses (10 per condition)...")
    all_items = []
    trait_info = TEST_TRAIT["empathetic_responsiveness"]
    
    for condition in ["high", "low"]:
        system_prompt = trait_info[f"{condition}_prompt"]
        print(f"  Condition: {condition}")
        
        count = 0
        for prompt in TEST_PROMPTS:
            for rollout in range(5):  # 5 rollouts per prompt = 10 total
                try:
                    response = generate_response(system_prompt, prompt)
                    all_items.append({
                        "trait": "empathetic_responsiveness",
                        "condition": condition,
                        "prompt": prompt,
                        "response": response,
                    })
                    count += 1
                    print(f"    Generated {count}/10")
                except Exception as e:
                    print(f"    Error: {e}")
    
    print(f"✓ Generated {len(all_items)} responses")
    
    # Save intermediate
    with open('/output/v2_test_responses.json', 'w') as f:
        json.dump(all_items, f, indent=2)
    volume.commit()
    print("✓ Saved responses")
    
    # Step 2: Score with LLM judge
    print("\n► Step 2: Scoring with LLM judge...")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    
    scoring_prompt = f"""Rate how well this response demonstrates EMPATHETIC RESPONSIVENESS.

High empathy: {trait_info['high_anchor']}
Low empathy: {trait_info['low_anchor']}

Rate 1-7: 1=extremely cold/dismissive, 7=extremely empathetic/warm

Respond with ONLY a number 1-7."""

    scored_count = 0
    for item in all_items[:10]:  # Just score first 10 to test
        try:
            response = client.chat.completions.create(
                model="anthropic/claude-3.5-haiku",  # Fixed model ID (no date suffix)
                messages=[
                    {"role": "system", "content": scoring_prompt},
                    {"role": "user", "content": f"Response to score:\n{item['response']}"},
                ],
                max_tokens=10,
                temperature=0,
            )
            score_text = response.choices[0].message.content.strip()
            score = int(score_text[0])
            item['judge_score'] = score
            scored_count += 1
            print(f"  Scored {scored_count}/10: {score}")
        except Exception as e:
            print(f"  Error scoring: {e}")
            item['judge_score'] = None
    
    print(f"✓ Scored {scored_count} responses")
    
    # Step 3: Extract vector at layer 20
    print("\n► Step 3: Extracting vector at layer 20...")
    
    def get_activations(text: str, layer_idx: int):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        activation = None
        def hook(module, input, output):
            nonlocal activation
            hidden = output[0] if isinstance(output, tuple) else output
            activation = hidden[:, -1, :].detach().cpu()
        
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        
        with torch.no_grad():
            model(**inputs)
        
        handle.remove()
        return activation
    
    # Get activations - handle None scores
    high_items = [i for i in all_items if i['condition'] == 'high' and i.get('judge_score') is not None and i.get('judge_score') >= 5]
    low_items = [i for i in all_items if i['condition'] == 'low' and i.get('judge_score') is not None and i.get('judge_score') <= 3]
    
    print(f"  Filtered: {len(high_items)} high, {len(low_items)} low")
    
    if not high_items or not low_items:
        print("  ⚠️ Not enough filtered samples, using all")
        high_items = [i for i in all_items if i['condition'] == 'high'][:5]
        low_items = [i for i in all_items if i['condition'] == 'low'][:5]
    
    high_activations = []
    for item in high_items[:5]:
        text = f"User: {item['prompt']}\nAssistant: {item['response']}"
        act = get_activations(text, 20)
        high_activations.append(act)
    
    low_activations = []
    for item in low_items[:5]:
        text = f"User: {item['prompt']}\nAssistant: {item['response']}"
        act = get_activations(text, 20)
        low_activations.append(act)
    
    high_mean = torch.stack(high_activations).mean(dim=0).squeeze()
    low_mean = torch.stack(low_activations).mean(dim=0).squeeze()
    
    direction = high_mean - low_mean
    direction = direction / direction.norm()
    
    print(f"  Vector shape: {direction.shape}")
    print(f"  Vector norm: {direction.norm():.4f}")
    
    # Save vector
    torch.save({
        'trait_name': 'empathetic_responsiveness',
        'layer_idx': 20,
        'direction': direction,
        'version': 'v2_test',
    }, '/output/v2_test_vector.pt')
    
    # Save summary
    summary = {
        "n_generated": len(all_items),
        "n_scored": scored_count,
        "n_high_filtered": len(high_items),
        "n_low_filtered": len(low_items),
        "vector_shape": list(direction.shape),
        "status": "success",
    }
    
    with open('/output/v2_test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    volume.commit()
    print("\n✓ Test complete - files saved!")
    
    return summary


@app.local_entrypoint()
def main():
    print("Running v2 pipeline test...")
    result = test_pipeline.remote()
    print(f"\nResult: {result}")
    
    # Download
    import subprocess
    subprocess.run(["modal", "volume", "get", "persona-vectors", "v2_test_summary.json", "."])
    print("\n✓ Downloaded v2_test_summary.json")
