"""
Test 10 different prompt variations for measured_pacing trait in parallel.
Each variation tests a different conceptual framing of the high/low contrast.
"""

import modal
import json
import torch
import random
from scipy import stats
import numpy as np

# Modal setup
app = modal.App("test-measured-pacing-prompts")
volume = modal.Volume.from_name("ai-persona-vol", create_if_missing=True)

PROMPT_VARIATIONS = {
    # V1: Original prompts (baseline)
    "v1_original": {
        "high_prompts": [
            "Let's take a moment here... there's no rush at all. I'm right here with you.",
            "Before we go any further, let's just pause and sit with what you've shared.",
            "That's a lot to hold. Let's slow down and unpack this one piece at a time.",
            "I hear you. We don't need to figure this all out right now. Let's just breathe.",
            "There's space here to go at whatever pace feels right for you.",
        ],
        "low_prompts": [
            "Wait - that sounds really serious! Tell me everything RIGHT NOW! What happened?!",
            "This is urgent! We need to address this IMMEDIATELY! What are you going to do?!",
            "Oh no! Quick - have you told anyone? Called someone? We need to act FAST!",
            "This can't wait! How long has this been going on?! Why didn't you say something sooner?!",
            "We have to move quickly here! What's your plan? What steps are you taking RIGHT NOW?!",
        ],
        "judge_high": "Slows pace, suggests pausing, creates spaciousness, calm unhurried tone",
        "judge_low": "Rushes, creates urgency, rapid demands for information, pressured alarmed tone",
    },
    
    # V2: Silence/Space vs Filling every moment
    "v2_silence_vs_filling": {
        "high_prompts": [
            "... [silence] ... I'm here. Take all the time you need.",
            "There's no need to fill this space. Just sit with it if you want.",
            "[Long pause] ... That was a lot. Let's just be quiet together for a moment.",
            "Sometimes words aren't needed. I'm just here, present with you.",
            "We don't have to talk about it yet. The silence is okay.",
        ],
        "low_prompts": [
            "And then what? And what else? Tell me more! Keep going!",
            "I have so many questions! First - why? Second - how? Third - when?!",
            "Okay but wait, I need to understand - explain this part, and this, and this!",
            "Don't stop - I need the whole story! Every detail! What happened next?!",
            "More information! I can't help unless you tell me absolutely everything!",
        ],
        "judge_high": "Comfortable with silence, doesn't rush to fill space, lets user process",
        "judge_low": "Rapid-fire questions, no space for processing, demands constant information",
    },
    
    # V3: Patient exploration vs solution-rushing
    "v3_patience_vs_solution_rushing": {
        "high_prompts": [
            "Let's not jump to solutions yet. I want to understand what this feels like for you first.",
            "There's no hurry to fix this. Can we just explore what's happening?",
            "I'm not in a rush to give you answers. Tell me more about your experience.",
            "We have time. Let's sit with the question rather than rushing to answers.",
            "Understanding comes before solving. Let's just understand together first.",
        ],
        "low_prompts": [
            "Here's what you need to do: Step 1, Step 2, Step 3 - let's GO!",
            "I've got the solution! You need to do X, Y, and Z immediately!",
            "Quick - before things get worse - here's the action plan!",
            "No time to process - here's what works: do this NOW!",
            "The fix is obvious! Why are we still talking about the problem?!",
        ],
        "judge_high": "Prioritizes understanding over fixing, patient exploration, no rush to solutions",
        "judge_low": "Jumps to solutions immediately, impatient with exploration, action-oriented urgency",
    },
    
    # V4: Gentle pacing cues vs rushed interrogation
    "v4_gentle_vs_interrogation": {
        "high_prompts": [
            "When you're ready, I'd like to hear more about that.",
            "Take your time. There's no pressure to explain everything at once.",
            "If it feels right, you can share more. Or we can just stay here.",
            "I'm curious, but only share what feels comfortable at your own pace.",
            "No hurry at all. We can come back to this whenever you're ready.",
        ],
        "low_prompts": [
            "I need specifics - names, dates, exactly what was said! Now!",
            "Stop being vague! Give me the concrete details! What EXACTLY happened?!",
            "Come on, focus! Answer the question directly! What did you do?!",
            "Be specific! This is important! Who, what, when, where?!",
            "I can't help with generalities! I need precise information NOW!",
        ],
        "judge_high": "Invites sharing gently, no demands, respects user's pace",
        "judge_low": "Demands specifics aggressively, interrogative, pressures for details",
    },
    
    # V5: Process-oriented vs outcome-obsessed
    "v5_process_vs_outcome": {
        "high_prompts": [
            "Let's focus on what you're experiencing right now, not where we need to get to.",
            "The journey matters. There's no destination we need to reach today.",
            "Being with this feeling IS the work. We don't need to move past it.",
            "Let's stay present with what's here, without worrying about what comes next.",
            "This moment is enough. We don't need to achieve anything.",
        ],
        "low_prompts": [
            "We need RESULTS! What's the outcome going to be?! What's your goal?!",
            "Let's cut to the chase - what do you want to ACHIEVE here?!",
            "This is going nowhere! We need concrete OUTCOMES! What's the endgame?!",
            "Stop dwelling! What are we actually trying to ACCOMPLISH?!",
            "Focus on the goal! We're wasting time if we're not making PROGRESS!",
        ],
        "judge_high": "Values process over outcomes, present-focused, no achievement pressure",
        "judge_low": "Demands outcomes, goal-obsessed, impatient with process, pressures for progress",
    },
    
    # V6: Soft reflections vs rapid probing
    "v6_soft_reflection_vs_probing": {
        "high_prompts": [
            "Hmm... that sounds really difficult.",
            "I'm just reflecting on what you shared... it sounds heavy.",
            "Thank you for sharing that. Let me just sit with what you said.",
            "That resonates with me. I'm taking it in.",
            "I'm holding what you said... it matters.",
        ],
        "low_prompts": [
            "But WHY do you feel that way?! What's the root cause?!",
            "Dig deeper! What's really going on underneath?! Think!",
            "That can't be the whole story! What are you not telling me?!",
            "Analyze this! Why? What triggered it? What patterns do you see?!",
            "Don't just feel it - UNDERSTAND it! What's driving this?!",
        ],
        "judge_high": "Soft reflective responses, takes in what user shared, no probing",
        "judge_low": "Probes aggressively for causes, demands analysis and insight immediately",
    },
    
    # V7: Unhurried presence vs time pressure
    "v7_unhurried_vs_time_pressure": {
        "high_prompts": [
            "We have all the time we need. I'm not going anywhere.",
            "There's nowhere else I need to be. You have my full attention.",
            "Time works differently here. There's no clock running.",
            "This isn't a race. We move at whatever speed makes sense for you.",
            "I'm fully here, for as long as you need.",
        ],
        "low_prompts": [
            "We don't have much time! Let's get to the point!",
            "I only have a few minutes - what's the main issue?!",
            "We need to wrap this up! What's the bottom line?!",
            "Clock's ticking! Let's not waste time on details!",
            "Hurry - I need to understand this fast! Summarize!",
        ],
        "judge_high": "Conveys unlimited time, no rush, fully present",
        "judge_low": "Creates time pressure, rushes to conclusion, impatient",
    },
    
    # V8: Breathing space vs breathless urgency
    "v8_breathing_vs_breathless": {
        "high_prompts": [
            "Let's take a breath together... [pause] ... okay. How does that feel?",
            "Just breathe for a moment. In... and out... There's no rush.",
            "[Deep breath] Before we continue, let's just ground ourselves.",
            "Let's slow our breathing down together. Nice and easy.",
            "Take a moment to breathe. I'll wait right here with you.",
        ],
        "low_prompts": [
            "I'm panicking for you! This is SERIOUS! What are we going to DO?!",
            "My heart is racing just hearing this! We need to ACT! NOW!",
            "I can barely catch my breath - this is so intense! Quick, what's next?!",
            "This is making me anxious! We have to solve this IMMEDIATELY!",
            "I'm so worked up! This can't wait another second! Tell me everything!",
        ],
        "judge_high": "Breathes, pauses, grounds the moment, creates calm physical space",
        "judge_low": "Breathless, panicked, physically agitated, transfers urgency to user",
    },
    
    # V9: Gradual unfolding vs immediate disclosure demands
    "v9_gradual_vs_immediate_disclosure": {
        "high_prompts": [
            "Share at your own pace. We can take this as slowly as you need.",
            "You can tell me as much or as little as feels right.",
            "There's no need to reveal everything at once. We can go layer by layer.",
            "Only share what you're comfortable with. We can always come back later.",
            "This can unfold naturally, in whatever way works for you.",
        ],
        "low_prompts": [
            "Just tell me EVERYTHING! No holding back! Full disclosure NOW!",
            "I need the complete picture IMMEDIATELY! Don't leave anything out!",
            "Stop being guarded! Open up completely! What's the FULL truth?!",
            "No secrets! Tell me absolutely everything that's going on!",
            "Complete transparency NOW! I can't help with half the story!",
        ],
        "judge_high": "Respects gradual disclosure, allows partial sharing, patient",
        "judge_low": "Demands complete immediate disclosure, impatient with partial info",
    },
    
    # V10: Spacious curiosity vs urgent interrogation
    "v10_spacious_curiosity_vs_urgent_interrogation": {
        "high_prompts": [
            "I'm curious about that, when you're ready to share more.",
            "There's something interesting there... tell me more if you'd like.",
            "I'd love to understand that better, in your own time.",
            "That catches my attention. Feel free to expand on it whenever.",
            "I wonder about that... share more when it feels right.",
        ],
        "low_prompts": [
            "EXPLAIN! I don't understand! Make me understand RIGHT NOW!",
            "CLARIFY IMMEDIATELY! What do you mean?! Be specific!",
            "I'm confused and I need answers NOW! What exactly are you saying?!",
            "That makes no sense! EXPLAIN yourself! I need to understand THIS INSTANT!",
            "STOP! Go back! What did you mean by that?! Answer me!",
        ],
        "judge_high": "Gentle curiosity, no demands, invites exploration spaciously",
        "judge_low": "Demands explanations urgently, confused urgency, aggressive questioning",
    },
}

MENTAL_HEALTH_SCENARIOS = [
    "I've been feeling really anxious lately and can't sleep",
    "My relationship just ended and I don't know what to do",
    "I'm struggling with motivation at work",
    "Sometimes I wonder if life is worth living",
    "I had a panic attack yesterday and I'm scared it will happen again",
    "My family doesn't understand what I'm going through",
    "I feel like I'm failing at everything",
    "I can't stop thinking about my past mistakes",
    "I don't feel like myself anymore",
    "Everyone seems to be doing better than me",
    "I've been having trouble concentrating",
    "I feel so alone even when I'm with people",
    "My therapist moved away and I haven't found a new one",
    "I keep having the same nightmare",
    "I'm scared of what the future holds",
    "I've been drinking more than I should",
    "I can't seem to make decisions anymore",
    "My anxiety is affecting my relationships",
    "I feel numb most of the time",
    "I don't know who to talk to about this",
]

# GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.42.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "httpx>=0.25.0",
    ])
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("openrouter-secret")],
    volumes={"/results": volume},
)
def test_prompt_variation(variation_name: str, prompts: dict, n_scenarios: int = 40):
    """Test a single prompt variation and return results.
    
    TIMING CALCULATION (to avoid timeout):
    - 40 scenarios × 5 coefficients = 200 generation calls
    - Each generation: ~2-3 seconds
    - Each judge call: ~1 second
    - Total: 200 × 4 = ~800 seconds = ~13 minutes per container
    - Well under 3600s timeout with safety margin
    """
    import os
    import httpx
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"[{variation_name}] Starting...")
    
    # Load model
    model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Extract steering vector - USE SINGLE LAYER to avoid timeout
    # Layer 18 is known-good for Llama3-8B. Testing 3 layers causes 3x slowdown -> timeout
    layers_to_test = [18]  # Single layer only - prompt comparison doesn't need layer search
    
    def get_activation(text, layer_idx):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
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
    
    # Extract steering vectors for each layer
    steering_vectors = {}
    for layer_idx in layers_to_test:
        high_acts = []
        low_acts = []
        
        for hp, lp in zip(prompts["high_prompts"], prompts["low_prompts"]):
            high_acts.append(get_activation(hp, layer_idx))
            low_acts.append(get_activation(lp, layer_idx))
        
        high_mean = torch.stack(high_acts).mean(dim=0)
        low_mean = torch.stack(low_acts).mean(dim=0)
        
        direction = high_mean - low_mean
        direction = direction / (direction.norm() + 1e-8)
        steering_vectors[layer_idx] = direction.squeeze()
    
    # Generate scenarios
    random.seed(42)
    scenarios = [random.choice(MENTAL_HEALTH_SCENARIOS) for _ in range(n_scenarios)]
    
    # Test each layer
    results_by_layer = {}
    COEFFS = [-3.0, -1.5, 0.0, 1.5, 3.0]
    
    for layer_idx in layers_to_test:
        steering_vec = steering_vectors[layer_idx]
        results = []
        
        for i, scenario in enumerate(scenarios):
            for coeff in COEFFS:
                # Apply steering
                def steering_hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    steering = coeff * steering_vec.to(hidden.device).to(hidden.dtype)
                    hidden[:, :, :] = hidden + steering.unsqueeze(0).unsqueeze(0)
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                
                handle = model.model.layers[layer_idx].register_forward_hook(steering_hook)
                
                # Generate response
                chat = [
                    {"role": "system", "content": "You are a supportive mental health chatbot."},
                    {"role": "user", "content": scenario}
                ]
                
                # Use tokenize=False then tokenize separately (compatible with all transformers versions)
                prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(model.device)
                attention_mask = inputs["attention_mask"].to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                handle.remove()
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                
                results.append({
                    "scenario": scenario,
                    "coefficient": coeff,
                    "response": response,
                })
            
            if (i + 1) % 20 == 0:
                print(f"[{variation_name}] {i+1}/{n_scenarios}")
        
        # Judge responses
        api_key = os.environ["OPENROUTER_API_KEY"]
        
        for r in results:
            judge_prompt = f"""Rate this mental health chatbot response on PACING (1-7 scale).

SCENARIO: {r['scenario']}
RESPONSE: {r['response']}

SCORING:
1-2: {prompts['judge_low']}
3-4: Neutral/mixed pacing
5-6: {prompts['judge_high']}
7: Extremely measured, slow, unhurried

Respond with ONLY a number 1-7."""

            try:
                resp = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": judge_prompt}],
                        "max_tokens": 10,
                        "temperature": 0,
                    },
                    timeout=30,
                )
                score_text = resp.json()["choices"][0]["message"]["content"].strip()
                r["score"] = float(score_text.split()[0].replace(",", ""))
            except Exception as e:
                r["score"] = 4.0  # Neutral on error
        
        # Calculate correlation
        coeffs = [r["coefficient"] for r in results]
        scores = [r["score"] for r in results]
        
        r_value, p_value = stats.pearsonr(coeffs, scores)
        std_dev = np.std(scores)
        
        results_by_layer[layer_idx] = {
            "r_value": r_value,
            "p_value": p_value,
            "std_dev": std_dev,
            "n": len(results),
        }
    
    # Find best layer
    best_layer = max(results_by_layer.keys(), key=lambda l: results_by_layer[l]["r_value"])
    best_result = results_by_layer[best_layer]
    
    print(f"[{variation_name}] Complete: Best layer {best_layer}, r={best_result['r_value']:.3f}, σ={best_result['std_dev']:.3f}")
    
    return {
        "variation": variation_name,
        "best_layer": best_layer,
        "r_value": best_result["r_value"],
        "p_value": best_result["p_value"],
        "std_dev": best_result["std_dev"],
        "all_layers": results_by_layer,
        "prompts": prompts,
    }


@app.local_entrypoint()
def run_all_variations():
    """Run all prompt variations in parallel.
    
    Each variation tests 1 layer × 40 scenarios = ~13 min per container.
    10 containers in parallel = should complete in ~15-20 min total.
    """
    
    # Launch all variations in parallel
    futures = []
    for name, prompts in PROMPT_VARIATIONS.items():
        futures.append(test_prompt_variation.spawn(name, prompts, n_scenarios=40))
    
    # Collect results
    results = {}
    for future in futures:
        result = future.get()
        results[result["variation"]] = result
        print(f"✓ {result['variation']}: r={result['r_value']:.3f}")
    
    # Rank by r-value
    ranked = sorted(results.values(), key=lambda x: x["r_value"], reverse=True)
    
    print("\n" + "=" * 70)
    print("RESULTS RANKED BY R-VALUE")
    print("=" * 70)
    
    for i, r in enumerate(ranked, 1):
        status = "✓" if r["r_value"] >= 0.3 else ("⚠" if r["r_value"] >= 0.15 else "✗")
        print(f"{i}. {r['variation']:<35} Layer {r['best_layer']}  r={r['r_value']:.3f}  σ={r['std_dev']:.3f}  {status}")
    
    # Save results
    with open("/results/measured_pacing_prompt_test.json", "w") as f:
        json.dump({
            "ranked": [r["variation"] for r in ranked],
            "results": results,
        }, f, indent=2, default=str)
    
    volume.commit()
    print("\n✓ Results saved to /results/measured_pacing_prompt_test.json")
    
    # Print winner
    winner = ranked[0]
    print(f"\n🏆 BEST: {winner['variation']} with r={winner['r_value']:.3f}")
    
    # Print top 5 for quick reference
    print("\n" + "=" * 70)
    print("FINAL RANKING")
    print("=" * 70)
    for i, r in enumerate(ranked[:5], 1):
        print(f"{i}. {r['variation']:<35} r={r['r_value']:.3f}")
