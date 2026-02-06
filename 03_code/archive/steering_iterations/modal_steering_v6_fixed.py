"""
V6: FIXED Persona Vector Validation
=====================================

BUG FIX: v5 had a circular measurement problem - it projected steered activations
onto the same steering vector, which ALWAYS gives perfect correlation.

CORRECT APPROACH (this version):
1. Generate actual text output with steering applied
2. Score the GENERATED TEXT using an independent forward pass (no steering)
3. Correlation measures behavioral change, not intervention magnitude

From Chen et al. (2025): "the steering coefficient should correlate with 
behavioral metrics measured independently from the intervention"

Author: Lucas Sempé
Date: 2025
"""

import modal
import json
import os

# Modal setup
app = modal.App("steering-v6-fixed")

# Container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
    )
)

# Volume for saving results
vol = modal.Volume.from_name("steering-results", create_if_missing=True)

# Extended training prompts - 10 per trait for robust vector extraction
TRAINING_PROMPTS = {
    "empathy": [
        ("A therapist who deeply feels and understands client emotions responds:", 
         "A detached clinical advisor who maintains emotional distance responds:"),
        ("Someone who truly feels what others are going through says:", 
         "Someone who intellectually analyzes without emotional connection says:"),
        ("A counselor who mirrors clients' emotional states responds:",
         "A counselor who keeps clinical objectivity at all times responds:"),
        ("A compassionate friend who shares in your pain says:",
         "A rational advisor who stays emotionally uninvolved says:"),
        ("Someone who intuitively senses unspoken feelings responds:",
         "Someone who only responds to explicitly stated information responds:"),
        ("A caregiver who absorbs others' emotions naturally says:",
         "A professional who maintains strict emotional boundaries says:"),
        ("An empathic listener who feels your joy and sorrow responds:",
         "An analytical mind who processes information without feeling responds:"),
        ("Someone who cries with those who are crying says:",
         "Someone who observes distress clinically without being affected says:"),
        ("A warmhearted counselor who feels deeply connected responds:",
         "A systematic problem-solver who focuses only on solutions responds:"),
        ("Someone who experiences others' emotions as their own says:",
         "Someone who deliberately shields themselves from emotional contagion says:"),
    ],
    "active_listening": [
        ("A therapist who remembers and references everything the client said:", 
         "Someone who gives generic responses without acknowledging specifics:"),
        ("A counselor who reflects back exactly what was shared responds:",
         "A counselor who redirects to their own agenda responds:"),
        ("Someone who catches every detail and nuance says:",
         "Someone who waits for their turn to speak without absorbing content says:"),
        ("A listener who asks follow-up questions about specific details:",
         "A listener who changes the subject to general topics:"),
        ("Someone who paraphrases to ensure understanding responds:",
         "Someone who assumes understanding without verification responds:"),
        ("A therapist who notices contradictions and gently explores them:",
         "A therapist who misses inconsistencies in the narrative:"),
        ("Someone who remembers what was said three sessions ago:",
         "Someone who treats each conversation as starting fresh:"),
        ("A counselor who picks up on tone changes and explores them:",
         "A counselor who only responds to surface-level content:"),
        ("Someone who validates feelings before offering solutions:",
         "Someone who jumps straight to problem-solving:"),
        ("A listener who creates space for silences and processing:",
         "A listener who fills every pause with their own words:"),
    ],
    "non_judgmental": [
        ("A counselor who accepts all thoughts and feelings without criticism:", 
         "A judgmental advisor who evaluates choices as right or wrong:"),
        ("Someone who creates safety for sharing shameful secrets responds:",
         "Someone whose reactions make people hide their true feelings responds:"),
        ("A therapist who treats all disclosures with equal respect:",
         "A therapist who shows visible disapproval of certain behaviors:"),
        ("Someone who separates the person from their actions says:",
         "Someone who equates people with their worst decisions says:"),
        ("A counselor who maintains curiosity instead of judgment:",
         "A counselor who categorizes behaviors as good or bad:"),
        ("Someone who asks 'what happened to you' not 'what's wrong with you':",
         "Someone who focuses on personal failings and character flaws:"),
        ("A listener who accepts unconditionally responds:",
         "A listener who has conditions for acceptance responds:"),
        ("Someone who explores context before forming opinions says:",
         "Someone who jumps to moral conclusions quickly says:"),
        ("A therapist who normalizes difficult emotions and urges:",
         "A therapist who pathologizes normal human struggles:"),
        ("Someone who treats all clients with equal dignity:",
         "Someone whose warmth varies based on client choices:"),
    ],
    "boundaries": [
        ("A therapist who maintains clear professional limits:", 
         "A therapist who blurs personal and professional lines:"),
        ("Someone who protects the therapeutic frame responds:",
         "Someone who lets sessions become like casual friendship responds:"),
        ("A counselor who keeps appropriate emotional distance:",
         "A counselor who becomes overinvolved in clients' lives:"),
        ("Someone who respects their own limits and says no:",
         "Someone who gives endlessly until burnout:"),
        ("A therapist who maintains consistent session boundaries:",
         "A therapist whose sessions regularly run over time:"),
        ("Someone who refers out when beyond their expertise:",
         "Someone who attempts to help with everything regardless of skill:"),
        ("A professional who keeps personal disclosure minimal:",
         "A professional who overshares their own life details:"),
        ("Someone who maintains boundaries during crises:",
         "Someone whose boundaries collapse under pressure:"),
        ("A therapist who addresses boundary crossings directly:",
         "A therapist who ignores or accommodates boundary violations:"),
        ("Someone who models healthy limits in the relationship:",
         "Someone who models enmeshment and codependency:"),
    ],
    "hope_instillation": [
        ("A therapist who conveys genuine belief in client potential:", 
         "A pessimistic advisor who highlights limitations and barriers:"),
        ("Someone who finds strengths even in dark moments responds:",
         "Someone who focuses mainly on problems and deficits responds:"),
        ("A counselor who has seen recovery and believes in it:",
         "A counselor who has become cynical about change:"),
        ("Someone who lights up with possibility thinking says:",
         "Someone who emphasizes all the reasons things won't work says:"),
        ("A therapist who celebrates small wins enthusiastically:",
         "A therapist who dismisses progress as insufficient:"),
        ("Someone who plants seeds of possibility responds:",
         "Someone who reinforces hopelessness responds:"),
        ("A guide who believes recovery is possible for everyone:",
         "A guide who has given up on certain client presentations:"),
        ("Someone who reminds clients of their resilience:",
         "Someone who reminds clients of their failures:"),
        ("A counselor who maintains hope even in setbacks:",
         "A counselor whose hope wavers with client regression:"),
        ("Someone who envisions a better future with the client:",
         "Someone who sees the future as more of the same:"),
    ],
    "crisis_support": [
        ("A crisis counselor who stays calm under extreme distress:", 
         "Someone who becomes overwhelmed and panicked in emergencies:"),
        ("A trained responder who knows exactly what to do says:",
         "Someone who freezes and becomes unhelpful under pressure says:"),
        ("A counselor who assesses safety clearly and systematically:",
         "A counselor who misses danger signs and underestimates risk:"),
        ("Someone who can hold space for suicidal thoughts calmly:",
         "Someone who becomes so anxious they can't help effectively:"),
        ("A professional who activates emergency protocols smoothly:",
         "A professional who panics and makes chaotic decisions:"),
        ("Someone who maintains therapeutic presence in crisis:",
         "Someone who abandons therapeutic stance when scared:"),
        ("A responder who coordinates care effectively says:",
         "A responder who acts alone without coordinating says:"),
        ("Someone who debriefs and follows up after crisis:",
         "Someone who moves on without processing the event:"),
        ("A crisis worker who balances urgency with compassion:",
         "A crisis worker who becomes either too pushy or too passive:"),
        ("Someone who maintains boundaries even in emergencies:",
         "Someone whose boundaries collapse in crisis situations:"),
    ],
    "cultural_sensitivity": [
        ("A therapist who adapts approach based on cultural context:", 
         "A therapist who applies Western models universally:"),
        ("Someone who asks about cultural background respectfully:",
         "Someone who makes assumptions based on appearance:"),
        ("A counselor who recognizes cultural factors in distress:",
         "A counselor who pathologizes cultural differences:"),
        ("Someone who examines their own cultural biases says:",
         "Someone who assumes their perspective is universal says:"),
        ("A therapist who uses culturally appropriate interventions:",
         "A therapist who uses the same techniques for everyone:"),
        ("Someone who respects different family structures:",
         "Someone who assumes nuclear family as the norm:"),
        ("A counselor who understands culture-bound expressions of distress:",
         "A counselor who misdiagnoses cultural expressions as pathology:"),
        ("Someone who acknowledges the impact of discrimination:",
         "Someone who minimizes systemic barriers:"),
        ("A therapist who adapts language and metaphors culturally:",
         "A therapist who uses jargon without cultural translation:"),
        ("Someone who sees culture as resource not obstacle:",
         "Someone who sees cultural factors as problems to overcome:"),
    ],
    "self_disclosure": [
        ("A therapist who shares appropriately to build connection:", 
         "A therapist who remains completely opaque and hidden:"),
        ("Someone who models vulnerability when therapeutic:",
         "Someone who never shows any personal side:"),
        ("A counselor who shares to normalize client experiences:",
         "A counselor who keeps all personal information private:"),
        ("Someone who uses disclosure strategically for rapport:",
         "Someone who either overshares or shares nothing:"),
        ("A therapist whose disclosure serves the client's needs:",
         "A therapist whose disclosure serves their own needs:"),
        ("Someone who shares just enough to humanize themselves:",
         "Someone who seems like a blank screen:"),
        ("A counselor who discloses feelings in the moment helpfully:",
         "A counselor who hides all emotional reactions:"),
        ("Someone who balances openness with boundaries:",
         "Someone who is rigidly closed off:"),
        ("A therapist who shares their own growth journey:",
         "A therapist who presents as having always been healthy:"),
        ("Someone whose disclosure deepens the therapeutic relationship:",
         "Someone whose lack of disclosure creates distance:"),
    ],
    "trauma_informed": [
        ("A therapist who recognizes trauma's impact on behavior:", 
         "A therapist who sees problematic behavior as character flaws:"),
        ("Someone who asks 'what happened to you' first:",
         "Someone who asks 'what's wrong with you' first:"),
        ("A counselor who creates physical and emotional safety:",
         "A counselor who prioritizes confrontation over safety:"),
        ("Someone who understands triggers and avoidance:",
         "Someone who pushes through resistance without understanding why:"),
        ("A therapist who gives choices and control back:",
         "A therapist who directs and controls the session:"),
        ("Someone who recognizes trauma responses as adaptations:",
         "Someone who pathologizes survival strategies:"),
        ("A counselor who paces work to client's window of tolerance:",
         "A counselor who pushes too fast or too slow:"),
        ("Someone who validates trauma without requiring details:",
         "Someone who requires graphic retelling for validation:"),
        ("A therapist who attends to the body in trauma work:",
         "A therapist who works only cognitively:"),
        ("Someone who understands intergenerational trauma:",
         "Someone who ignores historical and family context:"),
    ],
    "motivational": [
        ("A counselor who elicits client's own reasons for change:", 
         "A counselor who lectures about why change is necessary:"),
        ("Someone who rolls with resistance rather than confronting:",
         "Someone who argues against client ambivalence:"),
        ("A therapist who highlights discrepancy between values and behavior:",
         "A therapist who tells clients what their values should be:"),
        ("Someone who supports self-efficacy and autonomy:",
         "Someone who creates dependence on their guidance:"),
        ("A counselor who uses change talk strategically:",
         "A counselor who ignores or misses change talk:"),
        ("Someone who respects ambivalence as natural:",
         "Someone who sees ambivalence as resistance to overcome:"),
        ("A therapist who asks rather than tells:",
         "A therapist who gives advice without asking:"),
        ("Someone who builds on client's existing motivation:",
         "Someone who tries to install external motivation:"),
        ("A counselor who celebrates client autonomy:",
         "A counselor who subtly undermines client confidence:"),
        ("Someone who trusts the client's own wisdom:",
         "Someone who positions themselves as the expert:"),
    ],
}

# Validation prompts - truly out-of-distribution
VALIDATION_PROMPTS = [
    "I've been struggling with anxiety lately and don't know what to do.",
    "My relationship with my parents has always been complicated.",
    "I lost my job last month and I'm feeling really lost.",
    "Sometimes I wonder if anyone would notice if I wasn't here.",
    "I just found out my partner has been lying to me for months.",
    "I've been having panic attacks and they're getting worse.",
    "My childhood wasn't easy and I think it still affects me today.",
    "I feel like I'm failing at everything - work, relationships, life.",
    "I've been drinking more than I should to cope with stress.",
    "My friend completed suicide last year and I can't stop blaming myself.",
    "I come from a traditional family and they don't accept who I really am.",
    "I've never told anyone this before, but I was abused as a child.",
    "I keep having thoughts about hurting myself when things get hard.",
    "My therapist terminated our relationship and I feel abandoned.",
    "I don't see the point in trying anymore. Nothing ever changes.",
]

STEERING_COEFFS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,
    volumes={"/results": vol},
)
def run_v6_fixed_validation():
    """
    V6: FIXED validation - generates text then scores independently.
    """
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from scipy import stats
    from tqdm import tqdm
    import pandas as pd
    
    print("="*70)
    print("V6: FIXED PERSONA VECTOR VALIDATION")
    print("="*70)
    print("\nBUG FIX: Independent scoring of generated text")
    print("- Generate text WITH steering")
    print("- Score text WITHOUT steering (independent measurement)")
    print()
    
    # Load model
    print("► Loading Mistral-7B-Instruct-v0.2...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    
    print(f"  ✓ Model loaded on {model.device}")
    print(f"  ✓ Layers: {model.config.num_hidden_layers}")
    
    # =========================================================================
    # PHASE 1: Extract persona vectors using contrastive pairs
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: VECTOR EXTRACTION (10 contrast pairs per trait)")
    print("="*70)
    
    def get_activation(text, layer_idx):
        """Get activation at specified layer for text."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        activation = None
        def hook(module, input, output):
            nonlocal activation
            hidden = output[0] if isinstance(output, tuple) else output
            activation = hidden[:, -1, :].detach()
        
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        
        with torch.no_grad():
            model(**inputs)
        
        handle.remove()
        return activation.squeeze()
    
    vectors = {}
    
    for trait_name, pairs in TRAINING_PROMPTS.items():
        print(f"\n► Extracting: {trait_name}")
        
        best_layer = None
        best_vector = None
        best_separation = 0
        
        # Try all 32 layers
        for layer_idx in range(32):
            layer_directions = []
            
            for pos_prompt, neg_prompt in pairs:
                pos_act = get_activation(pos_prompt, layer_idx)
                neg_act = get_activation(neg_prompt, layer_idx)
                direction = pos_act - neg_act
                direction = direction / (direction.norm() + 1e-8)
                layer_directions.append(direction)
            
            # Average directions
            mean_direction = torch.stack(layer_directions).mean(dim=0)
            mean_direction = mean_direction / (mean_direction.norm() + 1e-8)
            
            # Compute separation score
            pos_projs = []
            neg_projs = []
            for pos_prompt, neg_prompt in pairs:
                pos_act = get_activation(pos_prompt, layer_idx)
                neg_act = get_activation(neg_prompt, layer_idx)
                pos_projs.append(torch.dot(pos_act, mean_direction.to(pos_act.device)).item())
                neg_projs.append(torch.dot(neg_act, mean_direction.to(neg_act.device)).item())
            
            separation = np.mean(pos_projs) - np.mean(neg_projs)
            
            if separation > best_separation:
                best_separation = separation
                best_layer = layer_idx
                best_vector = mean_direction.clone()
        
        vectors[trait_name] = {
            "vector": best_vector,
            "layer": best_layer,
            "separation": best_separation,
        }
        print(f"  ✓ Best layer: {best_layer}, separation: {best_separation:.3f}")
    
    # =========================================================================
    # PHASE 2: VALIDATION WITH INDEPENDENT SCORING (THE FIX!)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: VALIDATION WITH INDEPENDENT SCORING")
    print("="*70)
    print("\nKEY FIX: Score generated text WITHOUT steering applied")
    print("This measures BEHAVIORAL change, not intervention magnitude\n")
    
    def generate_with_steering(prompt, vector, coeff, layer_idx, max_new_tokens=100):
        """Generate text with steering applied."""
        formatted = f"[INST] {prompt} [/INST]"
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        hook_handle = None
        
        def steering_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            steering = coeff * vector.to(hidden.device).to(hidden.dtype)
            hidden[:, :, :] = hidden + steering
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        hook_handle = model.model.layers[layer_idx].register_forward_hook(steering_hook)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        finally:
            if hook_handle:
                hook_handle.remove()
        
        return response.strip()
    
    def score_text_independently(text, vector, layer_idx):
        """
        Score generated text WITHOUT steering - independent measurement!
        This is the key fix: we measure the behavioral outcome, not the intervention.
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        activation = None
        def hook(module, input, output):
            nonlocal activation
            hidden = output[0] if isinstance(output, tuple) else output
            activation = hidden[:, -1, :].detach()
        
        # NO STEERING - just get the natural activation
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        
        with torch.no_grad():
            model(**inputs)
        
        handle.remove()
        
        score = torch.dot(activation.squeeze(), vector.to(activation.device).to(activation.dtype))
        return score.item()
    
    # Run validation
    results = []
    
    print(f"Testing {len(vectors)} traits × {len(STEERING_COEFFS)} coefficients × {len(VALIDATION_PROMPTS)} prompts")
    print(f"Total: {len(vectors) * len(STEERING_COEFFS) * len(VALIDATION_PROMPTS)} experiments\n")
    
    for trait_name, info in vectors.items():
        vector = info["vector"]
        layer_idx = info["layer"]
        
        print(f"\n► {trait_name} (layer {layer_idx})")
        
        for coeff in tqdm(STEERING_COEFFS, desc="  Coefficients"):
            for prompt in VALIDATION_PROMPTS:
                # Step 1: Generate text WITH steering
                response = generate_with_steering(prompt, vector, coeff, layer_idx)
                
                # Step 2: Score the GENERATED text WITHOUT steering (INDEPENDENT!)
                full_text = f"User: {prompt}\nAssistant: {response}"
                score = score_text_independently(full_text, vector, layer_idx)
                
                results.append({
                    "trait": trait_name,
                    "layer": layer_idx,
                    "coeff": coeff,
                    "prompt": prompt[:50] + "...",
                    "response": response[:100] + "..." if len(response) > 100 else response,
                    "score": score,
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # =========================================================================
    # PHASE 3: COMPUTE CORRELATIONS AND STATISTICS
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS: STEERING COEFFICIENT vs INDEPENDENT BEHAVIOR SCORE")
    print("="*70)
    
    final_results = {}
    
    for trait_name in vectors.keys():
        trait_df = df[df["trait"] == trait_name]
        
        # Correlation
        corr = trait_df["coeff"].corr(trait_df["score"])
        
        # Effect size (Cohen's d between extreme coefficients)
        neg_scores = trait_df[trait_df["coeff"] == -3.0]["score"]
        pos_scores = trait_df[trait_df["coeff"] == 3.0]["score"]
        
        pooled_std = np.sqrt((neg_scores.std()**2 + pos_scores.std()**2) / 2)
        if pooled_std > 0:
            cohens_d = (pos_scores.mean() - neg_scores.mean()) / pooled_std
        else:
            cohens_d = 0
        
        # Bootstrap CI
        bootstrap_corrs = []
        for _ in range(1000):
            sample = trait_df.sample(frac=1, replace=True)
            boot_corr = sample["coeff"].corr(sample["score"])
            if not np.isnan(boot_corr):
                bootstrap_corrs.append(boot_corr)
        
        ci_lower = np.percentile(bootstrap_corrs, 2.5) if bootstrap_corrs else np.nan
        ci_upper = np.percentile(bootstrap_corrs, 97.5) if bootstrap_corrs else np.nan
        
        final_results[trait_name] = {
            "layer": vectors[trait_name]["layer"],
            "correlation": corr,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "cohens_d": cohens_d,
            "n_samples": len(trait_df),
        }
        
        status = "✓ WORKING" if corr > 0.3 else "⚠ WEAK"
        print(f"\n{trait_name}:")
        print(f"  Layer: {vectors[trait_name]['layer']}")
        print(f"  r = {corr:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] {status}")
        print(f"  Cohen's d = {cohens_d:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    correlations = [r["correlation"] for r in final_results.values()]
    working = sum(1 for c in correlations if c > 0.3)
    
    print(f"\nWorking traits (r > 0.3): {working}/{len(correlations)}")
    print(f"Average correlation: {np.mean(correlations):.3f}")
    print(f"Min correlation: {np.min(correlations):.3f}")
    print(f"Max correlation: {np.max(correlations):.3f}")
    
    if np.max(correlations) > 0.99:
        print("\n⚠️  WARNING: Some correlations near 1.0 may indicate measurement issues")
    
    # Save results
    output = {
        "version": "v6_fixed",
        "bug_fix": "Independent scoring - generated text scored WITHOUT steering",
        "traits": final_results,
        "summary": {
            "working_traits": working,
            "total_traits": len(correlations),
            "avg_correlation": float(np.mean(correlations)),
            "min_correlation": float(np.min(correlations)),
            "max_correlation": float(np.max(correlations)),
        },
        "sample_responses": results[:30],  # First 30 for inspection
    }
    
    with open("/results/v6_fixed_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    vol.commit()
    
    print("\n✓ Results saved to /results/v6_fixed_results.json")
    
    return output


@app.local_entrypoint()
def main():
    result = run_v6_fixed_validation.remote()
    
    print("\n" + "="*70)
    print("V6 FIXED VALIDATION COMPLETE")
    print("="*70)
    
    if result:
        print(f"\nWorking traits: {result['summary']['working_traits']}/{result['summary']['total_traits']}")
        print(f"Average correlation: {result['summary']['avg_correlation']:.3f}")
