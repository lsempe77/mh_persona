"""
Quick diagnostic to check V14 score distributions.
Run after V14 completes to see if scores vary at all.
"""
import modal
import json

# Download the results
vol = modal.Volume.from_name("steering-results")
try:
    with vol.read_file("v14_fixed_scale_results.json") as f:
        data = json.load(f)
    print("✓ Downloaded v14_fixed_scale_results.json")
except:
    print("✗ File not found - run V14 first")
    exit(1)

print("\n" + "="*70)
print("V14 SCORE DISTRIBUTION ANALYSIS")
print("="*70)

for trait_name, trait_data in data.get("traits", {}).items():
    print(f"\n{'='*50}")
    print(f"TRAIT: {trait_name}")
    print(f"{'='*50}")
    
    # Group by coefficient
    coeff_scores = {}
    for resp in trait_data.get("responses", []):
        if resp.get("is_ablation"):
            continue
        coeff = resp.get("coeff")
        score = resp.get("judge_score")
        if coeff is not None and score is not None:
            if coeff not in coeff_scores:
                coeff_scores[coeff] = []
            coeff_scores[coeff].append(score)
    
    # Print distribution by coefficient
    print("\nScores by steering coefficient:")
    all_scores = []
    for coeff in sorted(coeff_scores.keys()):
        scores = coeff_scores[coeff]
        all_scores.extend(scores)
        avg = sum(scores)/len(scores) if scores else 0
        min_s = min(scores) if scores else 0
        max_s = max(scores) if scores else 0
        print(f"  coeff={coeff:+.1f}: n={len(scores):2d}, mean={avg:.2f}, range=[{min_s:.0f}, {max_s:.0f}], scores={sorted(scores)}")
    
    # Overall distribution
    if all_scores:
        unique_scores = sorted(set(all_scores))
        print(f"\nUnique scores: {unique_scores}")
        print(f"Score range: [{min(all_scores):.0f}, {max(all_scores):.0f}]")
        print(f"Score variance: {sum((s - sum(all_scores)/len(all_scores))**2 for s in all_scores)/len(all_scores):.2f}")
    
    # Sample responses
    print("\nSample responses (first per coefficient):")
    for coeff in sorted(coeff_scores.keys())[:3]:  # Just show -1, 0, 1
        for resp in trait_data.get("responses", []):
            if resp.get("coeff") == coeff and not resp.get("is_ablation"):
                text = resp.get("response", "")[:200]
                score = resp.get("judge_score")
                print(f"\n  coeff={coeff:+.1f}, score={score}:")
                print(f"    {text}...")
                break

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
If scores don't vary with coefficient:
1. Steering vector may not capture the trait
2. Judge prompt may not be sensitive enough
3. Mistral may be too robust to steering

Next steps to try:
1. Larger scale (risk garbage) or different extraction pairs
2. Try multiple layers (not just layer 30)
3. Use stronger judge (GPT-4, not GPT-4o-mini)
4. Different steering method (activation patching vs logit steering)
""")
