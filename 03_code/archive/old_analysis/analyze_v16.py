import json

d = json.load(open('v16_per_trait_layers_results.json'))

print("V16 Per-Trait Layer Results Analysis")
print("=" * 60)

for trait_name, trait_data in d['traits'].items():
    analysis = trait_data.get('analysis', {})
    r = analysis.get('effective_correlation', 'N/A')
    p = analysis.get('p_value', 'N/A')
    print(f"\n{trait_name}:")
    print(f"  r={r}, p={p}")
    
    # Get mean scores by coefficient
    scores_by_coeff = {}
    for r_data in trait_data.get('data', []):
        coeff = r_data.get('coeff')
        if coeff != 'ablation':
            scores_by_coeff.setdefault(coeff, []).append(r_data['judge_score'])
    
    for c, s in sorted(scores_by_coeff.items()):
        mean = sum(s)/len(s) if s else 0
        print(f"    coeff {c:+.1f}: mean={mean:.2f}, n={len(s)}")

print("\n" + "=" * 60)
print("Summary: Working traits (r > 0.3, p < 0.05):")
for t, td in d['traits'].items():
    a = td.get('analysis', {})
    r = a.get('effective_correlation')
    p = a.get('p_value')
    if r is not None and p is not None and r > 0.3 and p < 0.05:
        print(f"  ✓ {t}: r={r:.3f}, p={p:.3f}")
print("(none found)" if not any(
    td.get('analysis', {}).get('effective_correlation', 0) > 0.3 and 
    td.get('analysis', {}).get('p_value', 1) < 0.05 
    for td in d['traits'].values()
) else "")
