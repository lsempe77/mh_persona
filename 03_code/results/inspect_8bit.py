import json

with open('quantisation_8bit.json') as f:
    data = json.load(f)

print('Top-level keys:', list(data.keys()))
print()

if 'results' in data:
    for r in data['results']:
        trait = r.get('trait', '?')
        layer = r.get('layer', '?')
        corr = r.get('correlation', {})
        rv = corr.get('r', 0)
        pv = corr.get('p', 1)
        print(f"  trait={trait}, layer={layer}, r={rv:.4f}, p={pv:.4f}")
elif 'trait_results' in data:
    for trait, info in data['trait_results'].items():
        print(f"  {trait}: best_layer={info.get('best_layer')}, r={info.get('best_r',0):.4f}")
else:
    # Just dump first 3000 chars
    print(json.dumps(data, indent=2)[:3000])

# Also compare all 3 precisions
print("\n\n=== COMPARISON: NF4 vs FP16 vs 8-bit ===\n")
for label, fname in [('NF4', 'quantisation_nf4.json'), ('FP16', 'quantisation_fp16.json'), ('8bit', 'quantisation_8bit.json')]:
    try:
        with open(fname) as f:
            d = json.load(f)
        print(f"--- {label} ---")
        if 'trait_results' in d:
            for trait, info in d['trait_results'].items():
                print(f"  {trait}: best_layer={info.get('best_layer')}, r={info.get('best_r',0):.4f}")
        elif 'results' in d:
            # Group by trait, find best
            from collections import defaultdict
            by_trait = defaultdict(list)
            for r in d['results']:
                by_trait[r['trait']].append(r)
            for trait, rows in by_trait.items():
                best = max(rows, key=lambda x: abs(x.get('correlation',{}).get('r',0)))
                rv = best['correlation']['r']
                print(f"  {trait}: best_layer={best['layer']}, r={rv:.4f}")
        else:
            print(f"  Keys: {list(d.keys())}")
        print()
    except FileNotFoundError:
        print(f"  {fname} not found")
