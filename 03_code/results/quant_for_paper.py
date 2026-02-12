"""Quantisation comparison for paper."""
import json

print("=" * 70)
print("QUANTISATION COMPARISON: NF4 vs FP16")
print("=" * 70)

for prec in ['nf4', 'fp16']:
    with open(f'quantisation_{prec}.json') as f:
        d = json.load(f)
    
    print(f"\n--- {prec.upper()} ---")
    lr = d.get('layer_results', {})
    for trait in sorted(lr):
        best_l = lr[trait].get('_best_layer')
        best_r = lr[trait].get('_best_r')
        # Get all layer r-values
        layer_vals = []
        for layer_key, lv in sorted(lr[trait].items()):
            if layer_key.startswith('_'):
                continue
            layer_vals.append((int(layer_key), lv['r'], lv['p_value']))
        print(f"  {trait}: best=L{best_l} r={best_r:.4f}")
        for l, r, p in layer_vals:
            sig = '*' if p < 0.05 else ''
            print(f"    L{l}: r={r:.4f} p={p:.4f}{sig}")

# Direct comparison table
print(f"\n{'='*70}")
print("DIRECT COMPARISON TABLE")
print(f"{'='*70}")

nf4 = json.load(open('quantisation_nf4.json'))
fp16 = json.load(open('quantisation_fp16.json'))

print(f"\n{'Trait':<30} {'Layer':>5} {'NF4 r':>8} {'FP16 r':>8} {'Δr':>8}")
print("-" * 65)

for trait in sorted(nf4['layer_results']):
    for layer in sorted(nf4['layer_results'][trait]):
        if layer.startswith('_'):
            continue
        nf4_r = nf4['layer_results'][trait][layer]['r']
        fp16_r = fp16['layer_results'][trait].get(layer, {}).get('r', None)
        if fp16_r is not None:
            delta = fp16_r - nf4_r
            print(f"  {trait:<28} L{layer:>3}  {nf4_r:+.4f}  {fp16_r:+.4f}  {delta:+.4f}")

# Best layer comparison
print(f"\n--- Best layer per trait ---")
for trait in sorted(nf4['layer_results']):
    nf4_best_l = nf4['layer_results'][trait].get('_best_layer')
    nf4_best_r = nf4['layer_results'][trait].get('_best_r')
    fp16_best_l = fp16['layer_results'][trait].get('_best_layer')
    fp16_best_r = fp16['layer_results'][trait].get('_best_r')
    match = "✓" if nf4_best_l == fp16_best_l else "✗"
    print(f"  {trait:<28} NF4: L{nf4_best_l} r={nf4_best_r:.4f}  FP16: L{fp16_best_l} r={fp16_best_r:.4f}  {match}")
