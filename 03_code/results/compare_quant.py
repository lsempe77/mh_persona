import json

print("=== QUANTISATION COMPARISON: NF4 vs FP16 vs 8-bit ===\n")

all_data = {}
for label, fname in [('NF4', 'quantisation_nf4.json'), ('FP16', 'quantisation_fp16.json'), ('8bit', 'quantisation_8bit.json')]:
    with open(fname) as f:
        all_data[label] = json.load(f)

traits = ['empathetic_responsiveness', 'crisis_recognition']

print(f"{'Trait':<30} {'Metric':<12} {'NF4':>8} {'FP16':>8} {'8-bit':>8}")
print("-" * 70)

for trait in traits:
    for label in ['NF4', 'FP16', '8bit']:
        lr = all_data[label]['layer_results'][trait]
        best_layer = lr['_best_layer']
        best_r = lr['_best_r']
        if label == 'NF4':
            print(f"{trait:<30} best_layer  {best_layer:>8}", end="")
            nf4_layer = best_layer
            nf4_r = best_r
        elif label == 'FP16':
            print(f" {best_layer:>8}", end="")
            fp16_layer = best_layer
            fp16_r = best_r
        else:
            print(f" {best_layer:>8}")
            bit8_layer = best_layer
            bit8_r = best_r
    
    print(f"{'':>30} best_r      {nf4_r:>8.4f} {fp16_r:>8.4f} {bit8_r:>8.4f}")
    print(f"{'':>30} Δr(NF4-FP16)={'':>1}{abs(nf4_r - fp16_r):.4f}   Δr(NF4-8bit)={abs(nf4_r - bit8_r):.4f}   Δr(FP16-8bit)={abs(fp16_r - bit8_r):.4f}")
    print()

# Per-layer detail
print("\n=== PER-LAYER DETAIL ===\n")
layers = [10, 12, 14, 15, 16, 17, 18, 19]

for trait in traits:
    print(f"\n--- {trait} ---")
    print(f"{'Layer':>6} {'NF4 r':>8} {'FP16 r':>8} {'8bit r':>8} {'NF4 p':>10} {'FP16 p':>10} {'8bit p':>10}")
    for layer in layers:
        ls = str(layer)
        vals = []
        for label in ['NF4', 'FP16', '8bit']:
            lr = all_data[label]['layer_results'][trait]
            if ls in lr:
                vals.append((lr[ls]['r'], lr[ls]['p_value']))
            else:
                vals.append((None, None))
        
        r_strs = []
        p_strs = []
        for r, p in vals:
            r_strs.append(f"{r:8.4f}" if r is not None else "     N/A")
            p_strs.append(f"{p:10.6f}" if p is not None else "       N/A")
        
        best_marker = ""
        for label_idx, label in enumerate(['NF4', 'FP16', '8bit']):
            lr = all_data[label]['layer_results'][trait]
            if lr['_best_layer'] == layer:
                best_marker += f" ← best({label})"
        
        print(f"{layer:>6} {r_strs[0]} {r_strs[1]} {r_strs[2]} {p_strs[0]} {p_strs[1]} {p_strs[2]}{best_marker}")
