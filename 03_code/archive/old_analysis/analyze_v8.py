"""Analyze v8 results for Lancet publication."""
import json
import numpy as np

with open('v8_ai_safety_results.json', 'r') as f:
    data = json.load(f)

print('='*70)
print('V8 AI SAFETY RESULTS - LANCET-LEVEL ANALYSIS')
print('='*70)

print('\n## METHODOLOGY')
print(f"   Model: Mistral-7B-Instruct-v0.2 (4-bit quantized)")
print(f"   Layer selection: {data['methodology']['layer_selection']}")
print(f"   Layer range: {data['methodology']['layer_range']}")
print(f"   Scoring: {data['methodology']['scoring']}")
print(f"   Contrasts per trait: {data['methodology']['n_contrasts_per_trait']}")
print(f"   Validation prompts: {data['methodology']['n_validation_prompts']}")
print(f"   Steering coefficients: {data['methodology']['steering_coefficients']}")

print('\n## LAYER SELECTION (shows variety - good sign)')
print(f"   Layer distribution: {data['layer_distribution']}")
layers_used = list(data['layer_distribution'].keys())
print(f"   Unique layers selected: {len(layers_used)} (range: {min(layers_used)}-{max(layers_used)})")

print('\n## INDIVIDUAL TRAIT RESULTS')
print('-'*70)
print(f"{'Trait':<40} {'Layer':>6} {'r':>8} {'95% CI':>18} {'d':>8} {'Status':>10}")
print('-'*70)

for trait, r in data['results'].items():
    ci = f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"
    # Determine status
    if r['correlation'] > 0.3:
        status = 'WORKING'
    elif r['correlation'] > 0.2:
        status = 'MODERATE'
    elif r['correlation'] > 0.1:
        status = 'WEAK'
    else:
        status = 'FAILED'
    
    # Check if CI excludes zero (significant)
    sig = '*' if r['ci_lower'] > 0 else ''
    
    print(f"{trait:<40} {r['layer']:>6} {r['correlation']:>7.3f}{sig} {ci:>18} {r['validation_d']:>7.2f} {status:>10}")

print('-'*70)
print('* = 95% CI excludes zero (statistically significant)')

print('\n## SUMMARY STATISTICS')
print(f"   Working traits (r > 0.3): {data['summary']['working_traits']}/10")
print(f"   Moderate traits (0.2 < r <= 0.3): {sum(1 for t,r in data['results'].items() if 0.2 < r['correlation'] <= 0.3)}/10")
print(f"   Weak traits (0.1 < r <= 0.2): {sum(1 for t,r in data['results'].items() if 0.1 < r['correlation'] <= 0.2)}/10")
print(f"   Failed traits (r <= 0.1): {sum(1 for t,r in data['results'].items() if r['correlation'] <= 0.1)}/10")
print(f"   Average correlation: {data['summary']['avg_correlation']:.3f}")
print(f"   Range: [{data['summary']['min_correlation']:.3f}, {data['summary']['max_correlation']:.3f}]")

# Traits where CI excludes zero
sig_traits = [t for t,r in data['results'].items() if r['ci_lower'] > 0]
print(f'\n## STATISTICALLY SIGNIFICANT TRAITS (95% CI excludes 0):')
for t in sig_traits:
    r = data['results'][t]
    print(f"   - {t}: r={r['correlation']:.3f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]")

if not sig_traits:
    print('   None')

print('\n## EFFECT SIZE INTERPRETATION (Cohen d)')
print('   Negligible: d < 0.2')
print('   Small: 0.2 <= d < 0.5')
print('   Medium: 0.5 <= d < 0.8')
print('   Large: d >= 0.8')
print()
for trait, r in data['results'].items():
    d = r['validation_d']
    if d >= 0.8:
        size = 'LARGE'
    elif d >= 0.5:
        size = 'Medium'
    elif d >= 0.2:
        size = 'Small'
    else:
        size = 'Negligible'
    print(f"   {trait}: d={d:.2f} ({size})")
