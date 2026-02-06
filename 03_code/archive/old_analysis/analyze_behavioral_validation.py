"""Analyze behavioral validation checkpoint files."""
import json
import numpy as np
from scipy import stats
import sys

def bootstrap_ci(x, y, n_bootstrap=1000, ci=95):
    """Compute bootstrap confidence interval for Pearson r."""
    rs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x), len(x), replace=True)
        r, _ = stats.pearsonr(x[idx], y[idx])
        if not np.isnan(r):
            rs.append(r)
    lower = np.percentile(rs, (100 - ci) / 2)
    upper = np.percentile(rs, 100 - (100 - ci) / 2)
    return lower, upper

def analyze_checkpoint(model_key):
    """Analyze a behavioral validation checkpoint file."""
    filepath = f"behavioral_validation_checkpoint_{model_key}.json"
    
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"MODEL: {data['model'].upper()}")
    print(f"{'='*60}")
    print(f"Scenarios completed: {len(data['results'])}")
    
    # Get traits dynamically from first result
    first_result = data['results'][0]
    traits = list(first_result['trait_results'].keys())
    
    print(f"Traits tested: {len(traits)}")
    
    print("\n--- TRAIT CORRELATIONS (Activation Drift vs Behavioral Drift) ---")
    results = {}
    validated = 0
    
    for trait in traits:
        act_drifts = []
        beh_drifts = []
        
        for s in data['results']:
            if 'trait_results' in s and trait in s['trait_results']:
                tr = s['trait_results'][trait]
                if 'activation_drift' in tr and 'behavioral_drift' in tr:
                    act_drifts.append(tr['activation_drift'])
                    beh_drifts.append(tr['behavioral_drift'])
        
        if len(act_drifts) >= 10:
            act_drifts = np.array(act_drifts)
            beh_drifts = np.array(beh_drifts)
            r, p = stats.pearsonr(act_drifts, beh_drifts)
            ci_lower, ci_upper = bootstrap_ci(act_drifts, beh_drifts)
            is_valid = r > 0.3 and ci_lower > 0
            if is_valid:
                validated += 1
            status = '✓ VALIDATED' if is_valid else ('⚠ WEAK' if r > 0.15 else '✗ FAILED')
            print(f"  {trait}: r={r:.3f}, p={p:.4f}, CI=[{ci_lower:.3f}, {ci_upper:.3f}] {status}")
            results[trait] = {
                'r': float(round(r, 4)), 
                'p': float(round(p, 4)), 
                'ci_lower': float(round(ci_lower, 4)), 
                'ci_upper': float(round(ci_upper, 4)), 
                'validated': bool(is_valid),
                'n': int(len(act_drifts))
            }
        else:
            print(f"  {trait}: Insufficient data (n={len(act_drifts)})")
            results[trait] = {'error': 'insufficient_data', 'n': len(act_drifts)}
    
    print(f"\n--- SUMMARY ---")
    n_traits = len(traits)
    print(f"Validated traits (r>0.3, CI>0): {validated}/{n_traits}")
    success = validated >= (n_traits * 2 // 3)  # 2/3 of traits need to validate
    print(f"SUCCESS CRITERION MET: {'YES ✓' if success else 'NO ✗'} (need {n_traits * 2 // 3}+ traits)")
    
    return {
        'model': model_key,
        'n_scenarios': len(data['results']),
        'n_traits': n_traits,
        'trait_correlations': results,
        'validated_count': validated,
        'success': bool(success)
    }

if __name__ == "__main__":
    models = ['llama3', 'mistral', 'qwen2']
    all_results = {}
    
    for model in models:
        try:
            all_results[model] = analyze_checkpoint(model)
        except FileNotFoundError:
            print(f"\n[SKIP] {model} - checkpoint file not found")
        except Exception as e:
            print(f"\n[ERROR] {model} - {e}")
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL BEHAVIORAL VALIDATION SUMMARY")
    print("="*60)
    
    for model, result in all_results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"  {model.upper()}: {result['validated_count']}/9 traits validated {status}")
    
    # Save combined results
    with open('behavioral_validation_combined_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: behavioral_validation_combined_results.json")
