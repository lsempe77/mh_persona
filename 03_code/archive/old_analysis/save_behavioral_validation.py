"""Save final behavioral validation results to JSON."""
import json
import numpy as np
from scipy import stats

def bootstrap_ci(x, y, n_bootstrap=1000, ci=95):
    rs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x), len(x), replace=True)
        try:
            r, _ = stats.pearsonr(x[idx], y[idx])
            if not np.isnan(r):
                rs.append(r)
        except:
            pass
    if not rs:
        return 0.0, 0.0
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))

results = {
    'summary': 'Behavioral Validation Results',
    'date': '2026-02-05',
    'methodology': {
        'n_scenarios': 60,
        'n_turns': 5,
        'judge_model': 'openai/gpt-4o-mini',
        'success_criterion': 'r > 0.3 AND p < 0.05 for 6+ traits'
    },
    'models': {}
}

for model in ['llama3', 'mistral', 'qwen2']:
    with open(f'behavioral_validation_checkpoint_{model}.json') as f:
        d = json.load(f)
    
    traits = list(d['results'][0]['trait_results'].keys())
    model_results = {
        'n_scenarios': len(d['results']),
        'n_traits': len(traits),
        'trait_correlations': {},
        'validated_traits': []
    }
    
    for trait in traits:
        act_drifts = [s['trait_results'][trait]['activation_drift'] for s in d['results']]
        beh_drifts = [s['trait_results'][trait]['behavioral_drift'] for s in d['results']]
        
        try:
            r, p = stats.pearsonr(act_drifts, beh_drifts)
            ci_lower, ci_upper = bootstrap_ci(np.array(act_drifts), np.array(beh_drifts))
            validated = bool(r > 0.3 and p < 0.05)
            if validated:
                model_results['validated_traits'].append(trait)
            model_results['trait_correlations'][trait] = {
                'r': float(round(r, 4)),
                'p': float(round(p, 4)),
                'ci_lower': float(round(ci_lower, 4)),
                'ci_upper': float(round(ci_upper, 4)),
                'validated': validated,
                'n': len(act_drifts)
            }
        except Exception as e:
            model_results['trait_correlations'][trait] = {'error': str(e)}
    
    model_results['validated_count'] = len(model_results['validated_traits'])
    model_results['success'] = model_results['validated_count'] >= 6
    results['models'][model] = model_results

# Cross-model summary
results['cross_model_summary'] = {
    'llama3_validated': results['models']['llama3']['validated_count'],
    'mistral_validated': results['models']['mistral']['validated_count'],
    'qwen2_validated': results['models']['qwen2']['validated_count'],
    'success_threshold': '6/9 traits',
    'overall_success': False,
    'key_finding': 'Only empathetic_responsiveness validated in 2/3 models'
}

with open('behavioral_validation_final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Saved behavioral_validation_final_results.json')
print()
print('Summary:')
print(f"  Llama3: {results['models']['llama3']['validated_count']}/9 validated {results['models']['llama3']['validated_traits']}")
print(f"  Mistral: {results['models']['mistral']['validated_count']}/9 validated {results['models']['mistral']['validated_traits']}")
print(f"  Qwen2: {results['models']['qwen2']['validated_count']}/9 validated {results['models']['qwen2']['validated_traits']}")
