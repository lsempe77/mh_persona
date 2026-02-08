"""
Cross-Model Analysis: Template Vectors vs Contrastive Probing (Solution B)

Compares steering validation across 3 models × 2 methods:
  - Template vectors: hand-crafted contrast prompts (Phase 2a)
  - Contrastive probing: data-driven vectors from model's own responses (Phase 2c)

Key finding: Contrastive probing achieves universal or near-universal coverage,
demonstrating that steering is architecture-general when vectors are derived
from each model's own representational geometry.
"""

import json
import os

# ── Load results ──────────────────────────────────────────────────────────────

models = ['llama3', 'mistral', 'qwen2']

# Template-based results (Phase 2a)
template = {}
for m in models:
    path = f'trait_layer_matrix_{m}.json'
    if os.path.exists(path):
        with open(path) as f:
            template[m] = json.load(f)

# Probe-based results (Phase 2c — Solution B)
probe = {}
for m in ['qwen2', 'mistral']:  # Llama3 doesn't need probing (8/8 with templates)
    path = f'trait_layer_matrix_probe_{m}.json'
    if os.path.exists(path):
        with open(path) as f:
            probe[m] = json.load(f)

# Probe diagnostics (sample counts, accuracy)
probe_diag = {}
for m in ['qwen2', 'mistral']:
    path = f'probe_diagnostics_{m}.json'
    if os.path.exists(path):
        with open(path) as f:
            probe_diag[m] = json.load(f)

traits = list(template['llama3']['traits'].keys())

# ── Section 1: Main Results Table ─────────────────────────────────────────────

print('=' * 130)
print('CROSS-MODEL STEERING VALIDATION: TEMPLATE vs CONTRASTIVE PROBING')
print('=' * 130)
print()
print(f'{"Trait":<40} {"Llama3":>14} {"Qwen2 (T)":>12} {"Qwen2 (P)":>12} {"Mistral (T)":>12} {"Mistral (P)":>12}')
print(f'{"":40} {"(template)":>14} {"template":>12} {"probe":>12} {"template":>12} {"probe":>12}')
print('-' * 130)

for t in traits:
    # Llama3 template
    rl = template['llama3']['traits'][t]['best_r']
    ll = template['llama3']['traits'][t]['best_layer']
    sl = '✓' if template['llama3']['traits'][t]['validated'] else '✗'

    # Qwen2 template
    rqt = template['qwen2']['traits'][t]['best_r']
    sqt = '✓' if template['qwen2']['traits'][t]['validated'] else '✗'

    # Qwen2 probe
    rqp = probe['qwen2']['traits'][t]['best_r'] if 'qwen2' in probe else 0
    lqp = probe['qwen2']['traits'][t]['best_layer'] if 'qwen2' in probe else '?'
    sqp = '✓' if 'qwen2' in probe and probe['qwen2']['traits'][t]['validated'] else '✗'

    # Mistral template
    rmt = template['mistral']['traits'][t]['best_r']
    smt = '✓' if template['mistral']['traits'][t]['validated'] else '✗'

    # Mistral probe
    rmp = probe['mistral']['traits'][t]['best_r'] if 'mistral' in probe else 0
    lmp = probe['mistral']['traits'][t]['best_layer'] if 'mistral' in probe else '?'
    smp = '✓' if 'mistral' in probe and probe['mistral']['traits'][t]['validated'] else '✗'
    # Weak but positive
    if not smp == '✓' and rmp > 0.15:
        smp = '⚠'

    col_l  = f'{rl:.3f} {sl}'
    col_qt = f'{rqt:.3f} {sqt}'
    col_qp = f'{rqp:.3f} {sqp}'
    col_mt = f'{rmt:.3f} {smt}'
    col_mp = f'{rmp:.3f} {smp}'

    print(f'{t:<40} {col_l:>14} {col_qt:>12} {col_qp:>12} {col_mt:>12} {col_mp:>12}')

print()

# ── Section 2: Summary Scorecard ──────────────────────────────────────────────

print('=' * 130)
print('VALIDATION SCORECARD (r > 0.3 = validated, 0.15-0.3 = weak, <0.15 = failed)')
print('=' * 130)

def count_status(data, model_traits):
    validated = sum(1 for t in traits if model_traits[t]['validated'])
    weak = sum(1 for t in traits if not model_traits[t]['validated'] and model_traits[t]['best_r'] > 0.15)
    failed = 8 - validated - weak
    return validated, weak, failed

# Llama3
v, w, f_ = count_status(template, template['llama3']['traits'])
print(f'  Llama3-8B  (template):  {v}/8 validated, {w} weak, {f_} failed')

# Qwen2
v, w, f_ = count_status(template, template['qwen2']['traits'])
print(f'  Qwen2-7B   (template):  {v}/8 validated, {w} weak, {f_} failed')
if 'qwen2' in probe:
    v, w, f_ = count_status(probe, probe['qwen2']['traits'])
    print(f'  Qwen2-7B   (probe):     {v}/8 validated, {w} weak, {f_} failed  ← Solution B')

# Mistral
v, w, f_ = count_status(template, template['mistral']['traits'])
print(f'  Mistral-7B (template):  {v}/8 validated, {w} weak, {f_} failed')
if 'mistral' in probe:
    v, w, f_ = count_status(probe, probe['mistral']['traits'])
    print(f'  Mistral-7B (probe):     {v}/8 validated, {w} weak, {f_} failed  ← Solution B')

print()

# ── Section 3: Improvement Delta ─────────────────────────────────────────────

print('=' * 130)
print('IMPROVEMENT: TEMPLATE → CONTRASTIVE PROBING')
print('=' * 130)
print(f'{"Trait":<40} {"Qwen2 Δr":>12} {"Qwen2 dir":>10} {"Mistral Δr":>12} {"Mistral dir":>10}')
print('-' * 130)

for t in traits:
    rqt = template['qwen2']['traits'][t]['best_r']
    rqp = probe.get('qwen2', {}).get('traits', {}).get(t, {}).get('best_r', 0)
    dq = rqp - rqt
    dq_dir = '↑' if dq > 0.05 else ('↓' if dq < -0.05 else '→')

    rmt = template['mistral']['traits'][t]['best_r']
    rmp = probe.get('mistral', {}).get('traits', {}).get(t, {}).get('best_r', 0)
    dm = rmp - rmt
    dm_dir = '↑' if dm > 0.05 else ('↓' if dm < -0.05 else '→')

    print(f'{t:<40} {dq:>+12.3f} {dq_dir:>10} {dm:>+12.3f} {dm_dir:>10}')

print()

# ── Section 4: Probe Diagnostics (sample quality) ────────────────────────────

print('=' * 130)
print('PROBE DIAGNOSTICS: SAMPLE COUNTS & CLASSIFIER ACCURACY')
print('=' * 130)

for m in ['qwen2', 'mistral']:
    if m not in probe_diag:
        continue
    print(f'\n  {m.upper()}:')
    print(f'    {"Trait":<40} {"N/class":>10} {"Accuracy":>10} {"Best Layer":>12} {"r":>8}')
    print(f'    {"-"*80}')
    for t in traits:
        diag_t = probe_diag[m].get(t, {})
        # Get sample count from best layer
        best_layer = probe[m]['traits'][t]['best_layer'] if m in probe else '?'
        layer_diag = diag_t.get(str(best_layer), {})
        n = layer_diag.get('n_high', '?')
        acc = layer_diag.get('train_accuracy', 0)
        r = probe[m]['traits'][t]['best_r'] if m in probe else 0
        flag = '  ← LOW SAMPLES' if isinstance(n, int) and n < 50 else ''
        print(f'    {t:<40} {str(n):>10} {acc:>10.3f} {str(best_layer):>12} {r:>8.3f}{flag}')

print()

# ── Section 5: Best Layer Comparison ──────────────────────────────────────────

print('=' * 130)
print('BEST LAYER PER TRAIT (Probe-based)')
print('=' * 130)
print(f'{"Trait":<40} {"Llama3 (T)":>12} {"Qwen2 (P)":>12} {"Mistral (P)":>12} {"Consensus?":>12}')
print('-' * 130)

for t in traits:
    ll = template['llama3']['traits'][t]['best_layer']
    lq = probe.get('qwen2', {}).get('traits', {}).get(t, {}).get('best_layer', '?')
    lm = probe.get('mistral', {}).get('traits', {}).get(t, {}).get('best_layer', '?')

    layers = [ll, lq, lm]
    spread = max(layers) - min(layers) if all(isinstance(l, int) for l in layers) else 99
    consensus = '✓ stable' if spread <= 3 else '✗ divergent'

    print(f'{t:<40} {"L"+str(ll):>12} {"L"+str(lq):>12} {"L"+str(lm):>12} {consensus:>12}')

print()

# ── Section 6: Final Cross-Model Summary ─────────────────────────────────────

print('=' * 130)
print('FINAL CROSS-MODEL SUMMARY (Best method per model)')
print('=' * 130)
print()
print('Method: Llama3=template, Qwen2=probe, Mistral=probe')
print()
print(f'{"Trait":<40} {"Llama3":>10} {"Qwen2":>10} {"Mistral":>10} {"Models ✓":>10}')
print('-' * 90)

total_cross = 0
for t in traits:
    rl = template['llama3']['traits'][t]['best_r']
    rq = probe.get('qwen2', {}).get('traits', {}).get(t, {}).get('best_r', 0)
    rm = probe.get('mistral', {}).get('traits', {}).get(t, {}).get('best_r', 0)

    sl = '✓' if rl >= 0.3 else ('⚠' if rl >= 0.15 else '✗')
    sq = '✓' if rq >= 0.3 else ('⚠' if rq >= 0.15 else '✗')
    sm = '✓' if rm >= 0.3 else ('⚠' if rm >= 0.15 else '✗')

    n_validated = sum(1 for r in [rl, rq, rm] if r >= 0.3)
    total_cross += n_validated

    print(f'{t:<40} {rl:>6.3f} {sl:>3} {rq:>6.3f} {sq:>3} {rm:>6.3f} {sm:>3} {n_validated:>7}/3')

print('-' * 90)
print(f'{"TOTAL":>40} {"":>10} {"":>10} {"":>10} {total_cross:>7}/24')
print()

# Count all significant and positive
sources = [
    ('llama3', template.get('llama3', {})),
    ('qwen2', probe.get('qwen2', {})),
    ('mistral', probe.get('mistral', {})),
]
n_sig = 0
n_pos = 0
for m, data in sources:
    for t in traits:
        tr = data.get('traits', {}).get(t, {})
        if tr.get('best_p', 1) < 0.05:
            n_sig += 1
        if tr.get('best_r', 0) > 0:
            n_pos += 1
print(f'  All significant (p < 0.05): {n_sig}/24')
print(f'  All positive r:             {n_pos}/24')
