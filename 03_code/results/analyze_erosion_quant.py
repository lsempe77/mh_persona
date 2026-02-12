"""Analyze context erosion + quantisation results for paper."""
import json

# ============================================================
# CONTEXT EROSION RESULTS
# ============================================================
print("=" * 70)
print("CONTEXT EROSION RESULTS")
print("=" * 70)

for model in ['llama3', 'qwen2', 'mistral']:
    with open(f'context_erosion_{model}.json') as f:
        d = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"  {model.upper()}")
    print(f"{'='*60}")
    print(f"  Status: {d.get('status')}")
    print(f"  Conversations: {d.get('completed_conversations')}")
    
    ta = d.get('trend_analysis', {})
    agg = d.get('aggregate_summary', {})
    
    print(f"  Sessions with alerts: {agg.get('sessions_with_alerts')}/{agg.get('total_sessions')}")
    print(f"  Sessions with critical: {agg.get('sessions_with_critical')}")
    print(f"  Significant trends: {agg.get('traits_with_significant_trend')}")
    print(f"  Concerning trends: {agg.get('traits_with_concerning_trend')}")
    
    print(f"\n  Trait-level trends:")
    for trait, v in sorted(ta.items()):
        if trait.startswith('_'):
            if trait == '_vader_sentiment':
                print(f"    VADER: slope={v.get('slope',0):+.6f} R²={v.get('r_squared',0):.4f} p={v.get('p_value',0):.6f} n={v.get('n')}")
            continue
        if v.get('status') == 'insufficient_data':
            print(f"    {trait}: insufficient data (n={v.get('n')})")
            continue
        slope = v.get('slope', 0)
        r2 = v.get('r_squared', 0)
        p = v.get('p_value', 0)
        r = v.get('r', 0)
        sig = '*' if v.get('significant') else ''
        conc = ' [CONCERNING]' if v.get('concerning') else ''
        direction = v.get('direction', '')
        slope_per_10 = slope * 10
        n = v.get('n', 0)
        print(f"    {trait:<42} slope/10t={slope_per_10:+.5f} R²={r2:.4f} r={r:+.4f} p={p:.6f}{sig}{conc} n={n}")

# ============================================================
# CROSS-MODEL COMBINED
# ============================================================
print(f"\n{'='*70}")
print("CROSS-MODEL COMBINED")
print(f"{'='*70}")

with open('context_erosion_results.json') as f:
    combined = json.load(f)

xm = combined.get('cross_model_summary', {})
print(f"  Concerning in any model: {xm.get('traits_concerning_in_any_model')}")
print(f"  Concerning in all models: {xm.get('traits_concerning_in_all_models')}")

# Per-trait cross-model summary
xmt = combined.get('cross_model_trends', {})
for trait, v in sorted(xmt.items()):
    print(f"\n  {trait}:")
    for model, mv in v.items():
        if model.startswith('_'):
            print(f"    {model}: {mv}")
            continue
        slope = mv.get('slope', 0)
        p = mv.get('p_value', 0)
        conc = mv.get('concerning', False)
        print(f"    {model}: slope={slope:+.6f} p={p:.4f} concerning={conc}")

# ============================================================
# QUANTISATION (NF4 + FP16 only for now)
# ============================================================
print(f"\n{'='*70}")
print("QUANTISATION RESULTS (NF4 + FP16)")
print(f"{'='*70}")

for prec in ['nf4', 'fp16']:
    try:
        with open(f'quantisation_{prec}.json') as f:
            d = json.load(f)
        
        print(f"\n--- {prec.upper()} ---")
        print(f"  N responses: {d.get('n_responses')}")
        lr = d.get('layer_results', {})
        for trait, tv in sorted(lr.items()):
            best_layer = tv.get('_best_layer', '?')
            best_r = tv.get('_best_r', '?')
            print(f"  {trait}: best_layer={best_layer} best_r={best_r}")
            for layer, lv in sorted(tv.items()):
                if layer.startswith('_'):
                    continue
                print(f"    L{layer}: r={lv.get('r',0):.4f} p={lv.get('p_value',0):.6f} n={lv.get('n',0)}")
    except FileNotFoundError:
        print(f"\n--- {prec.upper()} --- NOT FOUND")

# Session-level alert analysis
print(f"\n{'='*70}")
print("SESSION-LEVEL ALERT ANALYSIS")
print(f"{'='*70}")

for model in ['llama3', 'qwen2', 'mistral']:
    with open(f'context_erosion_{model}.json') as f:
        d = json.load(f)
    
    sessions = d.get('sessions', [])
    alert_counts = {'none': 0, 'watch': 0, 'warning': 0, 'critical': 0}
    
    for s in sessions:
        ss = s.get('session_summary', {})
        severity = ss.get('max_severity', 'none')
        if severity is None:
            severity = 'none'
        alert_counts[severity] = alert_counts.get(severity, 0) + 1
    
    n_turns_total = sum(s.get('n_turns', 0) for s in sessions)
    
    # Count alert types across turns
    turn_alerts = {'none': 0, 'watch': 0, 'warning': 0, 'critical': 0}
    for s in sessions:
        for turn in s.get('turns', []):
            # Check traits for alerts
            max_alert = 'none'
            for trait, td in turn.get('traits', {}).items():
                alerts = td.get('alerts', [])
                for a in alerts:
                    level = a.get('level', 'none')
                    if level == 'critical':
                        max_alert = 'critical'
                    elif level == 'warning' and max_alert != 'critical':
                        max_alert = 'warning'
                    elif level == 'watch' and max_alert not in ('critical', 'warning'):
                        max_alert = 'watch'
            turn_alerts[max_alert] += 1
    
    print(f"\n  {model.upper()}: {len(sessions)} sessions, {n_turns_total} total turns")
    print(f"  Session-level: {alert_counts}")
    print(f"  Turn-level: {turn_alerts}")
    total_turns = sum(turn_alerts.values())
    if total_turns > 0:
        print(f"  Turn alert rates: watch={turn_alerts['watch']/total_turns*100:.1f}% warning={turn_alerts['warning']/total_turns*100:.1f}% critical={turn_alerts['critical']/total_turns*100:.1f}%")
