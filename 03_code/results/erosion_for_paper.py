"""Comprehensive erosion analysis for paper."""
import json
import numpy as np
from collections import defaultdict

for model in ['llama3', 'qwen2', 'mistral']:
    with open(f'context_erosion_{model}.json') as f:
        d = json.load(f)
    
    sessions = d.get('sessions', [])
    n_sessions = len(sessions)
    
    print(f"\n{'='*70}")
    print(f"  {model.upper()}: {n_sessions} sessions")
    print(f"{'='*70}")
    
    # 1. Session-level alert distribution
    session_severity = defaultdict(int)
    for s in sessions:
        ss = s.get('session_summary', {})
        session_severity[ss.get('max_severity', 'none')] += 1
    print(f"\n  Session severity: {dict(session_severity)}")
    
    # 2. Turn-level alert rates (from trait-level alert_level)
    turn_alert_counts = defaultdict(int)
    total_turns = 0
    for s in sessions:
        for turn in s.get('turns', []):
            total_turns += 1
            max_level = turn.get('max_alert_level', 'none')
            if max_level is None:
                max_level = 'none'
            turn_alert_counts[max_level] += 1
    
    print(f"\n  Turn-level alerts ({total_turns} total turns):")
    for level in ['none', 'watch', 'warning', 'critical']:
        ct = turn_alert_counts.get(level, 0)
        pct = ct / total_turns * 100 if total_turns else 0
        print(f"    {level}: {ct} ({pct:.1f}%)")
    
    # 3. Per-trait alert breakdown
    print(f"\n  Per-trait alert counts (across all sessions):")
    trait_totals = defaultdict(lambda: defaultdict(int))
    for s in sessions:
        for trait, ts in s.get('trait_summaries', {}).items():
            ac = ts.get('alert_counts', {})
            for level, count in ac.items():
                trait_totals[trait][level] += count
    
    for trait in sorted(trait_totals):
        watch = trait_totals[trait].get('watch', 0)
        warn = trait_totals[trait].get('warning', 0)
        crit = trait_totals[trait].get('critical', 0)
        total_alerts = watch + warn + crit
        print(f"    {trait:<42} w={watch:4d} W={warn:4d} C={crit:4d} total={total_alerts}")
    
    # 4. When do first alerts appear? (first turn with any alert per session)
    first_alert_turns = []
    for s in sessions:
        found = False
        for turn in s.get('turns', []):
            if turn.get('max_alert_level', 'none') not in (None, 'none'):
                first_alert_turns.append(turn['turn'])
                found = True
                break
        if not found:
            first_alert_turns.append(None)
    
    valid_firsts = [t for t in first_alert_turns if t is not None]
    if valid_firsts:
        print(f"\n  First alert appearance (turn):")
        print(f"    Mean: {np.mean(valid_firsts):.1f}")
        print(f"    Median: {np.median(valid_firsts):.1f}")
        print(f"    Min: {min(valid_firsts)}, Max: {max(valid_firsts)}")
        print(f"    Sessions with no alerts: {first_alert_turns.count(None)}/{n_sessions}")
    
    # 5. CUSUM accumulation: values at turns 25, 50, 75, 100
    print(f"\n  Mean CUSUM values across sessions at key turns:")
    for checkpoint_turn in [25, 50, 75, 99]:
        cusums = defaultdict(list)
        for s in sessions:
            turns = s.get('turns', [])
            if checkpoint_turn < len(turns):
                turn = turns[checkpoint_turn]
                for trait, td in turn.get('traits', {}).items():
                    ch = td.get('cusum_high', 0)
                    cl = td.get('cusum_low', 0)
                    cusums[trait].append(max(ch, cl))
        
        if cusums:
            print(f"    Turn {checkpoint_turn+1}:")
            for trait in sorted(cusums):
                vals = cusums[trait]
                print(f"      {trait:<40} mean={np.mean(vals):.2f} max={np.max(vals):.2f}")

# Cross-model trend comparison table
print(f"\n{'='*70}")
print("CROSS-MODEL TREND TABLE (for paper)")
print(f"{'='*70}")
print(f"{'Trait':<42} {'Llama3':>10} {'Qwen2':>10} {'Mistral':>10}")
print(f"{'':42} {'slope/10t':>10} {'slope/10t':>10} {'slope/10t':>10}")
print("-" * 74)

all_data = {}
for model in ['llama3', 'qwen2', 'mistral']:
    with open(f'context_erosion_{model}.json') as f:
        all_data[model] = json.load(f)

traits = sorted(set(
    t for m in all_data.values() 
    for t in m.get('trend_analysis', {}) 
    if not t.startswith('_')
))

for trait in traits:
    vals = []
    for model in ['llama3', 'qwen2', 'mistral']:
        ta = all_data[model].get('trend_analysis', {}).get(trait, {})
        slope = ta.get('slope', 0)
        p = ta.get('p_value', 1)
        conc = ta.get('concerning', False)
        s10 = slope * 10
        sig = '*' if p < 0.05 else ''
        c = '†' if conc else ''
        vals.append(f"{s10:+.4f}{sig}{c}")
    print(f"  {trait:<40} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

# VADER
print()
for model in ['llama3', 'qwen2', 'mistral']:
    ta = all_data[model].get('trend_analysis', {}).get('_vader_sentiment', {})
    slope = ta.get('slope', 0)
    r2 = ta.get('r_squared', 0)
    print(f"  VADER {model}: slope={slope:+.6f} R²={r2:.4f}")
