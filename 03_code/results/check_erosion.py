import json

for model in ['llama3', 'qwen2']:
    with open(f'context_erosion_{model}.json') as f:
        d = json.load(f)
    print(f'\n{"="*60}')
    print(f'  {model.upper()}')
    print(f'{"="*60}')
    print(f'  Status: {d.get("status","?")}')
    print(f'  Conversations: {d.get("n_conversations","?")}')
    print(f'  Turns per conv: {d.get("turns_per_conversation","?")}')
    
    if 'trend_analysis' in d:
        ta = d['trend_analysis']
        sig = [t for t,v in ta.items() if v.get('significant')]
        conc = [t for t,v in ta.items() if v.get('concerning')]
        print(f'  Significant trends: {sig}')
        print(f'  Concerning trends: {conc}')
        for t, v in ta.items():
            slope = v.get('slope_per_10_turns', '?')
            p = v.get('p_value', '?')
            r2 = v.get('r_squared', '?')
            print(f'    {t}: slope/10t={slope}, p={p}, R2={r2}')
    else:
        print('  No trend analysis found')
    
    if 'summary' in d:
        s = d['summary']
        print(f'  Sessions with alerts: {s.get("sessions_with_alerts","?")}/{s.get("total_sessions","?")}')
        if 'alert_counts' in s:
            print(f'  Alert counts: {s["alert_counts"]}')
    
    # Check individual conversation alert levels
    if 'conversations' in d:
        levels = {}
        for conv in d['conversations']:
            lvl = conv.get('max_alert_level', 'unknown')
            levels[lvl] = levels.get(lvl, 0) + 1
        print(f'  Alert level distribution: {levels}')
