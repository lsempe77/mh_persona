"""Deeper analysis of session-level alert patterns."""
import json

for model in ['llama3', 'qwen2', 'mistral']:
    with open(f'context_erosion_{model}.json') as f:
        d = json.load(f)
    
    sessions = d.get('sessions', [])
    print(f"\n{'='*60}")
    print(f"  {model.upper()}: Session alert details")
    print(f"{'='*60}")
    
    for s in sessions:
        ss = s.get('session_summary', {})
        conv_idx = s.get('conversation_idx', '?')
        n_turns = s.get('n_turns', 0)
        severity = ss.get('max_severity', 'none')
        any_alert = ss.get('any_alert', False)
        
        # Collect which traits triggered alerts
        trait_alerts = ss.get('trait_alerts', {})
        alerted_traits = []
        for trait, ta in trait_alerts.items():
            if ta.get('any_alert'):
                max_sev = ta.get('max_severity', 'none')
                n_alerts = ta.get('n_alerts', 0)
                alerted_traits.append(f"{trait}({max_sev},{n_alerts})")
        
        if any_alert:
            print(f"  Conv {conv_idx}: {severity} | {n_turns}t | {', '.join(alerted_traits)}")

    # CUSUM analysis - when do alerts first appear?
    print(f"\n  First alert turn distribution:")
    first_alert_turns = []
    for s in sessions:
        for turn in s.get('turns', []):
            for trait, td in turn.get('traits', {}).items():
                cusum = td.get('cusum_value', 0)
                ewma_z = td.get('ewma_z', 0)
                alerts = td.get('alerts', [])
                if alerts:
                    first_alert_turns.append(turn['turn'])
                    break
            else:
                continue
            break
    
    if first_alert_turns:
        print(f"    Mean: {sum(first_alert_turns)/len(first_alert_turns):.1f}")
        print(f"    Min: {min(first_alert_turns)}, Max: {max(first_alert_turns)}")
        print(f"    Median: {sorted(first_alert_turns)[len(first_alert_turns)//2]}")

    # What does CUSUM look like at session end?
    print(f"\n  End-of-session CUSUM values (sample conv 2):")
    if len(sessions) > 2:
        s = sessions[2]
        last_turn = s.get('turns', [])[-1] if s.get('turns') else {}
        for trait, td in last_turn.get('traits', {}).items():
            cusum = td.get('cusum_value', 0)
            ewma_z = td.get('ewma_z', 0)
            print(f"    {trait}: CUSUM={cusum:.3f} EWMA_z={ewma_z:.3f}")
