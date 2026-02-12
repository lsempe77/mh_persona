import json
with open('context_erosion_llama3.json') as f:
    d = json.load(f)

# Check session summaries for trait alert details
s = d['sessions'][5]  # conv 5 (100 turns)
ss = s.get('session_summary', {})
print('Session summary keys:', list(ss.keys()))
print('max_severity:', ss.get('max_severity'))
print('any_alert:', ss.get('any_alert'))

# Check trait summaries
for trait, ts in s.get('trait_summaries', {}).items():
    wa = ts.get('worst_alert')
    mc = ts.get('max_cusum')
    mz = ts.get('max_abs_z')
    ac = ts.get('alert_counts')
    print(f"  {trait}: worst={wa}, max_cusum={mc}, max_z={mz}, alerts={ac}")

# Check a single turn data point
turn = s.get('turns', [])[50]
print(f"\nTurn 50 keys: {list(turn.keys())}")
print(f"Turn 50 max_alert_level: {turn.get('max_alert_level')}")
for trait, td in turn.get('traits', {}).items():
    print(f"  {trait}: {list(td.keys())}")
    print(f"    raw_proj={td.get('raw_projection', '?'):.4f}, ewma_z={td.get('ewma_z', '?')}, alert={td.get('alert_level')}")
    break  # just first trait
