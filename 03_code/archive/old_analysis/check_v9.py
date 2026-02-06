import json

d = json.load(open('v9_llm_judge_results.json'))
print('V9 Result keys for one trait:')
t = list(d['traits'].keys())[1]
print(f'{t}: {list(d["traits"][t].keys())}')
print()
print('Full data for non_judgmental_acceptance:')
print(json.dumps(d['traits']['non_judgmental_acceptance'], indent=2))
