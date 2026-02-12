"""Quick diagnostic: per-trait score distributions and disagreement patterns."""
import json
import numpy as np
from collections import defaultdict

gpt = json.load(open('results/judge_scores_gpt4o_mini.json'))
claude = json.load(open('results/judge_scores_claude.json'))
gemini = json.load(open('results/judge_scores_gemini.json'))

g = {s['response_idx']: s['score'] for s in gpt['scores']}
c = {s['response_idx']: s['score'] for s in claude['scores']}
ge = {s['response_idx']: s['score'] for s in gemini['scores']}

corpus = json.load(open('results/steered_corpus_combined.json'))
items = corpus if isinstance(corpus, list) else corpus.get('responses', corpus.get('items', []))

trait_data = defaultdict(lambda: {'gpt': [], 'claude': [], 'gemini': [], 'diffs': []})

for idx, item in enumerate(items):
    trait = item['trait']
    if idx in g and idx in c and idx in ge:
        gs, cs, ges = g[idx], c[idx], ge[idx]
        trait_data[trait]['gpt'].append(gs)
        trait_data[trait]['claude'].append(cs)
        trait_data[trait]['gemini'].append(ges)
        trait_data[trait]['diffs'].append(max(gs, cs, ges) - min(gs, cs, ges))

for trait in sorted(trait_data.keys()):
    d = trait_data[trait]
    print(f"\n=== {trait} ===")
    for judge_name, scores in [('GPT-4o', d['gpt']), ('Claude', d['claude']), ('Gemini', d['gemini'])]:
        dist = [0]*7
        for s in scores:
            dist[int(s)-1] += 1
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        print(f"  {judge_name:8s}: " + " ".join(f"{i+1}:{dist[i]:3d}" for i in range(7)) + f"  mean={mean_s:.2f} std={std_s:.2f}")
    diffs = d['diffs']
    big_disagree = sum(1 for x in diffs if x >= 3)
    print(f"  Max spread: mean={np.mean(diffs):.2f}, >=3pt: {big_disagree}/{len(diffs)} ({100*big_disagree/len(diffs):.0f}%)")

    # Show worst disagreements
    worst = []
    for idx2, item in enumerate(items):
        if item['trait'] == trait and idx2 in g and idx2 in c and idx2 in ge:
            spread = max(g[idx2], c[idx2], ge[idx2]) - min(g[idx2], c[idx2], ge[idx2])
            if spread >= 4:
                worst.append((spread, g[idx2], c[idx2], ge[idx2], item.get('coefficient', '?')))
    if worst:
        worst.sort(reverse=True)
        print(f"  Worst disagreements (spread>=4): {len(worst)}")
        for sp, gs, cs, ges, coeff in worst[:3]:
            print(f"    GPT={gs} Claude={cs} Gemini={ges} coeff={coeff} spread={sp}")
