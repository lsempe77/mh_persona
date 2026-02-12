"""Quick verification of re-scored Claude judge results."""
import json
from collections import Counter

with open("results/judge_scores_claude.json") as f:
    data = json.load(f)

scores = data["scores"]
print(f"Total scores: {len(scores)}")
print(f"Blocked: {sum(1 for s in scores if s.get('blocked', False))}")

print("\nScore=4.0 by scenario:")
scenarios = sorted(set(s["scenario_id"] for s in scores))
for sid in scenarios:
    n4 = sum(1 for s in scores if s["scenario_id"] == sid and s["score"] == 4.0)
    nt = sum(1 for s in scores if s["scenario_id"] == sid)
    print(f"  {sid:25s}: {n4:3d}/{nt:3d} ({100*n4/nt:5.1f}%)")

print("\nScore=4.0 by trait:")
traits = sorted(set(s["trait"] for s in scores))
for t in traits:
    n4 = sum(1 for s in scores if s["trait"] == t and s["score"] == 4.0)
    nt = sum(1 for s in scores if s["trait"] == t)
    print(f"  {t:45s}: {n4:3d}/{nt:3d} ({100*n4/nt:5.1f}%)")
