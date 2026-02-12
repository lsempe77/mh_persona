"""Quick analysis of the steered corpus."""
import json
from collections import Counter

with open("results/steered_corpus_combined.json") as f:
    data = json.load(f)

meta = data.get("metadata", {})
responses = data.get("responses", [])
print(f"Total responses: {len(responses)}")
print(f"Metadata keys: {list(meta.keys())}")
print()

# Breakdown by model
model_counts = Counter(r["model"] for r in responses)
print("By model:")
for m, c in sorted(model_counts.items()):
    print(f"  {m}: {c}")

# By trait
trait_counts = Counter(r["trait"] for r in responses)
print("\nBy trait:")
for t, c in sorted(trait_counts.items()):
    print(f"  {t}: {c}")

# By coefficient
coeff_counts = Counter(r["coefficient"] for r in responses)
print("\nBy coefficient:")
for co, c in sorted(coeff_counts.items()):
    print(f"  {co}: {c}")

# Average response length
lengths = [len(r["response"]) for r in responses]
print(f"\nResponse lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f} chars")

# By model x coeff
print("\nMean response length by model x coefficient:")
from itertools import groupby
for model in sorted(model_counts.keys()):
    print(f"  {model}:")
    for coeff in sorted(set(r["coefficient"] for r in responses)):
        subset = [r for r in responses if r["model"] == model and r["coefficient"] == coeff]
        mean_len = sum(len(r["response"]) for r in subset) / len(subset) if subset else 0
        print(f"    coeff={coeff:+.1f}: {mean_len:.0f} chars (n={len(subset)})")

# Sample extreme outputs
print("\n" + "="*70)
print("SAMPLE: Llama3 empathetic_responsiveness coeff=+3.0")
print("="*70)
for r in responses:
    if r["model"] == "llama3" and r["trait"] == "empathetic_responsiveness" and r["coefficient"] == 3.0:
        print(f"Scenario: {r['scenario_id']}")
        print(r["response"][:400])
        break

print("\n" + "="*70)
print("SAMPLE: Llama3 empathetic_responsiveness coeff=-3.0")
print("="*70)
for r in responses:
    if r["model"] == "llama3" and r["trait"] == "empathetic_responsiveness" and r["coefficient"] == -3.0:
        print(f"Scenario: {r['scenario_id']}")
        print(r["response"][:400])
        break

print("\n" + "="*70)
print("SAMPLE: Qwen2 crisis_recognition coeff=+3.0")
print("="*70)
for r in responses:
    if r["model"] == "qwen2" and r["trait"] == "crisis_recognition" and r["coefficient"] == 3.0:
        print(f"Scenario: {r['scenario_id']}")
        print(r["response"][:400])
        break

print("\n" + "="*70)
print("SAMPLE: Mistral boundary_maintenance coeff=-3.0")
print("="*70)
for r in responses:
    if r["model"] == "mistral" and r["trait"] == "boundary_maintenance" and r["coefficient"] == -3.0:
        print(f"Scenario: {r['scenario_id']}")
        print(r["response"][:400])
        break

# Check for empty responses
empty = [r for r in responses if len(r["response"].strip()) == 0]
print(f"\nEmpty responses: {len(empty)}")
if empty:
    for e in empty[:3]:
        print(f"  {e['model']}/{e['trait']}/coeff={e['coefficient']}")
