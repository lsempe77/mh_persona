"""Deep exploratory look at the steered corpus to inform analysis design."""
import json
from collections import Counter, defaultdict

with open("results/steered_corpus_combined.json") as f:
    data = json.load(f)

responses = data["responses"]
print(f"Total: {len(responses)} responses\n")

# ── Show full field names in one record ──
print("Fields per record:", list(responses[0].keys()))
print()

# ── Compare extreme coefficients for SAME scenario across ALL traits ──
# Pick scenario S04 (suicidal ideation) — most clinically critical
SCENARIO = "S04_suicidal_ideation"
MODEL = "llama3"

print("=" * 80)
print(f"SCENARIO: {SCENARIO}  |  MODEL: {MODEL}")
print("=" * 80)

for trait in sorted(set(r["trait"] for r in responses)):
    print(f"\n{'─' * 70}")
    print(f"TRAIT: {trait}")
    print(f"{'─' * 70}")
    for coeff in [-3.0, 0.0, 3.0]:
        match = [r for r in responses
                 if r["model"] == MODEL
                 and r["trait"] == trait
                 and r["scenario_id"] == SCENARIO
                 and r["coefficient"] == coeff]
        if match:
            text = match[0]["response"]
            # Show first 250 chars
            preview = text[:250].replace("\n", " ↵ ")
            print(f"  coeff={coeff:+.1f}: {preview}...")
        else:
            print(f"  coeff={coeff:+.1f}: [MISSING]")

# ── Same for Qwen2 on a different scenario ──
SCENARIO2 = "S09_self_harm"
MODEL2 = "qwen2"
print("\n\n" + "=" * 80)
print(f"SCENARIO: {SCENARIO2}  |  MODEL: {MODEL2}")
print("=" * 80)

for trait in ["crisis_recognition", "boundary_maintenance", "sycophancy_harmful_validation"]:
    print(f"\n{'─' * 70}")
    print(f"TRAIT: {trait}")
    print(f"{'─' * 70}")
    for coeff in [-3.0, 0.0, 3.0]:
        match = [r for r in responses
                 if r["model"] == MODEL2
                 and r["trait"] == trait
                 and r["scenario_id"] == SCENARIO2
                 and r["coefficient"] == coeff]
        if match:
            text = match[0]["response"]
            preview = text[:350].replace("\n", " ↵ ")
            print(f"  coeff={coeff:+.1f}: {preview}...")

# ── Quick word-level patterns ──
print("\n\n" + "=" * 80)
print("WORD-LEVEL PATTERNS BY COEFFICIENT (across all models)")
print("=" * 80)

# Count key therapeutic/harmful markers
markers = {
    "empathy": ["understand", "hear you", "must be", "sounds like", "that's really", "feel"],
    "crisis": ["emergency", "911", "crisis", "hotline", "immediate", "danger", "safety plan"],
    "boundary": ["professional", "therapist", "beyond my", "recommend", "qualified"],
    "validation": ["absolutely right", "totally", "you're right", "completely agree", "of course"],
    "self_disclosure": ["i feel", "i'm worried", "i'm scared", "my heart", "i can't imagine"],
    "directive": ["you should", "you need to", "you must", "it's important that you"],
}

for coeff in [-3.0, -1.5, 0.0, 1.5, 3.0]:
    subset = [r for r in responses if r["coefficient"] == coeff]
    total_words = sum(len(r["response"].split()) for r in subset)
    print(f"\nCoefficient {coeff:+.1f} ({len(subset)} responses, {total_words} total words):")
    for cat, words in markers.items():
        count = sum(
            sum(r["response"].lower().count(w) for w in words)
            for r in subset
        )
        rate = count / len(subset)
        print(f"  {cat:20s}: {count:4d} ({rate:.2f}/response)")
