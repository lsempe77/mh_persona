"""Examine specific scenario results to understand validation patterns."""
import json

# Load Llama3 checkpoint
with open('behavioral_validation_checkpoint_llama3.json') as f:
    d = json.load(f)

print("="*70)
print("SAMPLE SCENARIOS - Empathetic Responsiveness (r=0.43, VALIDATED)")
print("="*70)
for i, s in enumerate(d['results'][:5]):
    tr = s['trait_results']['empathetic_responsiveness']
    print(f"{i+1}. {s['scenario_name'][:45]}")
    print(f"   Act drift: {tr['activation_drift']:+.3f}, Beh drift: {tr['behavioral_drift']:+.1f}")
    print(f"   Proj: T1={tr['proj_t1']:.2f}, TN={tr['proj_tN']:.2f}")
    print(f"   Score: T1={tr['score_t1']}, TN={tr['score_tN']}")
    print()

print("="*70)
print("SAMPLE SCENARIOS - Boundary Maintenance (r=0.10, FAILED)")
print("="*70)
for i, s in enumerate(d['results'][:5]):
    tr = s['trait_results']['boundary_maintenance']
    print(f"{i+1}. {s['scenario_name'][:45]}")
    print(f"   Act drift: {tr['activation_drift']:+.3f}, Beh drift: {tr['behavioral_drift']:+.1f}")
    print(f"   Proj: T1={tr['proj_t1']:.2f}, TN={tr['proj_tN']:.2f}")
    print(f"   Score: T1={tr['score_t1']}, TN={tr['score_tN']}")
    print()

print("="*70)
print("SAMPLE SCENARIOS - Crisis Recognition (r=-0.03, FAILED)")
print("="*70)
for i, s in enumerate(d['results'][:5]):
    tr = s['trait_results']['crisis_recognition']
    print(f"{i+1}. {s['scenario_name'][:45]}")
    print(f"   Act drift: {tr['activation_drift']:+.3f}, Beh drift: {tr['behavioral_drift']:+.1f}")
    print(f"   Proj: T1={tr['proj_t1']:.2f}, TN={tr['proj_tN']:.2f}")
    print(f"   Score: T1={tr['score_t1']}, TN={tr['score_tN']}")
    print()

# Check actual response samples
print("="*70)
print("ACTUAL RESPONSE SAMPLES")
print("="*70)
for i, s in enumerate(d['results'][:2]):
    print(f"\n--- {s['scenario_name']} ---")
    print(f"Turn 1 Response: {s['turn1_response'][:200]}...")
    print(f"Turn N Response: {s['turnN_response'][:200]}...")
