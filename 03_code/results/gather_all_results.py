"""Gather all experiment results for paper documentation."""
import json
import os

# === CONTEXT EROSION ===
print("=" * 70)
print("CONTEXT EROSION")
print("=" * 70)
for model in ["llama3", "qwen2", "mistral"]:
    fname = f"context_erosion_{model}.json"
    if not os.path.exists(fname):
        print(f"\n--- {model.upper()} --- MISSING")
        continue
    with open(fname) as f:
        d = json.load(f)
    print(f"\n--- {model.upper()} ---")
    print(f"  Status: {d['status']}")
    print(f"  Completed: {d['completed_conversations']}")
    s = d.get("aggregate_summary", {})
    print(f"  Sessions with alerts: {s.get('sessions_with_alerts', '?')}/{s.get('total_sessions', '?')}")
    print(f"  Sessions critical: {s.get('sessions_with_critical', '?')}")
    print(f"  Sig trends: {s.get('traits_with_significant_trend', [])}")
    print(f"  Concerning: {s.get('traits_with_concerning_trend', [])}")
    ta = d.get("trend_analysis", {})
    for t, v in sorted(ta.items()):
        if t.startswith("_"):
            if t == "_vader_sentiment":
                print(f"  VADER: slope={v['slope']:.6f} R2={v['r_squared']:.4f} p={v['p_value']:.4f}")
            continue
        if v.get("status") == "insufficient_data":
            continue
        c = " ***CONCERNING***" if v.get("concerning") else ""
        sig = "*" if v.get("significant") else ""
        print(f"  {t:45s} slope={v['slope']:+.6f} R2={v['r_squared']:.4f} p={v['p_value']:.4f}{sig}{c}")

# === FINE-TUNING REGRESSION ===
print("\n" + "=" * 70)
print("FINE-TUNING REGRESSION")
print("=" * 70)
fname = "finetuning_regression_results.json"
if os.path.exists(fname):
    with open(fname) as f:
        d = json.load(f)
    print(f"  Version: {d.get('version', '?')}")
    print(f"  Model: {d.get('model', '?')}")
    s = d.get("summary", {})
    print(f"  Summary: {json.dumps(s, indent=4)}")
    # Trait shifts
    ts = d.get("trait_shifts", {})
    print(f"\n  Trait shifts (activation projection delta):")
    for trait, info in sorted(ts.items()):
        if isinstance(info, dict):
            delta = info.get("delta_mean", info.get("delta", "?"))
            sig = info.get("significant", "?")
            p = info.get("p_value", "?")
            print(f"    {trait:45s} delta={delta}  sig={sig}  p={p}")
    # Text shifts
    txs = d.get("text_shifts", {})
    if txs:
        print(f"\n  Text shifts:")
        for feat, info in sorted(txs.items()):
            if isinstance(info, dict):
                delta = info.get("delta_mean", info.get("delta", "?"))
                sig = info.get("significant", "?")
                print(f"    {feat:45s} delta={delta}  sig={sig}")
    # Monitor alerts
    ma = d.get("monitor_alerts", {})
    if ma:
        print(f"\n  Monitor alerts: {json.dumps(ma, indent=4)}")
else:
    print("  MISSING")

# === ADVERSARIAL RED-TEAM ===
print("\n" + "=" * 70)
print("ADVERSARIAL RED-TEAM")
print("=" * 70)
fname = "adversarial_redteam_results.json"
if os.path.exists(fname):
    with open(fname) as f:
        d = json.load(f)
    print(f"  Version: {d.get('version', '?')}")
    print(f"  N trajectories: {d.get('n_trajectories', '?')}")
    s = d.get("summary", {})
    print(f"  Summary: {json.dumps(s, indent=4)}")
    # By attack type
    bat = d.get("by_attack_type", {})
    if bat:
        print(f"\n  By attack type:")
        for atype, info in sorted(bat.items()):
            if isinstance(info, dict):
                det = info.get("detection_rate", info.get("detected", "?"))
                lead = info.get("mean_lead_turns", info.get("lead_time", "?"))
                print(f"    {atype:35s} detection={det}  lead_turns={lead}")
    # By model
    bm = d.get("by_model", {})
    if bm:
        print(f"\n  By model:")
        for m, info in sorted(bm.items()):
            if isinstance(info, dict):
                det = info.get("detection_rate", "?")
                print(f"    {m:20s} detection_rate={det}")
    # Detection timing
    dt = d.get("detection_timing", {})
    if dt:
        print(f"\n  Detection timing: {json.dumps(dt, indent=4)}")
else:
    print("  MISSING")

# === QUANTISATION ===
print("\n" + "=" * 70)
print("QUANTISATION COMPARISON")
print("=" * 70)
for prec in ["nf4", "fp16", "8bit"]:
    fname = f"quantisation_{prec}.json"
    if not os.path.exists(fname):
        print(f"\n  {prec}: MISSING (not downloaded)")
        continue
    with open(fname) as f:
        d = json.load(f)
    print(f"\n--- {prec.upper()} ---")
    print(f"  N responses: {d.get('n_responses', '?')}")
    lr = d.get("layer_results", {})
    for trait, layers in sorted(lr.items()):
        if isinstance(layers, dict):
            best_l = layers.get("_best_layer", "?")
            best_r = layers.get("_best_r", "?")
            print(f"  {trait:45s} best_layer={best_l}  best_r={best_r}")
