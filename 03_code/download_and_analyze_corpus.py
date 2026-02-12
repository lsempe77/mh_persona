"""
Download the steered corpus from Modal volume and run basic text analysis.

Usage:
    modal run download_and_analyze_corpus.py
"""

import modal
import os
import json

app = modal.App("download-steered-corpus")
vol = modal.Volume.from_name("ai-persona-results", create_if_missing=False)

# ==========================================================================
# DOWNLOAD
# ==========================================================================

@app.function(volumes={"/results": vol}, timeout=120)
def download_corpus():
    """Read corpus files from Modal volume and return them."""
    vol.reload()

    data = {}
    for fname in [
        "steered_corpus_llama3.json",
        "steered_corpus_qwen2.json",
        "steered_corpus_mistral.json",
        "steered_corpus_combined.json",
    ]:
        path = f"/results/{fname}"
        if os.path.exists(path):
            with open(path) as f:
                data[fname] = json.load(f)
            n = len(data[fname].get("responses", []))
            print(f"  ✓ {fname}: {n} responses")
        else:
            print(f"  ⚠ {fname}: not found")
    return data


@app.local_entrypoint()
def main():
    """Download corpus to local disk and print summary stats."""
    print("Downloading steered corpus from Modal volume...")
    data = download_corpus.remote()

    os.makedirs("results", exist_ok=True)

    for fname, content in data.items():
        local_path = os.path.join("results", fname)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        n = len(content.get("responses", []))
        print(f"  Saved {local_path} ({n} responses)")

    # Summary stats for combined
    if "steered_corpus_combined.json" in data:
        corpus = data["steered_corpus_combined.json"]
        responses = corpus["responses"]
        print(f"\n{'='*60}")
        print(f"CORPUS SUMMARY")
        print(f"{'='*60}")
        print(f"Total responses: {len(responses)}")

        # By model
        by_model = {}
        for r in responses:
            by_model.setdefault(r["model"], []).append(r)
        for m, rs in sorted(by_model.items()):
            print(f"  {m}: {len(rs)} responses")

        # By trait
        by_trait = {}
        for r in responses:
            by_trait.setdefault(r["trait"], []).append(r)
        print(f"\nTraits: {len(by_trait)}")
        for t, rs in sorted(by_trait.items()):
            print(f"  {t}: {len(rs)} responses")

        # By coefficient
        by_coeff = {}
        for r in responses:
            by_coeff.setdefault(r["coefficient"], []).append(r)
        print(f"\nCoefficients:")
        for c, rs in sorted(by_coeff.items()):
            avg_len = sum(len(r["response"].split()) for r in rs) / len(rs)
            print(f"  coeff={c:+.1f}: {len(rs)} responses, avg {avg_len:.0f} words")

        # Quick linguistic metrics
        print(f"\n{'='*60}")
        print(f"QUICK LINGUISTIC ANALYSIS")
        print(f"{'='*60}")

        for trait in sorted(by_trait):
            trait_responses = by_trait[trait]
            print(f"\n  {trait}:")
            for coeff in sorted(set(r["coefficient"] for r in trait_responses)):
                subset = [r for r in trait_responses if r["coefficient"] == coeff]
                avg_words = sum(len(r["response"].split()) for r in subset) / max(len(subset), 1)
                avg_sentences = sum(r["response"].count(".") + r["response"].count("!") + r["response"].count("?")
                                   for r in subset) / max(len(subset), 1)
                # Count empathy markers
                empathy_words = ["understand", "feel", "sorry", "hear", "care", "painful", "difficult", "tough"]
                avg_empathy = sum(
                    sum(1 for w in empathy_words if w in r["response"].lower())
                    for r in subset
                ) / max(len(subset), 1)
                print(f"    coeff={coeff:+.1f}: {len(subset):3d} responses | "
                      f"words={avg_words:.0f} | sents={avg_sentences:.1f} | "
                      f"empathy_markers={avg_empathy:.1f}")
