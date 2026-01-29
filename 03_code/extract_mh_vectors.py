"""
Example: Extract Mental Health Persona Vectors

This script demonstrates the full pipeline for extracting persona vectors
for mental health traits from a language model.

Usage:
    python extract_mh_vectors.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from persona_vectors.extractor import PersonaVectorExtractor, save_persona_vector
from persona_vectors.mental_health_traits import (
    ALL_MENTAL_HEALTH_TRAITS,
    POSITIVE_TRAITS,
    NEGATIVE_TRAITS,
    get_trait_by_name,
)


def load_model(model_name: str, device: str = "cuda", load_in_8bit: bool = False):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    load_kwargs = {
        "device_map": "auto" if device == "cuda" else device,
        "torch_dtype": torch.float16,
    }
    
    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
    return model, tokenizer


def extract_all_vectors(
    model,
    tokenizer,
    output_dir: Path,
    traits: list = None,
    device: str = "cuda",
):
    """Extract persona vectors for all specified traits."""
    
    if traits is None:
        traits = ALL_MENTAL_HEALTH_TRAITS
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = PersonaVectorExtractor(model, tokenizer, device=device)
    
    results = {}
    
    for trait in traits:
        print(f"\n{'='*60}")
        print(f"Extracting: {trait.name}")
        print(f"{'='*60}")
        
        try:
            pv = extractor.extract_persona_vector(trait)
            
            # Save the vector
            save_path = output_dir / f"{trait.name}_vector.pt"
            save_persona_vector(pv, str(save_path))
            
            results[trait.name] = {
                "success": True,
                "layer": pv.layer_idx,
                "strength": pv.strength,
                "save_path": str(save_path),
            }
            
            print(f"✓ Saved to: {save_path}")
            print(f"  Optimal layer: {pv.layer_idx}")
            print(f"  Strength: {pv.strength:.4f}")
            
        except Exception as e:
            results[trait.name] = {
                "success": False,
                "error": str(e),
            }
            print(f"✗ Error: {e}")
    
    # Save summary
    import json
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract mental health persona vectors")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vectors",
        help="Directory to save extracted vectors"
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        default=None,
        help="Specific traits to extract (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization"
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Extract only positive traits"
    )
    parser.add_argument(
        "--negative-only",
        action="store_true",
        help="Extract only negative traits"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(
        args.model, 
        device=args.device, 
        load_in_8bit=args.load_in_8bit
    )
    
    # Determine which traits to extract
    if args.traits:
        traits = [get_trait_by_name(name) for name in args.traits]
    elif args.positive_only:
        traits = POSITIVE_TRAITS
    elif args.negative_only:
        traits = NEGATIVE_TRAITS
    else:
        traits = ALL_MENTAL_HEALTH_TRAITS
    
    # Extract vectors
    results = extract_all_vectors(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        traits=traits,
        device=args.device,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r["success"])
    failed = sum(1 for r in results.values() if not r["success"])
    
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed traits:")
        for name, result in results.items():
            if not result["success"]:
                print(f"  - {name}: {result['error']}")


if __name__ == "__main__":
    main()
