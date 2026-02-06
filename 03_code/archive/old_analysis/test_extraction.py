"""
Quick Test: Validate persona vector extraction code structure

This test validates the code works without needing a full LLM.
Uses a tiny test model for structure validation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    from persona_vectors.extractor import (
        PersonaVectorExtractor,
        PersonaVector,
        TraitDefinition,
        save_persona_vector,
        load_persona_vector,
    )
    print("  ✓ extractor module")
    
    from persona_vectors.mental_health_traits import (
        ALL_MENTAL_HEALTH_TRAITS,
        POSITIVE_TRAITS,
        NEGATIVE_TRAITS,
        EMPATHETIC_RESPONSIVENESS,
        CRISIS_RECOGNITION,
        SYCOPHANCY_HARMFUL_VALIDATION,
        get_trait_by_name,
    )
    print("  ✓ mental_health_traits module")
    
    return True


def test_trait_definitions():
    """Test that trait definitions are well-formed."""
    print("\nValidating trait definitions...")
    
    from persona_vectors.mental_health_traits import ALL_MENTAL_HEALTH_TRAITS
    
    for trait in ALL_MENTAL_HEALTH_TRAITS:
        # Check required fields
        assert trait.name, f"Trait missing name"
        assert trait.description, f"{trait.name} missing description"
        assert len(trait.high_trait_prompts) >= 3, f"{trait.name} needs ≥3 high prompts"
        assert len(trait.low_trait_prompts) >= 3, f"{trait.name} needs ≥3 low prompts"
        
        # Check prompts are non-empty strings
        for prompt in trait.high_trait_prompts:
            assert isinstance(prompt, str) and len(prompt) > 50, \
                f"{trait.name} has invalid high prompt"
        for prompt in trait.low_trait_prompts:
            assert isinstance(prompt, str) and len(prompt) > 50, \
                f"{trait.name} has invalid low prompt"
        
        print(f"  ✓ {trait.name}: {len(trait.high_trait_prompts)} high / {len(trait.low_trait_prompts)} low prompts")
    
    return True


def test_extractor_init_mock():
    """Test extractor initialization with a mock/small model."""
    print("\nTesting extractor with GPT-2 (small model for structure validation)...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from persona_vectors.extractor import PersonaVectorExtractor, TraitDefinition
        
        # Use tiny GPT-2 for structure testing
        print("  Loading GPT-2 (tiny model for testing)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Initialize extractor
        extractor = PersonaVectorExtractor(model, tokenizer, device="cpu")
        
        print(f"  ✓ Extractor initialized")
        print(f"    - Layers: {extractor.n_layers}")
        print(f"    - Hidden size: {extractor.hidden_size}")
        print(f"    - Target layers: {extractor.target_layers}")
        
        # Test activation extraction
        test_text = "Hello, I'm feeling sad today."
        acts = extractor._get_activations(test_text)
        
        print(f"  ✓ Activation extraction works")
        print(f"    - Got activations from {len(acts)} layers")
        
        for layer_idx, act in acts.items():
            print(f"    - Layer {layer_idx}: shape {act.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_mini_extraction():
    """Test extraction with minimal prompts on GPT-2."""
    print("\nTesting mini extraction pipeline on GPT-2...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from persona_vectors.extractor import PersonaVectorExtractor, TraitDefinition
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        extractor = PersonaVectorExtractor(model, tokenizer, device="cpu")
        
        # Mini trait for testing
        mini_trait = TraitDefinition(
            name="test_empathy",
            description="Test trait",
            high_trait_prompts=[
                "I understand how you feel. That must be really hard.",
                "I hear your pain. I'm here for you.",
            ],
            low_trait_prompts=[
                "Just get over it. Everyone has problems.",
                "Stop complaining and do something about it.",
            ],
        )
        
        pv = extractor.extract_persona_vector(mini_trait)
        
        print(f"  ✓ Extraction completed!")
        print(f"    - Trait: {pv.trait_name}")
        print(f"    - Optimal layer: {pv.layer_idx}")
        print(f"    - Strength: {pv.strength:.4f}")
        print(f"    - Direction shape: {pv.direction.shape}")
        
        # Test projection
        test_response = "I'm so sorry you're going through this difficult time."
        proj = extractor.project_onto_vector(test_response, pv)
        print(f"  ✓ Projection test: {proj:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("PERSONA VECTOR EXTRACTION - CODE VALIDATION")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Trait Definitions", test_trait_definitions()))
    results.append(("Extractor Init", test_extractor_init_mock()))
    results.append(("Mini Extraction", test_mini_extraction()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + ("All tests passed! ✓" if all_passed else "Some tests failed ✗"))
    
    if all_passed:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("""
To extract persona vectors from a real model, run:

  python extract_mh_vectors.py --model meta-llama/Llama-3.1-8B-Instruct \\
                                --output-dir ../04_results/vectors \\
                                --load-in-8bit

Note: Requires HuggingFace access token for Llama models.
      Set HF_TOKEN environment variable or run `huggingface-cli login`
""")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
