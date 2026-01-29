"""
Persona Vector Extraction Pipeline

Implements the automated persona vector extraction from Chen et al. (2025):
1. Generate contrastive prompt pairs (high-trait vs low-trait)
2. Extract activation differences across layers
3. Find optimal direction for each trait
4. Validate on held-out examples
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class PersonaVector:
    """Represents an extracted persona vector for a trait."""
    trait_name: str
    direction: torch.Tensor  # Unit vector in activation space
    layer_idx: int           # Optimal layer for this trait
    strength: float          # Magnitude / discriminability
    metadata: Dict           # Additional info (prompts used, validation accuracy, etc.)


@dataclass 
class TraitDefinition:
    """Defines a trait for persona vector extraction."""
    name: str
    description: str
    high_trait_prompts: List[str]  # Prompts that elicit high-trait behavior
    low_trait_prompts: List[str]   # Prompts that elicit low-trait behavior
    validation_prompts: Optional[List[Tuple[str, bool]]] = None  # (prompt, is_high_trait)


class PersonaVectorExtractor:
    """
    Extracts persona vectors from language models using contrastive activation analysis.
    
    Based on Chen et al. (2025) "Persona Vectors: Monitoring and Controlling 
    Character Traits in Language Models"
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        target_layers: Optional[List[int]] = None,
    ):
        """
        Initialize the extractor.
        
        Args:
            model: HuggingFace model (e.g., LlamaForCausalLM)
            tokenizer: Corresponding tokenizer
            device: Device to run on
            target_layers: Which layers to extract from (None = all)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Get model architecture info
        self.n_layers = self._get_num_layers()
        self.hidden_size = self._get_hidden_size()
        
        if target_layers is None:
            # Default: middle to late layers (where traits are often most separable)
            mid = self.n_layers // 2
            self.target_layers = list(range(mid, self.n_layers))
        else:
            self.target_layers = target_layers
            
        # Storage for activation hooks
        self._activations = {}
        self._hooks = []
        
    def _get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'n_layer'):
            return self.model.config.n_layer
        else:
            raise ValueError("Cannot determine number of layers")
    
    def _get_hidden_size(self) -> int:
        """Get hidden dimension size."""
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        elif hasattr(self.model.config, 'n_embd'):
            return self.model.config.n_embd
        else:
            raise ValueError("Cannot determine hidden size")
    
    def _get_layer_module(self, layer_idx: int):
        """Get the module for a specific layer."""
        # Handle different model architectures
        if hasattr(self.model, 'model'):
            # Llama-style
            if hasattr(self.model.model, 'layers'):
                return self.model.model.layers[layer_idx]
        if hasattr(self.model, 'transformer'):
            # GPT-style
            if hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h[layer_idx]
        raise ValueError(f"Cannot find layer {layer_idx}")
    
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        self._activations = {}
        self._hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Get the hidden states (usually the first element of output tuple)
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                self._activations[layer_idx] = hidden.detach()
            return hook
        
        for layer_idx in self.target_layers:
            module = self._get_layer_module(layer_idx)
            hook = module.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}
    
    @torch.no_grad()
    def _get_activations(self, text: str, position: str = "last") -> Dict[int, torch.Tensor]:
        """
        Get activations for a text input.
        
        Args:
            text: Input text
            position: Which token position to use ("last", "mean", or int)
            
        Returns:
            Dict mapping layer_idx -> activation tensor [hidden_size]
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        self._register_hooks()
        try:
            _ = self.model(**inputs)
            
            result = {}
            for layer_idx, acts in self._activations.items():
                # acts shape: [batch, seq_len, hidden_size]
                if position == "last":
                    result[layer_idx] = acts[0, -1, :]  # Last token
                elif position == "mean":
                    result[layer_idx] = acts[0].mean(dim=0)  # Mean over sequence
                elif isinstance(position, int):
                    result[layer_idx] = acts[0, position, :]
                else:
                    raise ValueError(f"Unknown position: {position}")
                    
            return result
        finally:
            self._remove_hooks()
    
    def extract_persona_vector(
        self,
        trait: TraitDefinition,
        position: str = "last",
        normalize: bool = True,
    ) -> PersonaVector:
        """
        Extract a persona vector for a given trait.
        
        Args:
            trait: TraitDefinition with contrastive prompts
            position: Token position for activation extraction
            normalize: Whether to normalize the final direction
            
        Returns:
            PersonaVector with the extracted direction
        """
        print(f"Extracting persona vector for: {trait.name}")
        
        # Collect activations for high-trait prompts
        high_activations = {layer: [] for layer in self.target_layers}
        for prompt in tqdm(trait.high_trait_prompts, desc="High-trait prompts"):
            acts = self._get_activations(prompt, position)
            for layer_idx, act in acts.items():
                high_activations[layer_idx].append(act)
        
        # Collect activations for low-trait prompts
        low_activations = {layer: [] for layer in self.target_layers}
        for prompt in tqdm(trait.low_trait_prompts, desc="Low-trait prompts"):
            acts = self._get_activations(prompt, position)
            for layer_idx, act in acts.items():
                low_activations[layer_idx].append(act)
        
        # Compute mean difference at each layer
        directions = {}
        strengths = {}
        
        for layer_idx in self.target_layers:
            high_mean = torch.stack(high_activations[layer_idx]).mean(dim=0)
            low_mean = torch.stack(low_activations[layer_idx]).mean(dim=0)
            
            diff = high_mean - low_mean
            strength = diff.norm().item()
            
            if normalize:
                diff = F.normalize(diff, dim=0)
            
            directions[layer_idx] = diff
            strengths[layer_idx] = strength
        
        # Find optimal layer (highest discriminability)
        best_layer = max(strengths.keys(), key=lambda k: strengths[k])
        
        return PersonaVector(
            trait_name=trait.name,
            direction=directions[best_layer],
            layer_idx=best_layer,
            strength=strengths[best_layer],
            metadata={
                "n_high_prompts": len(trait.high_trait_prompts),
                "n_low_prompts": len(trait.low_trait_prompts),
                "all_layer_strengths": strengths,
                "position": position,
            }
        )
    
    def project_onto_vector(
        self,
        text: str,
        persona_vector: PersonaVector,
        position: str = "last",
    ) -> float:
        """
        Project a text's activation onto a persona vector.
        
        Returns the scalar projection (how much the text exhibits the trait).
        """
        acts = self._get_activations(text, position)
        layer_act = acts[persona_vector.layer_idx]
        
        projection = torch.dot(layer_act, persona_vector.direction).item()
        return projection
    
    def monitor_conversation(
        self,
        messages: List[Dict[str, str]],
        persona_vectors: List[PersonaVector],
    ) -> Dict[str, List[float]]:
        """
        Monitor persona trait levels across a conversation.
        
        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            persona_vectors: List of PersonaVectors to track
            
        Returns:
            Dict mapping trait_name -> list of projections after each assistant turn
        """
        trajectories = {pv.trait_name: [] for pv in persona_vectors}
        
        conversation_so_far = ""
        for msg in messages:
            conversation_so_far += f"{msg['role']}: {msg['content']}\n"
            
            if msg['role'] == 'assistant':
                for pv in persona_vectors:
                    proj = self.project_onto_vector(conversation_so_far, pv)
                    trajectories[pv.trait_name].append(proj)
        
        return trajectories


def save_persona_vector(pv: PersonaVector, path: str):
    """Save a PersonaVector to disk."""
    torch.save({
        'trait_name': pv.trait_name,
        'direction': pv.direction,
        'layer_idx': pv.layer_idx,
        'strength': pv.strength,
        'metadata': pv.metadata,
    }, path)


def load_persona_vector(path: str) -> PersonaVector:
    """Load a PersonaVector from disk."""
    data = torch.load(path)
    return PersonaVector(**data)
