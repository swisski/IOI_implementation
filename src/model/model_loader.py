"""
Model Loading and Configuration for IOI Experiments

Loads GPT2-small using TransformerLens (HookedTransformer) with proper
caching configuration for mechanistic interpretability experiments.

Following ARENA 1.4.1 IOI tutorial conventions.
"""

from typing import Dict, Any
import torch
from transformer_lens import HookedTransformer


def load_ioi_model(device: str = "cpu") -> Dict[str, Any]:
    """
    Load GPT2-small model for IOI experiments with activation caching.

    This function loads the GPT2-small model using TransformerLens's HookedTransformer,
    which provides hooks for caching and analyzing activations at every layer.
    The model is configured following ARENA 1.4.1 IOI tutorial conventions.

    Args:
        device: Device to load model on. Options:
                - "cpu": CPU (default)
                - "cuda": NVIDIA GPU
                - "mps": Apple Silicon GPU
                - "cuda:0", "cuda:1", etc.: Specific GPU

    Returns:
        Dictionary containing:
            - model: HookedTransformer instance with caching enabled
            - config: Model configuration dict with:
                - n_layers: Number of transformer layers
                - n_heads: Number of attention heads per layer
                - d_model: Model dimension / residual stream dimension
                - d_vocab: Vocabulary size
                - d_head: Dimension per attention head
                - n_ctx: Context window size
                - device: Device model is loaded on

    Example:
        >>> result = load_ioi_model(device="cuda")
        >>> model = result["model"]
        >>> config = result["config"]
        >>> print(f"Loaded GPT2-small with {config['n_layers']} layers")

    Notes:
        - The model is set to evaluation mode (model.eval())
        - Gradients are disabled for inference efficiency
        - All activations can be cached using model.run_with_cache()
        - Model parameters are NOT frozen, allowing for techniques like
          activation patching if needed
    """
    # Validate device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available. Falling back to CPU.")
        device = "cpu"

    # Load GPT2-small using HookedTransformer
    # This wraps the model with hooks at every layer for activation caching
    print(f"Loading GPT2-small on {device}...")
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=False,  # Don't center unembed (following ARENA conventions)
        center_writing_weights=False,  # Don't center writing weights
        fold_ln=False,  # Don't fold layer norm into weights
        device=device
    )

    # Set model to evaluation mode
    model.eval()

    # Extract model configuration
    # TransformerLens stores config in model.cfg
    cfg = model.cfg
    config = {
        "n_layers": cfg.n_layers,  # Number of transformer blocks (12 for GPT2-small)
        "n_heads": cfg.n_heads,    # Number of attention heads (12 for GPT2-small)
        "d_model": cfg.d_model,    # Model dimension (768 for GPT2-small)
        "d_vocab": cfg.d_vocab,    # Vocabulary size (50257 for GPT2)
        "d_head": cfg.d_head,      # Dimension per head (64 for GPT2-small)
        "n_ctx": cfg.n_ctx,        # Context window (1024 for GPT2)
        "device": str(device)
    }

    print(f"Model loaded successfully!")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Heads: {config['n_heads']}")
    print(f"  Model dim: {config['d_model']}")
    print(f"  Vocab size: {config['d_vocab']}")

    return {
        "model": model,
        "config": config
    }


def get_model_info(model: HookedTransformer) -> Dict[str, Any]:
    """
    Extract configuration information from a loaded HookedTransformer model.

    Args:
        model: HookedTransformer instance

    Returns:
        Dictionary with model configuration details
    """
    cfg = model.cfg
    return {
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_model": cfg.d_model,
        "d_vocab": cfg.d_vocab,
        "d_head": cfg.d_head,
        "n_ctx": cfg.n_ctx,
        "model_name": cfg.model_name if hasattr(cfg, 'model_name') else "unknown"
    }


def run_with_cache(
    model: HookedTransformer,
    prompts: list,
    return_type: str = "logits"
) -> tuple:
    """
    Run model on prompts and cache all activations.

    This is a convenience wrapper around model.run_with_cache() that follows
    ARENA conventions for IOI experiments.

    Args:
        model: HookedTransformer instance
        prompts: List of prompt strings or tensor of token IDs
        return_type: What to return - "logits" (default) or "loss"

    Returns:
        Tuple of (output, cache) where:
            - output: Model output (logits or loss)
            - cache: ActivationCache with all intermediate activations

    Example:
        >>> result = load_ioi_model()
        >>> model = result["model"]
        >>> prompts = ["When Alice and Bob went to the store, Alice gave a bottle to"]
        >>> logits, cache = run_with_cache(model, prompts)
        >>> # Access cached activations
        >>> attn_pattern = cache["pattern", 0]  # Attention patterns at layer 0
        >>> resid_post = cache["resid_post", 5]  # Residual stream after layer 5
    """
    with torch.no_grad():
        output, cache = model.run_with_cache(
            prompts,
            return_type=return_type
        )
    return output, cache


if __name__ == "__main__":
    # Example usage and testing
    print("Testing IOI Model Loader")
    print("=" * 50)

    # Load model
    result = load_ioi_model(device="cpu")
    model = result["model"]
    config = result["config"]

    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Test with a simple prompt
    print("\nTesting inference...")
    test_prompt = "When Alice and Bob went to the store, Alice gave a bottle to"
    logits, cache = run_with_cache(model, [test_prompt])

    print(f"Input prompt: {test_prompt}")
    print(f"Logits shape: {logits.shape}")
    print(f"Number of cached activations: {len(cache)}")

    # Show some cached activation names
    print("\nSample cached activations:")
    for i, key in enumerate(list(cache.keys())[:10]):
        print(f"  {key}")
    if len(cache) > 10:
        print(f"  ... and {len(cache) - 10} more")

    print("\nModel loader test complete!")
