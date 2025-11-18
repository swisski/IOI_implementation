"""
Activation Patching for IOI Analysis

Implements activation patching following ARENA 1.4.1 IOI tutorial and the
Indirect Object Identification paper (Wang et al. 2022).

Activation patching is a causal intervention technique that:
1. Runs model on clean prompt (ABBA) and corrupted prompt (ABC)
2. Caches all activations from both runs
3. Replaces specific activations from corrupted run with clean activations
4. Measures how much this restores performance (logit difference)

Effect metric = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
- Effect ≈ 1.0: patching fully restores performance (component is important)
- Effect ≈ 0.0: patching has no effect (component is not important)
- Effect < 0.0: patching makes things worse
- Effect > 1.0: patching overshoots (rare but possible)
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
import torch
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from functools import partial

from src.analysis.ioi_baseline import compute_logit_diff


def get_logit_diff(
    logits: torch.Tensor,
    io_token_id: int,
    s_token_id: int,
    position: int = -1
) -> torch.Tensor:
    """
    Compute logit difference as a tensor (for gradients/hooks).

    Args:
        logits: Model logits of shape (batch, seq_len, vocab_size)
        io_token_id: Token ID for IO (correct answer)
        s_token_id: Token ID for S (incorrect answer)
        position: Position to extract logits from

    Returns:
        Logit difference tensor (scalar)
    """
    position_logits = logits[0, position, :]
    io_logit = position_logits[io_token_id]
    s_logit = position_logits[s_token_id]
    return io_logit - s_logit


def run_with_cache_both(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor
) -> Tuple[ActivationCache, ActivationCache, torch.Tensor, torch.Tensor]:
    """
    Run model on both clean and corrupted prompts, caching all activations.

    Args:
        model: HookedTransformer instance
        clean_tokens: Clean prompt tokens (ABBA pattern)
        corrupted_tokens: Corrupted prompt tokens (ABC pattern)

    Returns:
        Tuple of:
            - clean_cache: ActivationCache from clean run
            - corrupted_cache: ActivationCache from corrupted run
            - clean_logits: Logits from clean run
            - corrupted_logits: Logits from corrupted run

    Example:
        >>> clean_tokens = model.to_tokens("When Alice and Bob went to the store, Alice gave a bottle to")
        >>> corrupted_tokens = model.to_tokens("When Alice and Bob went to the store, Charlie gave a bottle to")
        >>> clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
        ...     model, clean_tokens, corrupted_tokens
        ... )
    """
    # Run clean prompt
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)

    # Run corrupted prompt
    with torch.no_grad():
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    return clean_cache, corrupted_cache, clean_logits, corrupted_logits


def patch_residual_stream(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    layer_idx: int,
    cache_clean: ActivationCache,
    io_token_id: int,
    s_token_id: int,
    position: int = -1
) -> float:
    """
    Patch the residual stream at a specific layer and return patched logit difference.

    This replaces the residual stream activations at layer_idx from the corrupted run
    with the activations from the clean run, then continues the forward pass.

    Args:
        model: HookedTransformer instance
        clean_tokens: Clean prompt tokens (not used directly, for consistency)
        corrupted_tokens: Corrupted prompt tokens to run
        layer_idx: Layer index to patch at (0 to n_layers-1)
        cache_clean: ActivationCache from clean run
        io_token_id: Token ID for IO (correct answer)
        s_token_id: Token ID for S (incorrect answer)
        position: Position to extract logits from

    Returns:
        Patched logit difference (float)

    Example:
        >>> patched_logit_diff = patch_residual_stream(
        ...     model, clean_tokens, corrupted_tokens, layer_idx=5,
        ...     cache_clean=clean_cache, io_token_id=bob_id, s_token_id=alice_id
        ... )
    """
    # Define hook function that patches activations
    def patch_hook(activation: torch.Tensor, hook):
        """Replace corrupted activations with clean activations."""
        # activation shape: (batch, seq_len, d_model)
        # Replace with clean activations
        return cache_clean[hook.name]

    # Run model with patching hook
    with torch.no_grad():
        # Add hook at the specified layer's residual stream
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_hook)]
        )

    # Compute logit difference
    patched_logit_diff = get_logit_diff(
        patched_logits, io_token_id, s_token_id, position
    ).item()

    return patched_logit_diff


def patch_attention_pattern(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    cache_clean: ActivationCache,
    io_token_id: int,
    s_token_id: int,
    position: int = -1
) -> float:
    """
    Patch attention pattern at a specific head and return patched logit difference.

    Args:
        model: HookedTransformer instance
        corrupted_tokens: Corrupted prompt tokens to run
        layer_idx: Layer index
        head_idx: Head index within layer
        cache_clean: ActivationCache from clean run
        io_token_id: Token ID for IO
        s_token_id: Token ID for S
        position: Position to extract logits from

    Returns:
        Patched logit difference (float)
    """
    def patch_hook(activation: torch.Tensor, hook):
        """Patch specific attention head."""
        # activation shape: (batch, n_heads, seq_len, seq_len)
        # Only patch the specified head
        activation[:, head_idx, :, :] = cache_clean[hook.name][:, head_idx, :, :]
        return activation

    with torch.no_grad():
        hook_name = f"blocks.{layer_idx}.attn.hook_pattern"
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_hook)]
        )

    patched_logit_diff = get_logit_diff(
        patched_logits, io_token_id, s_token_id, position
    ).item()

    return patched_logit_diff


def patch_attention_output(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    cache_clean: ActivationCache,
    io_token_id: int,
    s_token_id: int,
    position: int = -1
) -> float:
    """
    Patch attention head output (result of attention before adding to residual).

    Args:
        model: HookedTransformer instance
        corrupted_tokens: Corrupted prompt tokens to run
        layer_idx: Layer index
        head_idx: Head index within layer
        cache_clean: ActivationCache from clean run
        io_token_id: Token ID for IO
        s_token_id: Token ID for S
        position: Position to extract logits from

    Returns:
        Patched logit difference (float)
    """
    def patch_hook(activation: torch.Tensor, hook):
        """Patch specific attention head output."""
        # activation shape: (batch, seq_len, n_heads, d_head)
        # Only patch the specified head
        activation[:, :, head_idx, :] = cache_clean[hook.name][:, :, head_idx, :]
        return activation

    with torch.no_grad():
        hook_name = f"blocks.{layer_idx}.attn.hook_z"
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_hook)]
        )

    patched_logit_diff = get_logit_diff(
        patched_logits, io_token_id, s_token_id, position
    ).item()

    return patched_logit_diff


def compute_patching_effect(
    clean_logit_diff: float,
    corrupted_logit_diff: float,
    patched_logit_diff: float
) -> float:
    """
    Compute the effect of patching as a normalized metric.

    Effect = (patched - corrupted) / (clean - corrupted)

    Interpretation:
    - Effect ≈ 1.0: Patching fully restores clean performance (component is crucial)
    - Effect ≈ 0.0: Patching has no effect (component is not involved)
    - Effect < 0.0: Patching makes performance worse
    - Effect > 1.0: Patching overshoots clean performance (rare)

    Args:
        clean_logit_diff: Logit difference from clean run (ABBA)
        corrupted_logit_diff: Logit difference from corrupted run (ABC)
        patched_logit_diff: Logit difference after patching

    Returns:
        Patching effect (float)

    Example:
        >>> clean_diff = 3.5
        >>> corrupted_diff = -1.2
        >>> patched_diff = 2.8
        >>> effect = compute_patching_effect(clean_diff, corrupted_diff, patched_diff)
        >>> # effect ≈ 0.85, meaning patching restored 85% of performance
    """
    denominator = clean_logit_diff - corrupted_logit_diff

    # Avoid division by zero
    if abs(denominator) < 1e-6:
        return 0.0

    effect = (patched_logit_diff - corrupted_logit_diff) / denominator
    return effect


def patch_all_layers(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    io_token_id: int,
    s_token_id: int,
    position: int = -1
) -> Dict[str, Any]:
    """
    Patch residual stream at each layer and compute effects.

    Args:
        model: HookedTransformer instance
        clean_tokens: Clean prompt tokens
        corrupted_tokens: Corrupted prompt tokens
        io_token_id: Token ID for IO
        s_token_id: Token ID for S
        position: Position to extract logits from

    Returns:
        Dictionary with:
            - clean_logit_diff: Logit diff from clean run
            - corrupted_logit_diff: Logit diff from corrupted run
            - layer_effects: List of effects for each layer
            - layer_patched_diffs: List of patched logit diffs for each layer
    """
    # Get caches and logits
    clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
        model, clean_tokens, corrupted_tokens
    )

    # Compute baseline logit diffs
    clean_logit_diff = get_logit_diff(
        clean_logits, io_token_id, s_token_id, position
    ).item()

    corrupted_logit_diff = get_logit_diff(
        corrupted_logits, io_token_id, s_token_id, position
    ).item()

    # Patch each layer
    n_layers = model.cfg.n_layers
    layer_effects = []
    layer_patched_diffs = []

    print(f"Clean logit diff: {clean_logit_diff:.3f}")
    print(f"Corrupted logit diff: {corrupted_logit_diff:.3f}")
    print(f"\nPatching residual stream at each layer...")

    for layer_idx in range(n_layers):
        # Patch this layer
        patched_logit_diff = patch_residual_stream(
            model, clean_tokens, corrupted_tokens, layer_idx,
            clean_cache, io_token_id, s_token_id, position
        )

        # Compute effect
        effect = compute_patching_effect(
            clean_logit_diff, corrupted_logit_diff, patched_logit_diff
        )

        layer_effects.append(effect)
        layer_patched_diffs.append(patched_logit_diff)

        print(f"  Layer {layer_idx:2d}: effect = {effect:6.3f}, patched_diff = {patched_logit_diff:6.3f}")

    return {
        "clean_logit_diff": clean_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff,
        "layer_effects": layer_effects,
        "layer_patched_diffs": layer_patched_diffs
    }


def patch_all_heads(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    io_token_id: int,
    s_token_id: int,
    position: int = -1,
    patch_type: str = "output"
) -> Dict[str, Any]:
    """
    Patch each attention head and compute effects.

    Args:
        model: HookedTransformer instance
        clean_tokens: Clean prompt tokens
        corrupted_tokens: Corrupted prompt tokens
        io_token_id: Token ID for IO
        s_token_id: Token ID for S
        position: Position to extract logits from
        patch_type: "output" or "pattern" - what to patch

    Returns:
        Dictionary with:
            - clean_logit_diff: Logit diff from clean run
            - corrupted_logit_diff: Logit diff from corrupted run
            - head_effects: 2D array (n_layers, n_heads) of effects
            - head_patched_diffs: 2D array of patched logit diffs
    """
    # Get caches and logits
    clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
        model, clean_tokens, corrupted_tokens
    )

    # Compute baseline logit diffs
    clean_logit_diff = get_logit_diff(
        clean_logits, io_token_id, s_token_id, position
    ).item()

    corrupted_logit_diff = get_logit_diff(
        corrupted_logits, io_token_id, s_token_id, position
    ).item()

    # Patch each head
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_effects = np.zeros((n_layers, n_heads))
    head_patched_diffs = np.zeros((n_layers, n_heads))

    print(f"Clean logit diff: {clean_logit_diff:.3f}")
    print(f"Corrupted logit diff: {corrupted_logit_diff:.3f}")
    print(f"\nPatching attention heads ({patch_type})...")

    patch_fn = patch_attention_output if patch_type == "output" else patch_attention_pattern

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Patch this head
            patched_logit_diff = patch_fn(
                model, corrupted_tokens, layer_idx, head_idx,
                clean_cache, io_token_id, s_token_id, position
            )

            # Compute effect
            effect = compute_patching_effect(
                clean_logit_diff, corrupted_logit_diff, patched_logit_diff
            )

            head_effects[layer_idx, head_idx] = effect
            head_patched_diffs[layer_idx, head_idx] = patched_logit_diff

        print(f"  Layer {layer_idx:2d}: max effect = {head_effects[layer_idx].max():6.3f} "
              f"at head {head_effects[layer_idx].argmax()}")

    return {
        "clean_logit_diff": clean_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff,
        "head_effects": head_effects,
        "head_patched_diffs": head_patched_diffs
    }


def analyze_example_patching(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    io_name: str,
    s_name: str
) -> Dict[str, Any]:
    """
    Run complete patching analysis on a single example.

    Args:
        model: HookedTransformer instance
        clean_prompt: Clean ABBA prompt
        corrupted_prompt: Corrupted ABC prompt
        io_name: IO token name (correct answer)
        s_name: S token name (subject)

    Returns:
        Dictionary with all patching results
    """
    # Tokenize
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    # Get token IDs
    io_token_id = model.to_single_token(io_name)
    s_token_id = model.to_single_token(s_name)

    print(f"Clean prompt: {clean_prompt}")
    print(f"Corrupted prompt: {corrupted_prompt}")
    print(f"IO token: {io_name} (id: {io_token_id})")
    print(f"S token: {s_name} (id: {s_token_id})")
    print("=" * 60)

    # Patch all layers
    layer_results = patch_all_layers(
        model, clean_tokens, corrupted_tokens,
        io_token_id, s_token_id
    )

    print("\n" + "=" * 60)

    # Patch all heads (output)
    head_results = patch_all_heads(
        model, clean_tokens, corrupted_tokens,
        io_token_id, s_token_id,
        patch_type="output"
    )

    return {
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "io_name": io_name,
        "s_name": s_name,
        "layer_results": layer_results,
        "head_results": head_results
    }


if __name__ == "__main__":
    from src.model.model_loader import load_ioi_model

    print("Testing Activation Patching")
    print("=" * 60)

    # Load model
    result = load_ioi_model(device="cpu")
    model = result["model"]

    # Test with example prompts
    clean_prompt = "When Alice and Bob went to the store, Alice gave a bottle to"
    corrupted_prompt = "When Alice and Bob went to the store, Charlie gave a bottle to"

    results = analyze_example_patching(
        model, clean_prompt, corrupted_prompt,
        io_name="Bob", s_name="Alice"
    )

    print("\n" + "=" * 60)
    print("Top 3 layers by effect:")
    layer_effects = results["layer_results"]["layer_effects"]
    top_layers = np.argsort(layer_effects)[::-1][:3]
    for i, layer_idx in enumerate(top_layers):
        print(f"  {i+1}. Layer {layer_idx}: effect = {layer_effects[layer_idx]:.3f}")

    print("\nTop 3 heads by effect:")
    head_effects = results["head_results"]["head_effects"]
    top_heads_flat = np.argsort(head_effects.flatten())[::-1][:3]
    for i, flat_idx in enumerate(top_heads_flat):
        layer_idx = flat_idx // model.cfg.n_heads
        head_idx = flat_idx % model.cfg.n_heads
        effect = head_effects[layer_idx, head_idx]
        print(f"  {i+1}. Layer {layer_idx}, Head {head_idx}: effect = {effect:.3f}")
