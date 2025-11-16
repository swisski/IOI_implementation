"""
Path Patching for IOI Circuit Discovery

Implements path patching following ARENA 1.4.1 IOI tutorial and
IOI paper (Wang et al. 2022) Section 5.

Path patching is more sophisticated than activation patching:
- Activation patching: Replace ALL inputs to a component
- Path patching: Replace ONLY the contribution from a SPECIFIC sender to receiver

This isolates direct causal paths between components in the circuit.

Example: To test if head 9.9 receives information from head 2.2:
1. Patch ONLY the contribution from head 2.2 that flows into head 9.9
2. Leave all other inputs to 9.9 unchanged
3. Measure effect on output

Expected paths from IOI paper:
- Duplicate token heads (L0-3) → S-inhibition heads (L7-8)
- Duplicate token heads (L0-3) → Name mover heads (L9-11)
- S-inhibition heads (L7-8) → Name mover heads (L9-11)
"""

from typing import Dict, List, Tuple, Optional, Callable, Union
import torch
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from functools import partial

from src.analysis.activation_patching import (
    run_with_cache_both,
    get_logit_diff,
    compute_patching_effect
)


def get_path_patching_hook(
    sender_layer: int,
    sender_head: Optional[int],
    receiver_layer: int,
    receiver_head: Optional[int],
    cache_clean: ActivationCache,
    component_type: str = "attn"
) -> Callable:
    """
    Create hook function for path patching from sender to receiver.

    Path patching patches only the contribution from a specific sender component
    (e.g., attention head) to a specific receiver component, leaving all other
    inputs unchanged.

    Args:
        sender_layer: Layer index of sender
        sender_head: Head index of sender (None for MLP or residual)
        receiver_layer: Layer index of receiver
        receiver_head: Head index of receiver (None for MLP)
        cache_clean: Clean activation cache
        component_type: "attn" or "mlp"

    Returns:
        Hook function that patches the sender's contribution

    Example:
        >>> hook_fn = get_path_patching_hook(
        ...     sender_layer=2, sender_head=2,
        ...     receiver_layer=9, receiver_head=9,
        ...     cache_clean=clean_cache
        ... )
        >>> # Use hook_fn with model.run_with_hooks()
    """
    def path_patch_hook(activation: torch.Tensor, hook):
        """
        Patch only the sender's contribution to the receiver.

        For attention heads:
        - activation shape: (batch, seq_len, n_heads, d_head)
        - We patch only sender_head's output
        """
        if component_type == "attn":
            # Patch specific attention head output
            if sender_head is not None:
                # activation is attn.hook_result: (batch, seq, n_heads, d_head)
                activation[:, :, sender_head, :] = cache_clean[hook.name][:, :, sender_head, :]
            else:
                # Patch entire attention layer
                activation[:, :, :, :] = cache_clean[hook.name][:, :, :, :]

        elif component_type == "mlp":
            # Patch MLP output
            activation[:, :, :] = cache_clean[hook.name][:, :, :]

        return activation

    return path_patch_hook


def patch_path(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    sender: Tuple[int, Union[int, str]],
    receiver: Tuple[int, Union[int, str]],
    io_token_id: int,
    s_token_id: int,
    cache_clean: ActivationCache,
    position: int = -1
) -> float:
    """
    Patch the path from sender component to receiver component.

    Args:
        model: HookedTransformer instance
        clean_tokens: Clean prompt tokens
        corrupted_tokens: Corrupted prompt tokens
        sender: (layer, head) for attention head, (layer, 'mlp') for MLP
        receiver: (layer, head) or (layer, 'mlp')
        io_token_id: Token ID for IO
        s_token_id: Token ID for S
        cache_clean: Clean activation cache
        position: Position to measure logit diff

    Returns:
        Patched logit difference (float)

    Example:
        >>> # Patch path from head L2H2 to head L9H9
        >>> effect = patch_path(
        ...     model, clean_tokens, corrupted_tokens,
        ...     sender=(2, 2), receiver=(9, 9),
        ...     io_token_id=bob_id, s_token_id=alice_id,
        ...     cache_clean=clean_cache
        ... )
    """
    sender_layer, sender_component = sender
    receiver_layer, receiver_component = receiver

    # Determine component type
    sender_is_mlp = (sender_component == 'mlp')
    sender_head = None if sender_is_mlp else sender_component

    receiver_is_mlp = (receiver_component == 'mlp')
    receiver_head = None if receiver_is_mlp else receiver_component

    # Create hook for sender's output
    if sender_is_mlp:
        hook_name = f"blocks.{sender_layer}.hook_mlp_out"
        component_type = "mlp"
    else:
        hook_name = f"blocks.{sender_layer}.attn.hook_result"
        component_type = "attn"

    # Create patching hook
    hook_fn = get_path_patching_hook(
        sender_layer, sender_head,
        receiver_layer, receiver_head,
        cache_clean, component_type
    )

    # Run with hook
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)]
        )

    # Compute logit difference
    patched_logit_diff = get_logit_diff(
        patched_logits, io_token_id, s_token_id, position
    ).item()

    return patched_logit_diff


def compute_path_patching_matrix(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    sender_heads: List[Tuple[int, int]],
    receiver_heads: List[Tuple[int, int]],
    io_token_id: int,
    s_token_id: int,
    position: int = -1
) -> Dict[str, np.ndarray]:
    """
    Compute path patching effects for all sender->receiver pairs.

    This creates a matrix where entry [i, j] represents the effect of
    patching the path from sender_heads[i] to receiver_heads[j].

    Args:
        model: HookedTransformer instance
        clean_tokens: Clean prompt tokens
        corrupted_tokens: Corrupted prompt tokens
        sender_heads: List of (layer, head) tuples for senders
        receiver_heads: List of (layer, head) tuples for receivers
        io_token_id: Token ID for IO
        s_token_id: Token ID for S
        position: Position to measure logit diff

    Returns:
        Dictionary with:
            - effect_matrix: np.ndarray of shape (n_senders, n_receivers)
            - sender_heads: List of sender (layer, head) tuples
            - receiver_heads: List of receiver (layer, head) tuples
            - clean_logit_diff: Baseline clean logit diff
            - corrupted_logit_diff: Baseline corrupted logit diff

    Example:
        >>> # Test paths from duplicate token heads to name mover heads
        >>> sender_heads = [(0, 1), (1, 4), (2, 2)]  # Duplicate token heads
        >>> receiver_heads = [(9, 6), (9, 9), (10, 0)]  # Name mover heads
        >>> results = compute_path_patching_matrix(
        ...     model, clean_tokens, corrupted_tokens,
        ...     sender_heads, receiver_heads,
        ...     io_token_id=bob_id, s_token_id=alice_id
        ... )
        >>> # results['effect_matrix'][i, j] = effect of path from sender_i to receiver_j
    """
    # Get clean and corrupted caches
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

    # Initialize effect matrix
    n_senders = len(sender_heads)
    n_receivers = len(receiver_heads)
    effect_matrix = np.zeros((n_senders, n_receivers))

    print(f"Computing path patching matrix ({n_senders} senders × {n_receivers} receivers)...")
    print(f"Clean logit diff: {clean_logit_diff:.3f}")
    print(f"Corrupted logit diff: {corrupted_logit_diff:.3f}")
    print()

    # Compute effect for each sender->receiver pair
    for i, sender in enumerate(sender_heads):
        for j, receiver in enumerate(receiver_heads):
            # Patch this path
            patched_logit_diff = patch_path(
                model, clean_tokens, corrupted_tokens,
                sender, receiver,
                io_token_id, s_token_id,
                clean_cache, position
            )

            # Compute effect
            effect = compute_patching_effect(
                clean_logit_diff, corrupted_logit_diff, patched_logit_diff
            )

            effect_matrix[i, j] = effect

        # Print progress
        sender_layer, sender_head = sender
        print(f"  Sender L{sender_layer}H{sender_head}: "
              f"max effect = {effect_matrix[i].max():.3f} "
              f"(to L{receiver_heads[effect_matrix[i].argmax()][0]}H{receiver_heads[effect_matrix[i].argmax()][1]})")

    return {
        "effect_matrix": effect_matrix,
        "sender_heads": sender_heads,
        "receiver_heads": receiver_heads,
        "clean_logit_diff": clean_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff
    }


def find_important_paths(
    effect_matrix: np.ndarray,
    sender_heads: List[Tuple[int, int]],
    receiver_heads: List[Tuple[int, int]],
    threshold: float = 0.3
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """
    Find important paths based on effect threshold.

    Args:
        effect_matrix: Effect matrix from compute_path_patching_matrix
        sender_heads: List of sender (layer, head) tuples
        receiver_heads: List of receiver (layer, head) tuples
        threshold: Minimum effect to consider path important

    Returns:
        List of (sender, receiver, effect) tuples for important paths

    Example:
        >>> important_paths = find_important_paths(
        ...     results['effect_matrix'],
        ...     results['sender_heads'],
        ...     results['receiver_heads'],
        ...     threshold=0.3
        ... )
        >>> for sender, receiver, effect in important_paths:
        ...     print(f"L{sender[0]}H{sender[1]} -> L{receiver[0]}H{receiver[1]}: {effect:.3f}")
    """
    important_paths = []

    for i, sender in enumerate(sender_heads):
        for j, receiver in enumerate(receiver_heads):
            effect = effect_matrix[i, j]

            if effect >= threshold:
                important_paths.append((sender, receiver, effect))

    # Sort by effect (descending)
    important_paths.sort(key=lambda x: x[2], reverse=True)

    return important_paths


def analyze_ioi_circuit_paths(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    duplicate_token_heads: List[Tuple[int, int]],
    s_inhibition_heads: List[Tuple[int, int]],
    name_mover_heads: List[Tuple[int, int]],
    io_token_id: int,
    s_token_id: int
) -> Dict[str, any]:
    """
    Analyze all paths in the IOI circuit.

    Tests three types of paths expected from the IOI paper:
    1. Duplicate token heads → S-inhibition heads
    2. Duplicate token heads → Name mover heads
    3. S-inhibition heads → Name mover heads

    Args:
        model: HookedTransformer instance
        clean_tokens: Clean prompt tokens
        corrupted_tokens: Corrupted prompt tokens
        duplicate_token_heads: List of (layer, head) for duplicate token heads
        s_inhibition_heads: List of (layer, head) for S-inhibition heads
        name_mover_heads: List of (layer, head) for name mover heads
        io_token_id: Token ID for IO
        s_token_id: Token ID for S

    Returns:
        Dictionary with:
            - dup_to_s_inhibition: Results for duplicate → S-inhibition paths
            - dup_to_name_mover: Results for duplicate → name mover paths
            - s_inhibition_to_name_mover: Results for S-inhibition → name mover paths

    Example:
        >>> results = analyze_ioi_circuit_paths(
        ...     model, clean_tokens, corrupted_tokens,
        ...     duplicate_token_heads=[(0, 1), (1, 4), (2, 2)],
        ...     s_inhibition_heads=[(7, 3), (7, 9), (8, 6)],
        ...     name_mover_heads=[(9, 6), (9, 9), (10, 0)],
        ...     io_token_id=bob_id, s_token_id=alice_id
        ... )
    """
    print("=" * 80)
    print("ANALYZING IOI CIRCUIT PATHS")
    print("=" * 80)

    results = {}

    # 1. Duplicate token heads → S-inhibition heads
    if duplicate_token_heads and s_inhibition_heads:
        print("\n1. Duplicate Token Heads → S-Inhibition Heads")
        print("-" * 80)
        results["dup_to_s_inhibition"] = compute_path_patching_matrix(
            model, clean_tokens, corrupted_tokens,
            duplicate_token_heads, s_inhibition_heads,
            io_token_id, s_token_id
        )

    # 2. Duplicate token heads → Name mover heads
    if duplicate_token_heads and name_mover_heads:
        print("\n2. Duplicate Token Heads → Name Mover Heads")
        print("-" * 80)
        results["dup_to_name_mover"] = compute_path_patching_matrix(
            model, clean_tokens, corrupted_tokens,
            duplicate_token_heads, name_mover_heads,
            io_token_id, s_token_id
        )

    # 3. S-inhibition heads → Name mover heads
    if s_inhibition_heads and name_mover_heads:
        print("\n3. S-Inhibition Heads → Name Mover Heads")
        print("-" * 80)
        results["s_inhibition_to_name_mover"] = compute_path_patching_matrix(
            model, clean_tokens, corrupted_tokens,
            s_inhibition_heads, name_mover_heads,
            io_token_id, s_token_id
        )

    # Find and print important paths
    print("\n" + "=" * 80)
    print("IMPORTANT PATHS (threshold = 0.3)")
    print("=" * 80)

    for path_type, path_results in results.items():
        print(f"\n{path_type}:")
        important = find_important_paths(
            path_results["effect_matrix"],
            path_results["sender_heads"],
            path_results["receiver_heads"],
            threshold=0.3
        )

        if important:
            for sender, receiver, effect in important[:10]:  # Top 10
                print(f"  L{sender[0]}H{sender[1]:2d} → L{receiver[0]}H{receiver[1]:2d}: effect = {effect:.3f}")
        else:
            print("  No paths above threshold")

    return results


if __name__ == "__main__":
    from src.model.model_loader import load_ioi_model

    print("Testing Path Patching")
    print("=" * 80)

    # Load model
    result = load_ioi_model(device="cpu")
    model = result["model"]

    # Test with example prompts
    clean_prompt = "When Alice and Bob went to the store, Alice gave a bottle to"
    corrupted_prompt = "When Alice and Bob went to the store, Charlie gave a bottle to"

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    # Get token IDs
    bob_id = model.to_single_token(" Bob")
    alice_id = model.to_single_token(" Alice")

    # Test path from one head to another
    print("\nTesting single path: L2H2 -> L9H9")

    # Get clean cache
    from src.analysis.activation_patching import run_with_cache_both
    clean_cache, _, clean_logits, corrupted_logits = run_with_cache_both(
        model, clean_tokens, corrupted_tokens
    )

    patched_diff = patch_path(
        model, clean_tokens, corrupted_tokens,
        sender=(2, 2), receiver=(9, 9),
        io_token_id=bob_id, s_token_id=alice_id,
        cache_clean=clean_cache
    )

    clean_diff = get_logit_diff(clean_logits, bob_id, alice_id).item()
    corrupted_diff = get_logit_diff(corrupted_logits, bob_id, alice_id).item()
    effect = compute_patching_effect(clean_diff, corrupted_diff, patched_diff)

    print(f"Clean logit diff: {clean_diff:.3f}")
    print(f"Corrupted logit diff: {corrupted_diff:.3f}")
    print(f"Patched logit diff: {patched_diff:.3f}")
    print(f"Effect: {effect:.3f}")

    # Test path matrix
    print("\n" + "=" * 80)
    print("Testing path matrix")

    sender_heads = [(0, 1), (2, 2)]
    receiver_heads = [(9, 6), (9, 9)]

    matrix_results = compute_path_patching_matrix(
        model, clean_tokens, corrupted_tokens,
        sender_heads, receiver_heads,
        bob_id, alice_id
    )

    print("\nEffect matrix:")
    print(f"              ", end="")
    for r in receiver_heads:
        print(f"L{r[0]}H{r[1]:2d}  ", end="")
    print()

    for i, s in enumerate(sender_heads):
        print(f"L{s[0]}H{s[1]:2d}  ", end="")
        for j in range(len(receiver_heads)):
            print(f"{matrix_results['effect_matrix'][i, j]:6.3f} ", end="")
        print()
