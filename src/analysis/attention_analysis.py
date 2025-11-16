"""
Attention Pattern Analysis for IOI Circuit Discovery

Implements attention pattern extraction and analysis following ARENA 1.4.1
IOI tutorial and the IOI paper (Wang et al. 2022) Section 3.2.

The IOI circuit consists of three types of attention heads:
1. Duplicate Token Heads: Attend from second occurrence of name to first occurrence
2. S-Inhibition Heads: Attend from end position to S token (to suppress it)
3. Name Mover Heads: Attend from end position to IO token (to move it to output)
"""

import json
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from transformer_lens import HookedTransformer
from collections import defaultdict

from src.analysis.ioi_baseline import get_token_positions


def get_attention_patterns(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    head: int
) -> torch.Tensor:
    """
    Extract attention patterns for a specific head.

    Args:
        model: HookedTransformer instance
        tokens: Input token tensor of shape (batch, seq_len)
        layer: Layer index (0 to n_layers-1)
        head: Head index (0 to n_heads-1)

    Returns:
        Attention pattern tensor of shape (batch, seq_len_q, seq_len_k)
        For causal attention: (batch, seq_len, seq_len) where seq_len_q = seq_len_k

    Example:
        >>> tokens = model.to_tokens("When Alice and Bob went to the store, Alice gave a bottle to")
        >>> attn = get_attention_patterns(model, tokens, layer=9, head=6)
        >>> # attn[0, -1, :] gives attention from final position to all positions
    """
    # Run model with caching
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # Extract attention pattern for this head
    # Cache key format: "blocks.{layer}.attn.hook_pattern"
    attn_pattern = cache["pattern", layer]  # Shape: (batch, n_heads, seq_len, seq_len)

    # Extract specific head
    head_pattern = attn_pattern[:, head, :, :]  # Shape: (batch, seq_len, seq_len)

    return head_pattern


def compute_attention_to_position(
    attention_pattern: torch.Tensor,
    source_pos: int,
    target_pos: int
) -> float:
    """
    Get attention weight from source position to target position.

    Args:
        attention_pattern: Attention tensor of shape (batch, seq_len, seq_len)
        source_pos: Source (query) position
        target_pos: Target (key) position

    Returns:
        Attention weight (float between 0 and 1)

    Example:
        >>> attn = get_attention_patterns(model, tokens, layer=9, head=6)
        >>> # Attention from final position (-1) to position 5
        >>> weight = compute_attention_to_position(attn, source_pos=-1, target_pos=5)
    """
    # attention_pattern[batch, source, target]
    # For batch size 1, just extract [0, source, target]
    weight = attention_pattern[0, source_pos, target_pos].item()
    return weight


def get_name_positions(
    model: HookedTransformer,
    prompt: str,
    name_a: str,
    name_b: str
) -> Dict[str, List[int]]:
    """
    Get positions of names A and B in the prompt.

    Args:
        model: HookedTransformer instance
        prompt: Input prompt string
        name_a: First name (S in ABBA)
        name_b: Second name (IO in ABBA)

    Returns:
        Dictionary with:
            - "A_positions": List of positions where A appears
            - "B_positions": List of positions where B appears
    """
    a_positions = get_token_positions(model, prompt, name_a)
    b_positions = get_token_positions(model, prompt, name_b)

    return {
        "A_positions": a_positions,
        "B_positions": b_positions
    }


def analyze_duplicate_token_attention(
    model: HookedTransformer,
    prompt: str,
    name_a: str,
    layer: int,
    head: int
) -> float:
    """
    Analyze if head attends from second occurrence of A to first occurrence.

    Duplicate token heads are expected to attend from the second instance of
    the subject name (A) to the first instance.

    Args:
        model: HookedTransformer instance
        prompt: Input prompt (ABBA format)
        name_a: Subject name (appears twice)
        layer: Layer index
        head: Head index

    Returns:
        Attention weight from second A to first A

    Example:
        >>> prompt = "When Alice and Bob went to the store, Alice gave a bottle to"
        >>> # Alice appears at positions [0, X]. We want attention from X to 0
        >>> score = analyze_duplicate_token_attention(model, prompt, "Alice", layer=0, head=1)
    """
    tokens = model.to_tokens(prompt)
    attn_pattern = get_attention_patterns(model, tokens, layer, head)

    # Find positions of name A
    a_positions = get_token_positions(model, prompt, name_a)

    if len(a_positions) < 2:
        return 0.0

    # Get attention from second A to first A
    first_a = a_positions[0]
    second_a = a_positions[-1]

    attn_weight = compute_attention_to_position(attn_pattern, second_a, first_a)
    return attn_weight


def analyze_s_inhibition_attention(
    model: HookedTransformer,
    prompt: str,
    name_a: str,
    layer: int,
    head: int,
    end_position: int = -1
) -> float:
    """
    Analyze if head attends from end position to S token (for inhibition).

    S-inhibition heads attend from the final position to the subject name
    position to suppress it in the output.

    Args:
        model: HookedTransformer instance
        prompt: Input prompt (ABBA format)
        name_a: Subject name (S)
        layer: Layer index
        head: Head index
        end_position: Position to measure attention from (default: -1 for last)

    Returns:
        Average attention weight from end position to all S occurrences
    """
    tokens = model.to_tokens(prompt)
    attn_pattern = get_attention_patterns(model, tokens, layer, head)

    # Find positions of S name
    s_positions = get_token_positions(model, prompt, name_a)

    if len(s_positions) == 0:
        return 0.0

    # Get attention from end position to all S positions
    attn_weights = []
    for s_pos in s_positions:
        weight = compute_attention_to_position(attn_pattern, end_position, s_pos)
        attn_weights.append(weight)

    # Return average (or could return max)
    return np.mean(attn_weights)


def analyze_name_mover_attention(
    model: HookedTransformer,
    prompt: str,
    name_b: str,
    layer: int,
    head: int,
    end_position: int = -1
) -> float:
    """
    Analyze if head attends from end position to IO token (name mover).

    Name mover heads attend from the final position to the indirect object
    name and move it to the output.

    Args:
        model: HookedTransformer instance
        prompt: Input prompt (ABBA format)
        name_b: IO name (B in ABBA)
        layer: Layer index
        head: Head index
        end_position: Position to measure attention from (default: -1)

    Returns:
        Average attention weight from end position to all IO occurrences
    """
    tokens = model.to_tokens(prompt)
    attn_pattern = get_attention_patterns(model, tokens, layer, head)

    # Find positions of IO name
    io_positions = get_token_positions(model, prompt, name_b)

    if len(io_positions) == 0:
        return 0.0

    # Get attention from end position to all IO positions
    attn_weights = []
    for io_pos in io_positions:
        weight = compute_attention_to_position(attn_pattern, end_position, io_pos)
        attn_weights.append(weight)

    return np.mean(attn_weights)


def find_duplicate_token_heads(
    model: HookedTransformer,
    dataset: List[Dict],
    threshold: float = 0.5,
    max_examples: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Find attention heads that perform duplicate token attention.

    Duplicate token heads attend from the second occurrence of a name to the
    first occurrence. This is measured across the dataset and heads above
    the threshold are returned.

    Args:
        model: HookedTransformer instance
        dataset: List of IOI examples (dicts with "prompt", "A", "B" keys)
        threshold: Minimum average attention weight to be classified as duplicate token head
        max_examples: Optional limit on examples to analyze

    Returns:
        List of (layer, head) tuples for duplicate token heads

    Example:
        >>> import json
        >>> with open("data/ioi_abba.json") as f:
        ...     dataset = json.load(f)
        >>> duplicate_heads = find_duplicate_token_heads(model, dataset, threshold=0.5)
        >>> print(f"Found {len(duplicate_heads)} duplicate token heads")
    """
    if max_examples is not None:
        dataset = dataset[:max_examples]

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Store attention scores for each head across examples
    head_scores = defaultdict(list)

    print(f"Analyzing {len(dataset)} examples for duplicate token heads...")

    for i, example in enumerate(dataset):
        if (i + 1) % 20 == 0:
            print(f"  Processing example {i + 1}/{len(dataset)}...")

        prompt = example["prompt"]
        name_a = example["A"]

        # Get positions of A
        a_positions = get_token_positions(model, prompt, name_a)

        # Skip if A doesn't appear twice
        if len(a_positions) < 2:
            continue

        # Get tokens and run model once
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        first_a = a_positions[0]
        second_a = a_positions[-1]

        # Check each head
        for layer in range(n_layers):
            attn_pattern = cache["pattern", layer]  # (batch, n_heads, seq, seq)

            for head in range(n_heads):
                # Get attention from second A to first A
                attn_weight = attn_pattern[0, head, second_a, first_a].item()
                head_scores[(layer, head)].append(attn_weight)

    # Compute average scores and filter by threshold
    duplicate_heads = []

    print(f"\nDuplicate Token Head Detection (threshold = {threshold}):")
    print("=" * 60)

    for (layer, head), scores in sorted(head_scores.items()):
        avg_score = np.mean(scores)

        if avg_score >= threshold:
            duplicate_heads.append((layer, head))
            print(f"  Layer {layer:2d}, Head {head:2d}: avg attention = {avg_score:.3f}")

    print(f"\nFound {len(duplicate_heads)} duplicate token heads")

    return duplicate_heads


def find_s_inhibition_heads(
    model: HookedTransformer,
    dataset: List[Dict],
    threshold: float = 0.3,
    max_examples: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Find attention heads that perform S-inhibition.

    S-inhibition heads attend from the final position to the subject (S) token
    to suppress it in the output.

    Args:
        model: HookedTransformer instance
        dataset: List of IOI examples
        threshold: Minimum average attention weight
        max_examples: Optional limit on examples

    Returns:
        List of (layer, head) tuples for S-inhibition heads
    """
    if max_examples is not None:
        dataset = dataset[:max_examples]

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_scores = defaultdict(list)

    print(f"Analyzing {len(dataset)} examples for S-inhibition heads...")

    for i, example in enumerate(dataset):
        if (i + 1) % 20 == 0:
            print(f"  Processing example {i + 1}/{len(dataset)}...")

        prompt = example["prompt"]
        name_s = example["A"]  # S is A in ABBA

        # Get positions of S
        s_positions = get_token_positions(model, prompt, name_s)

        if len(s_positions) == 0:
            continue

        # Get tokens and run model once
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Check each head
        for layer in range(n_layers):
            attn_pattern = cache["pattern", layer]  # (batch, n_heads, seq, seq)

            for head in range(n_heads):
                # Get attention from end position to S positions
                attn_weights = []
                for s_pos in s_positions:
                    weight = attn_pattern[0, head, -1, s_pos].item()
                    attn_weights.append(weight)

                # Average across S positions
                avg_weight = np.mean(attn_weights)
                head_scores[(layer, head)].append(avg_weight)

    # Filter by threshold
    s_inhibition_heads = []

    print(f"\nS-Inhibition Head Detection (threshold = {threshold}):")
    print("=" * 60)

    for (layer, head), scores in sorted(head_scores.items()):
        avg_score = np.mean(scores)

        if avg_score >= threshold:
            s_inhibition_heads.append((layer, head))
            print(f"  Layer {layer:2d}, Head {head:2d}: avg attention = {avg_score:.3f}")

    print(f"\nFound {len(s_inhibition_heads)} S-inhibition heads")

    return s_inhibition_heads


def find_name_mover_heads(
    model: HookedTransformer,
    dataset: List[Dict],
    threshold: float = 0.3,
    max_examples: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Find attention heads that move the IO name to output.

    Name mover heads attend from the final position to the indirect object (IO)
    token and move it to the output.

    Args:
        model: HookedTransformer instance
        dataset: List of IOI examples
        threshold: Minimum average attention weight
        max_examples: Optional limit on examples

    Returns:
        List of (layer, head) tuples for name mover heads
    """
    if max_examples is not None:
        dataset = dataset[:max_examples]

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_scores = defaultdict(list)

    print(f"Analyzing {len(dataset)} examples for name mover heads...")

    for i, example in enumerate(dataset):
        if (i + 1) % 20 == 0:
            print(f"  Processing example {i + 1}/{len(dataset)}...")

        prompt = example["prompt"]
        name_io = example["B"]  # IO is B in ABBA

        # Get positions of IO
        io_positions = get_token_positions(model, prompt, name_io)

        if len(io_positions) == 0:
            continue

        # Get tokens and run model once
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Check each head
        for layer in range(n_layers):
            attn_pattern = cache["pattern", layer]  # (batch, n_heads, seq, seq)

            for head in range(n_heads):
                # Get attention from end position to IO positions
                attn_weights = []
                for io_pos in io_positions:
                    weight = attn_pattern[0, head, -1, io_pos].item()
                    attn_weights.append(weight)

                # Average across IO positions
                avg_weight = np.mean(attn_weights)
                head_scores[(layer, head)].append(avg_weight)

    # Filter by threshold
    name_mover_heads = []

    print(f"\nName Mover Head Detection (threshold = {threshold}):")
    print("=" * 60)

    for (layer, head), scores in sorted(head_scores.items()):
        avg_score = np.mean(scores)

        if avg_score >= threshold:
            name_mover_heads.append((layer, head))
            print(f"  Layer {layer:2d}, Head {head:2d}: avg attention = {avg_score:.3f}")

    print(f"\nFound {len(name_mover_heads)} name mover heads")

    return name_mover_heads


def find_all_ioi_heads(
    model: HookedTransformer,
    dataset_path: str,
    max_examples: Optional[int] = 100,
    duplicate_threshold: float = 0.5,
    s_inhibition_threshold: float = 0.3,
    name_mover_threshold: float = 0.3
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Find all three types of IOI circuit heads.

    Args:
        model: HookedTransformer instance
        dataset_path: Path to IOI dataset JSON
        max_examples: Maximum examples to analyze
        duplicate_threshold: Threshold for duplicate token heads
        s_inhibition_threshold: Threshold for S-inhibition heads
        name_mover_threshold: Threshold for name mover heads

    Returns:
        Dictionary with:
            - "duplicate_token_heads": List of (layer, head) tuples
            - "s_inhibition_heads": List of (layer, head) tuples
            - "name_mover_heads": List of (layer, head) tuples
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} examples from {dataset_path}")
    print("=" * 60)

    # Find each type of head
    duplicate_heads = find_duplicate_token_heads(
        model, dataset, threshold=duplicate_threshold, max_examples=max_examples
    )

    print("\n" + "=" * 60)

    s_inhibition_heads = find_s_inhibition_heads(
        model, dataset, threshold=s_inhibition_threshold, max_examples=max_examples
    )

    print("\n" + "=" * 60)

    name_mover_heads = find_name_mover_heads(
        model, dataset, threshold=name_mover_threshold, max_examples=max_examples
    )

    return {
        "duplicate_token_heads": duplicate_heads,
        "s_inhibition_heads": s_inhibition_heads,
        "name_mover_heads": name_mover_heads
    }


if __name__ == "__main__":
    import argparse
    from src.model.model_loader import load_ioi_model

    parser = argparse.ArgumentParser(description="Analyze attention patterns for IOI circuit")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/ioi_abba.json",
        help="Path to dataset JSON"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples to analyze"
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    result = load_ioi_model(device=args.device)
    model = result["model"]

    # Find all IOI heads
    ioi_heads = find_all_ioi_heads(
        model,
        dataset_path=args.dataset,
        max_examples=args.max_examples
    )

    # Print summary
    print("\n" + "=" * 60)
    print("IOI CIRCUIT HEAD SUMMARY")
    print("=" * 60)
    print(f"Duplicate Token Heads: {len(ioi_heads['duplicate_token_heads'])}")
    print(f"S-Inhibition Heads: {len(ioi_heads['s_inhibition_heads'])}")
    print(f"Name Mover Heads: {len(ioi_heads['name_mover_heads'])}")
