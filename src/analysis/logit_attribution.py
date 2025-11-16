"""
Direct Logit Attribution (DLA) for IOI Analysis

Implements Direct Logit Attribution following ARENA 1.4.1 and
IOI paper (Wang et al. 2022) Section 3.3.

Direct Logit Attribution decomposes the final logit output into contributions
from each component in the model. This reveals which components are responsible
for predicting specific tokens.

Key insight: The residual stream is a sum of all component outputs:
    final_residual = embed + sum(attn_out[layer][head]) + sum(mlp_out[layer])

Each component's contribution to a token's logit is computed by projecting
its output through the unembedding matrix:
    contribution = component_output @ W_U[:, token_id]

From IOI paper, we expect:
- Name mover heads: Large positive contribution to IO token
- S-inhibition heads: Large negative contribution to S token
"""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformer_lens import HookedTransformer


def compute_logit_attribution(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_token_id: int,
    position: int = -1
) -> Dict:
    """
    Compute logit attribution for a target token.

    Decomposes the logit for target_token_id at position into contributions
    from each component (embeddings, attention heads, MLPs).

    Args:
        model: HookedTransformer instance
        tokens: Input tokens of shape (batch, seq_len)
        target_token_id: Token ID to compute attribution for
        position: Position to analyze (default: -1 for last position)

    Returns:
        Dictionary with:
            - embed_contribution: Float, contribution from token embeddings
            - pos_embed_contribution: Float, contribution from positional embeddings
            - head_contributions: Dict mapping (layer, head) to contribution
            - mlp_contributions: Dict mapping layer to contribution
            - total_logit: Float, total logit for target token
            - components_sum: Float, sum of all component contributions (should ≈ total_logit)

    Example:
        >>> tokens = model.to_tokens("When Alice and Bob went to the store, Alice gave a bottle to")
        >>> bob_id = model.to_single_token(" Bob")
        >>> attribution = compute_logit_attribution(model, tokens, bob_id)
        >>> # Find top contributing heads
        >>> sorted_heads = sorted(
        ...     attribution['head_contributions'].items(),
        ...     key=lambda x: x[1],
        ...     reverse=True
        ... )
        >>> print(f"Top head: L{sorted_heads[0][0][0]}H{sorted_heads[0][0][1]}: {sorted_heads[0][1]:.3f}")
    """
    # Run model with cache
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    # Get total logit for target token
    total_logit = logits[0, position, target_token_id].item()

    # Get unembedding matrix
    W_U = model.W_U  # Shape: (d_model, d_vocab)

    # Get unembedding vector for target token
    unembed_vector = W_U[:, target_token_id]  # Shape: (d_model,)

    # 1. Embedding contribution
    embed_out = cache["hook_embed"]  # Shape: (batch, seq, d_model)
    embed_contribution = (embed_out[0, position] @ unembed_vector).item()

    # 2. Positional embedding contribution
    pos_embed_out = cache["hook_pos_embed"]  # Shape: (batch, seq, d_model)
    pos_embed_contribution = (pos_embed_out[0, position] @ unembed_vector).item()

    # 3. Attention head contributions
    head_contributions = {}
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    for layer in range(n_layers):
        # Get attention head outputs
        # cache["result", layer] has shape (batch, seq, n_heads, d_head)
        attn_out = cache["result", layer]  # (batch, seq, n_heads, d_head)

        # Get output weights for this layer
        W_O = model.W_O[layer]  # Shape: (n_heads, d_head, d_model)

        for head in range(n_heads):
            # Get this head's output at the position
            head_out = attn_out[0, position, head, :]  # Shape: (d_head,)

            # Project through W_O to get contribution to residual stream
            head_residual = head_out @ W_O[head]  # Shape: (d_model,)

            # Project through unembedding to get contribution to logit
            head_contribution = (head_residual @ unembed_vector).item()

            head_contributions[(layer, head)] = head_contribution

    # 4. MLP contributions
    mlp_contributions = {}

    for layer in range(n_layers):
        # Get MLP output
        mlp_out = cache["mlp_out", layer]  # Shape: (batch, seq, d_model)

        # Project through unembedding
        mlp_contribution = (mlp_out[0, position] @ unembed_vector).item()

        mlp_contributions[layer] = mlp_contribution

    # Compute sum of all components
    components_sum = (
        embed_contribution +
        pos_embed_contribution +
        sum(head_contributions.values()) +
        sum(mlp_contributions.values())
    )

    return {
        "embed_contribution": embed_contribution,
        "pos_embed_contribution": pos_embed_contribution,
        "head_contributions": head_contributions,
        "mlp_contributions": mlp_contributions,
        "total_logit": total_logit,
        "components_sum": components_sum,
        "residual": total_logit - components_sum  # Should be small
    }


def compare_io_vs_s_attribution(
    model: HookedTransformer,
    tokens: torch.Tensor,
    io_token_id: int,
    s_token_id: int,
    position: int = -1
) -> Dict:
    """
    Compare logit attribution for IO token vs S token.

    This reveals the circuit mechanism:
    - Name mover heads: Positive contribution to IO, negative to S
    - S-inhibition heads: Negative contribution to S (suppress it)

    Args:
        model: HookedTransformer instance
        tokens: Input tokens
        io_token_id: Indirect object token ID (correct answer)
        s_token_id: Subject token ID (incorrect answer)
        position: Position to analyze

    Returns:
        Dictionary with:
            - io_attribution: Attribution dict for IO token
            - s_attribution: Attribution dict for S token
            - head_differences: Dict mapping (layer, head) to (IO_contrib - S_contrib)
            - top_io_heads: List of (layer, head, contribution) for IO
            - top_s_suppression_heads: List of (layer, head, -S_contribution) for S suppression
            - logit_diff: IO_logit - S_logit

    Example:
        >>> comparison = compare_io_vs_s_attribution(
        ...     model, tokens, bob_id, alice_id
        ... )
        >>> print(f"Logit difference: {comparison['logit_diff']:.3f}")
        >>> print("Top IO contributors:")
        >>> for layer, head, contrib in comparison['top_io_heads'][:5]:
        ...     print(f"  L{layer}H{head}: {contrib:.3f}")
    """
    # Compute attribution for both tokens
    io_attribution = compute_logit_attribution(model, tokens, io_token_id, position)
    s_attribution = compute_logit_attribution(model, tokens, s_token_id, position)

    # Compute differences for each head
    head_differences = {}
    for (layer, head) in io_attribution["head_contributions"].keys():
        io_contrib = io_attribution["head_contributions"][(layer, head)]
        s_contrib = s_attribution["head_contributions"][(layer, head)]
        head_differences[(layer, head)] = io_contrib - s_contrib

    # Find top heads for IO (positive contribution)
    io_heads = [
        (layer, head, contrib)
        for (layer, head), contrib in io_attribution["head_contributions"].items()
    ]
    top_io_heads = sorted(io_heads, key=lambda x: x[2], reverse=True)

    # Find top heads for S suppression (negative contribution to S)
    s_heads = [
        (layer, head, -contrib)  # Negate so positive means suppressing S
        for (layer, head), contrib in s_attribution["head_contributions"].items()
    ]
    top_s_suppression_heads = sorted(s_heads, key=lambda x: x[2], reverse=True)

    # Compute logit difference
    logit_diff = io_attribution["total_logit"] - s_attribution["total_logit"]

    return {
        "io_attribution": io_attribution,
        "s_attribution": s_attribution,
        "head_differences": head_differences,
        "top_io_heads": top_io_heads,
        "top_s_suppression_heads": top_s_suppression_heads,
        "logit_diff": logit_diff
    }


def plot_logit_attribution(
    comparison: Dict,
    save_path: Optional[str] = None,
    top_n: int = 15
):
    """
    Create visualization of logit attribution.

    Plots two bar charts:
    1. Top heads contributing to IO token (should include name movers)
    2. Top heads suppressing S token (should include S-inhibition heads)

    Args:
        comparison: Output from compare_io_vs_s_attribution
        save_path: Optional path to save figure
        top_n: Number of top heads to show

    Example:
        >>> comparison = compare_io_vs_s_attribution(model, tokens, bob_id, alice_id)
        >>> plot_logit_attribution(comparison, "results/logit_attribution.png")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Top IO contributors
    top_io = comparison["top_io_heads"][:top_n]
    io_labels = [f"L{layer}H{head}" for layer, head, _ in top_io]
    io_values = [contrib for _, _, contrib in top_io]

    colors_io = ['green' if v > 0 else 'red' for v in io_values]
    ax1.barh(range(len(io_labels)), io_values, color=colors_io)
    ax1.set_yticks(range(len(io_labels)))
    ax1.set_yticklabels(io_labels)
    ax1.set_xlabel('Contribution to IO Token Logit')
    ax1.set_title(f'Top {top_n} Heads Contributing to IO Token\n(Name Mover Heads)', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Top S suppressors
    top_s_suppress = comparison["top_s_suppression_heads"][:top_n]
    s_labels = [f"L{layer}H{head}" for layer, head, _ in top_s_suppress]
    s_values = [contrib for _, _, contrib in top_s_suppress]

    colors_s = ['green' if v > 0 else 'red' for v in s_values]
    ax2.barh(range(len(s_labels)), s_values, color=colors_s)
    ax2.set_yticks(range(len(s_labels)))
    ax2.set_yticklabels(s_labels)
    ax2.set_xlabel('Negative Contribution to S Token Logit')
    ax2.set_title(f'Top {top_n} Heads Suppressing S Token\n(S-Inhibition Heads)', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()

    # Add logit diff info
    logit_diff = comparison["logit_diff"]
    fig.suptitle(f'Direct Logit Attribution Analysis\nLogit Difference (IO - S): {logit_diff:.3f}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attribution plot to {save_path}")

    plt.show()


def analyze_circuit_with_dla(
    model: HookedTransformer,
    tokens: torch.Tensor,
    io_token_id: int,
    s_token_id: int,
    circuit_heads: Optional[Dict] = None,
    position: int = -1
) -> Dict:
    """
    Analyze circuit using Direct Logit Attribution.

    If circuit_heads is provided, compares attribution of circuit heads vs
    non-circuit heads to validate the circuit.

    Args:
        model: HookedTransformer instance
        tokens: Input tokens
        io_token_id: IO token ID
        s_token_id: S token ID
        circuit_heads: Optional dict with 'name_mover_heads' and 's_inhibition_heads'
        position: Position to analyze

    Returns:
        Dictionary with:
            - comparison: Output from compare_io_vs_s_attribution
            - circuit_analysis: If circuit_heads provided, analysis of circuit vs non-circuit

    Example:
        >>> circuit = discover_ioi_circuit(model, "data/ioi_abba.json")
        >>> analysis = analyze_circuit_with_dla(
        ...     model, tokens, bob_id, alice_id,
        ...     circuit_heads=circuit
        ... )
        >>> print(f"Circuit contribution to logit diff: {analysis['circuit_analysis']['circuit_logit_diff']:.3f}")
    """
    # Compute basic comparison
    comparison = compare_io_vs_s_attribution(
        model, tokens, io_token_id, s_token_id, position
    )

    result = {
        "comparison": comparison
    }

    # If circuit heads provided, analyze circuit vs non-circuit
    if circuit_heads is not None:
        circuit_set = set()
        if "name_mover_heads" in circuit_heads:
            circuit_set.update(circuit_heads["name_mover_heads"])
        if "s_inhibition_heads" in circuit_heads:
            circuit_set.update(circuit_heads["s_inhibition_heads"])
        if "duplicate_token_heads" in circuit_heads:
            circuit_set.update(circuit_heads["duplicate_token_heads"])

        # Compute contributions from circuit vs non-circuit heads
        circuit_io_contrib = 0.0
        circuit_s_contrib = 0.0
        non_circuit_io_contrib = 0.0
        non_circuit_s_contrib = 0.0

        for (layer, head), io_contrib in comparison["io_attribution"]["head_contributions"].items():
            s_contrib = comparison["s_attribution"]["head_contributions"][(layer, head)]

            if (layer, head) in circuit_set:
                circuit_io_contrib += io_contrib
                circuit_s_contrib += s_contrib
            else:
                non_circuit_io_contrib += io_contrib
                non_circuit_s_contrib += s_contrib

        circuit_logit_diff = circuit_io_contrib - circuit_s_contrib
        non_circuit_logit_diff = non_circuit_io_contrib - non_circuit_s_contrib
        total_logit_diff = comparison["logit_diff"]

        result["circuit_analysis"] = {
            "circuit_heads_count": len(circuit_set),
            "circuit_io_contrib": circuit_io_contrib,
            "circuit_s_contrib": circuit_s_contrib,
            "circuit_logit_diff": circuit_logit_diff,
            "non_circuit_io_contrib": non_circuit_io_contrib,
            "non_circuit_s_contrib": non_circuit_s_contrib,
            "non_circuit_logit_diff": non_circuit_logit_diff,
            "total_logit_diff": total_logit_diff,
            "circuit_percentage": circuit_logit_diff / total_logit_diff if total_logit_diff != 0 else 0.0
        }

        print("\nCircuit Analysis with Direct Logit Attribution:")
        print("=" * 60)
        print(f"Circuit heads: {result['circuit_analysis']['circuit_heads_count']}")
        print(f"Total logit diff (IO - S): {total_logit_diff:.3f}")
        print(f"  Circuit contribution: {circuit_logit_diff:.3f} ({result['circuit_analysis']['circuit_percentage']:.1%})")
        print(f"  Non-circuit contribution: {non_circuit_logit_diff:.3f}")

    return result


if __name__ == "__main__":
    from src.model.model_loader import load_ioi_model

    print("Testing Direct Logit Attribution")
    print("=" * 80)

    # Load model
    result = load_ioi_model(device="cpu")
    model = result["model"]

    # Test prompt
    prompt = "When Alice and Bob went to the store, Alice gave a bottle to"
    tokens = model.to_tokens(prompt)

    # Get token IDs
    bob_id = model.to_single_token(" Bob")
    alice_id = model.to_single_token(" Alice")

    print(f"\nPrompt: {prompt}")
    print(f"IO token: Bob (id: {bob_id})")
    print(f"S token: Alice (id: {alice_id})")

    # Compute attribution
    print("\n" + "=" * 80)
    print("Computing Direct Logit Attribution...")

    comparison = compare_io_vs_s_attribution(
        model, tokens, bob_id, alice_id
    )

    print(f"\nLogit difference (IO - S): {comparison['logit_diff']:.3f}")

    # Show top IO contributors
    print("\nTop 10 heads contributing to IO token:")
    for i, (layer, head, contrib) in enumerate(comparison["top_io_heads"][:10]):
        print(f"  {i+1}. L{layer}H{head:2d}: {contrib:6.3f}")

    # Show top S suppressors
    print("\nTop 10 heads suppressing S token:")
    for i, (layer, head, contrib) in enumerate(comparison["top_s_suppression_heads"][:10]):
        print(f"  {i+1}. L{layer}H{head:2d}: {contrib:6.3f}")

    # Create visualization
    print("\n" + "=" * 80)
    print("Creating visualization...")
    plot_logit_attribution(comparison, save_path="results/logit_attribution_test.png")

    print("\nDirect Logit Attribution test complete!")
