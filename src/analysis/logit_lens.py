"""
Logit Lens Analysis for IOI Circuit

Implements the "logit lens" technique (nostalgebraist, 2020) to visualize
how the model's prediction evolves through each layer.

For the IOI task, this shows when the model "decides" between IO and S tokens,
revealing which layers are most important for the task.

References:
- nostalgebraist: "interpreting GPT: the logit lens" (2020)
- IOI paper (Wang et al. 2022): Used layer-wise analysis
"""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from pathlib import Path


def compute_layer_wise_logit_diff(
    model: HookedTransformer,
    tokens: torch.Tensor,
    io_token_id: int,
    s_token_id: int,
    position: int = -1
) -> Dict[str, np.ndarray]:
    """
    Compute logit difference (IO - S) at each layer using logit lens.

    The logit lens technique applies the unembedding matrix after each layer
    to see what the model "thinks" at that point in the computation.

    Args:
        model: HookedTransformer instance
        tokens: Input tokens (batch, seq_len)
        io_token_id: Token ID for indirect object (correct answer)
        s_token_id: Token ID for subject (incorrect answer)
        position: Position to measure (-1 for last token)

    Returns:
        Dictionary with:
            - layer_logit_diffs: Array of logit diffs at each layer (length n_layers+1)
            - layer_io_logits: Array of IO logits at each layer
            - layer_s_logits: Array of S logits at each layer
            - final_logit_diff: Final logit difference

    Example:
        >>> results = compute_layer_wise_logit_diff(
        ...     model, tokens, bob_id, alice_id
        ... )
        >>> print(f"Layer 0 logit diff: {results['layer_logit_diffs'][0]:.3f}")
        >>> print(f"Layer 11 logit diff: {results['layer_logit_diffs'][11]:.3f}")
    """
    n_layers = model.cfg.n_layers

    # Run model with cache
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    # Get unembed matrix and layernorm parameters
    W_U = model.W_U  # (d_model, vocab_size)
    ln_f_scale = model.ln_final.w  # Final layernorm scale
    ln_f_bias = model.ln_final.b   # Final layernorm bias

    layer_logit_diffs = []
    layer_io_logits = []
    layer_s_logits = []

    # Analyze residual stream after each layer
    for layer in range(n_layers + 1):  # +1 for final output
        if layer == 0:
            # After embeddings (layer 0)
            residual = cache["hook_embed"] + cache["hook_pos_embed"]
        elif layer == n_layers:
            # Final output (after all layers + final layernorm)
            residual = cache[f"blocks.{n_layers-1}.hook_resid_post"]
            # Apply final layernorm
            residual = (residual - residual.mean(dim=-1, keepdim=True)) / (residual.std(dim=-1, keepdim=True) + 1e-5)
            residual = residual * ln_f_scale + ln_f_bias
        else:
            # After layer `layer`
            residual = cache[f"blocks.{layer-1}.hook_resid_post"]

        # Project through unembedding to get logits
        # residual shape: (batch, seq, d_model)
        # W_U shape: (d_model, vocab_size)
        layer_logits = residual @ W_U  # (batch, seq, vocab_size)

        # Extract logits at target position
        io_logit = layer_logits[0, position, io_token_id].item()
        s_logit = layer_logits[0, position, s_token_id].item()
        logit_diff = io_logit - s_logit

        layer_logit_diffs.append(logit_diff)
        layer_io_logits.append(io_logit)
        layer_s_logits.append(s_logit)

    return {
        "layer_logit_diffs": np.array(layer_logit_diffs),
        "layer_io_logits": np.array(layer_io_logits),
        "layer_s_logits": np.array(layer_s_logits),
        "final_logit_diff": layer_logit_diffs[-1]
    }


def plot_logit_lens(
    layer_results: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Create visualization of logit lens results.

    Shows how the logit difference evolves through the layers, revealing
    when the model "figures out" the answer.

    Args:
        layer_results: Output from compute_layer_wise_logit_diff
        save_path: Optional path to save figure
        title: Optional custom title

    Example:
        >>> results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)
        >>> plot_logit_lens(results, "results/logit_lens.png")
    """
    layer_logit_diffs = layer_results["layer_logit_diffs"]
    n_layers = len(layer_logit_diffs) - 1  # -1 because we include layer 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Logit difference evolution
    layers = np.arange(len(layer_logit_diffs))
    ax1.plot(layers, layer_logit_diffs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No preference')
    ax1.axhline(y=layer_logit_diffs[-1], color='green', linestyle='--', alpha=0.3,
                label=f'Final: {layer_logit_diffs[-1]:.2f}')

    # Shade important regions
    ax1.axvspan(-0.5, 3.5, alpha=0.1, color='purple', label='Duplicate Token Heads (L0-3)')
    ax1.axvspan(6.5, 8.5, alpha=0.1, color='orange', label='S-Inhibition Heads (L7-8)')
    ax1.axvspan(8.5, 11.5, alpha=0.1, color='green', label='Name Mover Heads (L9-11)')

    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Logit Difference (IO - S)', fontsize=12)
    ax1.set_title('Logit Lens: How Model "Decides" Between IO and S\n(Higher = Stronger preference for IO token)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xticks(layers)
    # layer_logit_diffs has n_layers+1 elements: embed + L0 through L(n_layers-1)
    # So labels are: 'Embed', '0', '1', ..., str(n_layers-1)
    ax1.set_xticklabels(['Embed'] + [str(i) for i in range(n_layers)])

    # Plot 2: Layer-by-layer changes (delta logit diff)
    layer_deltas = np.diff(layer_logit_diffs)
    delta_layers = np.arange(1, len(layer_logit_diffs))

    colors = ['green' if d > 0 else 'red' for d in layer_deltas]
    ax2.bar(delta_layers, layer_deltas, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Shade same regions
    ax2.axvspan(0.5, 3.5, alpha=0.1, color='purple')
    ax2.axvspan(6.5, 8.5, alpha=0.1, color='orange')
    ax2.axvspan(8.5, 11.5, alpha=0.1, color='green')

    ax2.set_xlabel('Layer Transition', fontsize=12)
    ax2.set_ylabel('Change in Logit Diff', fontsize=12)
    ax2.set_title('Layer-by-Layer Changes\n(Green = Increases preference for IO, Red = Decreases)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(delta_layers)
    ax2.set_xticklabels([f'{i-1}→{i}' for i in delta_layers], rotation=45)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved logit lens plot to {save_path}")

    plt.show()


def analyze_logit_lens_for_dataset(
    model: HookedTransformer,
    dataset_path: str,
    max_examples: int = 50
) -> Dict[str, np.ndarray]:
    """
    Compute average logit lens across multiple examples.

    This shows the typical behavior across the dataset, reducing noise
    from individual examples.

    Args:
        model: HookedTransformer instance
        dataset_path: Path to IOI dataset JSON
        max_examples: Number of examples to analyze

    Returns:
        Dictionary with:
            - mean_logit_diffs: Average logit diff at each layer
            - std_logit_diffs: Standard deviation at each layer
            - all_logit_diffs: Array of all individual results (examples x layers)

    Example:
        >>> results = analyze_logit_lens_for_dataset(
        ...     model, "data/ioi_abba.json", max_examples=100
        ... )
        >>> plot_logit_lens(
        ...     {"layer_logit_diffs": results["mean_logit_diffs"]},
        ...     "results/logit_lens_avg.png"
        ... )
    """
    import json

    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    examples = dataset[:max_examples]
    n_layers = model.cfg.n_layers + 1  # +1 for embeddings

    all_logit_diffs = []

    print(f"Computing logit lens for {len(examples)} examples...")
    for i, example in enumerate(examples):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(examples)} examples...")

        # Get tokens
        tokens = model.to_tokens(example["prompt"])
        io_token_id = model.to_single_token(" " + example["io_name"])
        s_token_id = model.to_single_token(" " + example["s_name"])

        # Compute logit lens
        results = compute_layer_wise_logit_diff(
            model, tokens, io_token_id, s_token_id
        )

        all_logit_diffs.append(results["layer_logit_diffs"])

    # Convert to array and compute statistics
    all_logit_diffs = np.array(all_logit_diffs)  # (n_examples, n_layers)
    mean_logit_diffs = np.mean(all_logit_diffs, axis=0)
    std_logit_diffs = np.std(all_logit_diffs, axis=0)

    print(f"\nCompleted logit lens analysis:")
    print(f"  Layer 0 (embeddings): {mean_logit_diffs[0]:.3f} ± {std_logit_diffs[0]:.3f}")
    print(f"  Layer {n_layers-2} (final): {mean_logit_diffs[-1]:.3f} ± {std_logit_diffs[-1]:.3f}")
    print(f"  Total change: {mean_logit_diffs[-1] - mean_logit_diffs[0]:.3f}")

    return {
        "mean_logit_diffs": mean_logit_diffs,
        "std_logit_diffs": std_logit_diffs,
        "all_logit_diffs": all_logit_diffs
    }


if __name__ == "__main__":
    from src.model.model_loader import load_ioi_model

    print("Testing Logit Lens")
    print("=" * 80)

    # Load model
    result = load_ioi_model(device="cuda" if torch.cuda.is_available() else "cpu")
    model = result["model"]

    # Test on example
    prompt = "When Alice and Bob went to the store, Alice gave a bottle to"
    tokens = model.to_tokens(prompt)
    bob_id = model.to_single_token(" Bob")
    alice_id = model.to_single_token(" Alice")

    print(f"\nPrompt: {prompt}")
    print(f"IO token: Bob (id={bob_id})")
    print(f"S token: Alice (id={alice_id})")

    # Compute logit lens
    results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)

    print("\nLogit differences by layer:")
    for i, logit_diff in enumerate(results["layer_logit_diffs"]):
        layer_name = "Embed" if i == 0 else f"L{i-1}" if i < len(results["layer_logit_diffs"])-1 else "Final"
        print(f"  {layer_name:6s}: {logit_diff:6.3f}")

    # Plot
    plot_logit_lens(results)
