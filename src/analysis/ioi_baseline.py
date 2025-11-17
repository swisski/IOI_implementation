"""
IOI Baseline Metrics Computation

Computes baseline performance metrics for GPT2-small on the IOI task,
following the Indirect Object Identification paper (Wang et al. 2022)
and ARENA 1.4.1 tutorial section on "Running the model".

Key metrics:
- accuracy: Fraction where model's top prediction is the IO token (correct answer)
- mean_logit_diff: Average of logit(IO) - logit(S) across examples
- median_logit_diff: Median logit difference
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from transformer_lens import HookedTransformer

from src.model.model_loader import load_ioi_model


def get_token_positions(
    model: HookedTransformer,
    prompt: str,
    name: str
) -> List[int]:
    """
    Find all positions where a name appears in the tokenized prompt.

    Args:
        model: HookedTransformer instance
        prompt: Input prompt string
        name: Name to find in the prompt

    Returns:
        List of token positions where the name appears
    """
    tokens = model.to_tokens(prompt)

    # In GPT-2 tokenization, tokens in the middle of text have a leading space
    # Try to find token ID for both " Name" and "Name"
    name_token_ids = []

    # Try with leading space (most common in middle of text)
    try:
        name_token_ids.append(model.to_single_token(" " + name))
    except (KeyError, ValueError):
        pass

    # Try without leading space (start of text or after punctuation)
    try:
        name_token_ids.append(model.to_single_token(name))
    except (KeyError, ValueError):
        pass

    if not name_token_ids:
        raise ValueError(f"Could not tokenize name '{name}' as single token")

    # Find all positions where any of these tokens appear
    positions = []
    for token_id in name_token_ids:
        pos = (tokens[0] == token_id).nonzero(as_tuple=True)[0].tolist()
        positions.extend(pos)

    # Sort and remove duplicates
    positions = sorted(list(set(positions)))
    return positions


def get_io_and_s_positions(
    model: HookedTransformer,
    prompt: str,
    name_io: str,
    name_s: str
) -> tuple:
    """
    Get the final token positions for IO and S names in the prompt.

    For ABBA templates: "When [S] and [IO] went to the store, [S] gave a bottle to"
    - IO (B) appears once at position 1 (after first S)
    - S (A) appears twice at positions 0 and 2 (beginning and second occurrence)

    We want the LAST position where each name appears before the final IO position.

    Args:
        model: HookedTransformer instance
        prompt: Input prompt string
        name_io: Indirect object name (correct answer, B in ABBA)
        name_s: Subject name (A in ABBA)

    Returns:
        Tuple of (io_position, s_position)
    """
    # Get all positions for each name
    io_positions = get_token_positions(model, prompt, name_io)
    s_positions = get_token_positions(model, prompt, name_s)

    # For standard IOI, we want:
    # - IO position: where IO name appears (typically once)
    # - S position: last occurrence of S name (the second A in ABBA)

    if not io_positions:
        raise ValueError(f"IO name '{name_io}' not found in prompt")
    if not s_positions:
        raise ValueError(f"S name '{name_s}' not found in prompt")

    # Get the last occurrence of each
    io_pos = io_positions[-1] if len(io_positions) == 1 else io_positions[0]
    s_pos = s_positions[-1]

    return io_pos, s_pos


def compute_logit_diff(
    logits: torch.Tensor,
    io_token_id: int,
    s_token_id: int,
    position: int = -1
) -> float:
    """
    Compute logit difference: logit(IO) - logit(S) at specified position.

    Args:
        logits: Model logits tensor of shape (batch, seq_len, vocab_size)
        io_token_id: Token ID for IO (correct answer)
        s_token_id: Token ID for S (incorrect answer)
        position: Position to extract logits from (default: -1 for last position)

    Returns:
        Logit difference as float
    """
    # Extract logits at the specified position
    position_logits = logits[0, position, :]  # Shape: (vocab_size,)

    # Get logits for IO and S tokens
    io_logit = position_logits[io_token_id].item()
    s_logit = position_logits[s_token_id].item()

    # Compute difference
    logit_diff = io_logit - s_logit

    return logit_diff


def run_baseline(
    model_or_dataset_path,
    dataset_path: Optional[str] = None,
    model_name: str = "gpt2-small",
    device: str = "cpu",
    max_examples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run baseline IOI evaluation on dataset.

    Following ARENA 1.4.1 IOI tutorial and the IOI paper metrics:
    1. Load dataset from JSON
    2. Run model on clean prompts
    3. Extract logits at final position (where IO should be predicted)
    4. Compute logit(IO) - logit(S) for each example
    5. Compute accuracy (fraction where IO is top prediction)

    Args:
        model_or_dataset_path: Either a pre-loaded model or path to dataset JSON
        dataset_path: Path to dataset JSON (if first arg is a model)
        model_name: Model to use (default: "gpt2-small")
        device: Device to run on ("cpu", "cuda", "mps")
        max_examples: Optional limit on number of examples to evaluate

    Returns:
        Dictionary with:
            - accuracy: Fraction where model predicts IO token (correct answer)
            - mean_logit_diff: Average logit(IO) - logit(S)
            - median_logit_diff: Median logit difference
            - std_logit_diff: Standard deviation of logit difference
            - per_example_results: List of dicts with per-example metrics
            - num_examples: Number of examples evaluated

    Example:
        >>> results = run_baseline("data/ioi_abba.json", device="cuda")
        >>> # Or with pre-loaded model:
        >>> results = run_baseline(model, "data/ioi_abba.json")
    """
    # Determine if first arg is a model or dataset path
    if isinstance(model_or_dataset_path, str):
        # First arg is dataset path (old signature)
        dataset_path_to_use = model_or_dataset_path
        model = None
    else:
        # First arg is a model
        model = model_or_dataset_path
        dataset_path_to_use = dataset_path

    # Load dataset
    print(f"Loading dataset from {dataset_path_to_use}...")
    with open(dataset_path_to_use, 'r') as f:
        dataset = json.load(f)

    if max_examples is not None:
        dataset = dataset[:max_examples]

    print(f"Loaded {len(dataset)} examples")

    # Load model if not provided
    if model is None:
        print(f"\nLoading {model_name}...")
        result = load_ioi_model(device=device)
        model = result["model"]

    print("\nRunning baseline evaluation...")

    # Store results
    per_example_results = []
    logit_diffs = []
    correct_predictions = []

    # Process each example
    for i, example in enumerate(dataset):
        if (i + 1) % 50 == 0:
            print(f"  Processing example {i + 1}/{len(dataset)}...")

        prompt = example["prompt"]
        name_io = example["B"]  # IO is B in ABBA template (correct answer)
        name_s = example["A"]   # S is A in ABBA template (subject)

        # Get token IDs for IO and S (with space prefix for proper GPT-2 tokenization)
        io_token_id = model.to_single_token(" " + name_io)
        s_token_id = model.to_single_token(" " + name_s)

        # Run model to get logits
        with torch.no_grad():
            logits = model(prompt)  # Shape: (1, seq_len, vocab_size)

        # Get logits at final position (where we expect IO prediction)
        final_logits = logits[0, -1, :]  # Shape: (vocab_size,)

        # Compute logit difference
        logit_diff = compute_logit_diff(logits, io_token_id, s_token_id, position=-1)
        logit_diffs.append(logit_diff)

        # Check if prediction is correct (IO is top prediction)
        predicted_token_id = torch.argmax(final_logits).item()
        is_correct = (predicted_token_id == io_token_id)
        correct_predictions.append(is_correct)

        # Get top-k predictions for analysis
        top_k = 5
        top_k_logits, top_k_indices = torch.topk(final_logits, top_k)

        # Store per-example results
        per_example_results.append({
            "example_idx": i,
            "prompt": prompt,
            "io_name": name_io,
            "s_name": name_s,
            "io_token_id": io_token_id,
            "s_token_id": s_token_id,
            "logit_diff": logit_diff,
            "io_logit": final_logits[io_token_id].item(),
            "s_logit": final_logits[s_token_id].item(),
            "correct": is_correct,
            "predicted_token_id": predicted_token_id,
            "top_k_token_ids": top_k_indices.tolist(),
            "top_k_logits": top_k_logits.tolist()
        })

    # Compute aggregate metrics
    accuracy = np.mean(correct_predictions)
    mean_logit_diff = np.mean(logit_diffs)
    median_logit_diff = np.median(logit_diffs)
    std_logit_diff = np.std(logit_diffs)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Number of examples: {len(dataset)}")
    print(f"Accuracy: {accuracy:.2%} ({sum(correct_predictions)}/{len(dataset)})")
    print(f"Mean logit diff: {mean_logit_diff:.3f}")
    print(f"Median logit diff: {median_logit_diff:.3f}")
    print(f"Std logit diff: {std_logit_diff:.3f}")
    print("=" * 60)

    return {
        "accuracy": accuracy,
        "mean_logit_diff": mean_logit_diff,
        "median_logit_diff": median_logit_diff,
        "std_logit_diff": std_logit_diff,
        "num_examples": len(dataset),
        "num_correct": sum(correct_predictions),
        "per_example_results": per_example_results
    }


def save_baseline_results(results: Dict[str, Any], output_path: str):
    """
    Save baseline results to JSON file.

    Args:
        results: Results dictionary from run_baseline()
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    results_serializable = {
        "accuracy": float(results["accuracy"]),
        "mean_logit_diff": float(results["mean_logit_diff"]),
        "median_logit_diff": float(results["median_logit_diff"]),
        "std_logit_diff": float(results["std_logit_diff"]),
        "num_examples": int(results["num_examples"]),
        "num_correct": int(results["num_correct"]),
        "per_example_results": results["per_example_results"]
    }

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to {output_path}")


def analyze_errors(results: Dict[str, Any], top_n: int = 10):
    """
    Analyze examples where model made errors.

    Args:
        results: Results dictionary from run_baseline()
        top_n: Number of top errors to display

    Returns:
        List of error examples sorted by logit difference (most negative first)
    """
    # Get incorrect examples
    errors = [ex for ex in results["per_example_results"] if not ex["correct"]]

    # Sort by logit difference (most negative first)
    errors_sorted = sorted(errors, key=lambda x: x["logit_diff"])

    print(f"\n{'=' * 60}")
    print(f"ERROR ANALYSIS (Top {min(top_n, len(errors))} worst examples)")
    print(f"{'=' * 60}")
    print(f"Total errors: {len(errors)} / {results['num_examples']}")
    print()

    for i, error in enumerate(errors_sorted[:top_n]):
        print(f"Error {i + 1}:")
        print(f"  Prompt: {error['prompt']}")
        print(f"  Correct (IO): {error['io_name']} (token {error['io_token_id']})")
        print(f"  Subject (S): {error['s_name']} (token {error['s_token_id']})")
        print(f"  Predicted: token {error['predicted_token_id']}")
        print(f"  Logit diff: {error['logit_diff']:.3f}")
        print(f"  IO logit: {error['io_logit']:.3f}, S logit: {error['s_logit']:.3f}")
        print()

    return errors_sorted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run IOI baseline evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/ioi_abba.json",
        help="Path to dataset JSON file"
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
        default=None,
        help="Maximum number of examples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--analyze-errors",
        action="store_true",
        help="Show detailed error analysis"
    )

    args = parser.parse_args()

    # Run baseline
    results = run_baseline(
        dataset_path=args.dataset,
        device=args.device,
        max_examples=args.max_examples
    )

    # Save results
    save_baseline_results(results, args.output)

    # Analyze errors if requested
    if args.analyze_errors:
        analyze_errors(results)
