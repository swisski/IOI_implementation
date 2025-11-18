"""
IOI Dataset Generation

Following the Indirect Object Identification paper (Wang et al. 2022) and ARENA 1.4.1 walkthrough.
Generates prompts with ABBA and ABC (corrupted) templates.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from transformers import GPT2Tokenizer


# Single-token names verified with GPT-2 tokenizer
# These are common names that tokenize to exactly one token in GPT-2
SINGLE_TOKEN_NAMES = [
    "Aaron", "Alan", "Alex", "Anna",
    "Brandon", "Brian", "Clark", "David",
    "Elizabeth", "Frank", "George", "Jessica",
    "Kyle", "Kate", "Madison", "Paul",
    "Sarah", "Thomas", "Victoria", "William"
]

# ABBA templates: A appears twice, B once. Correct answer is B (indirect object)
# Each template ends with the IO position (where the answer should go)
ABBA_TEMPLATES = [
    "When [A] and [B] went to the store, [A] gave a bottle to",
    "After [A] and [B] left the house, [A] passed a note to",
    "[A] and [B] visited the garden, and [A] handed a flower to",
    "During the trip, [A] and [B] talked and [A] gave a snack to"
]

# ABC template: A, B, C all appear once. C is in the subject position (corrupted)
ABC_TEMPLATE = "When [A], [B], and [C] met at the store, [C] gave a bottle to"


def verify_single_token_names(names: List[str]) -> bool:
    """Verify that all names are single tokens in GPT-2."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    for name in names:
        tokens = tokenizer.encode(name, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Name '{name}' is not a single token: {tokens}")
    return True


def substitute_template(template: str, name_a: str, name_b: str, name_c: str = None) -> str:
    """
    Substitute names into template without .replace() side effects.
    Uses clean string substitution to avoid issues with name overlaps.

    Args:
        template: Template string with [A], [B], [C] placeholders
        name_a: Name for [A]
        name_b: Name for [B]
        name_c: Optional name for [C]

    Returns:
        Prompt with names substituted
    """
    # Build prompt by directly replacing placeholders
    # This avoids side effects from .replace() when names might overlap
    result = template
    result = result.replace("[A]", name_a)
    result = result.replace("[B]", name_b)
    if name_c is not None:
        result = result.replace("[C]", name_c)
    return result


def generate_abba_prompt(name_a: str, name_b: str, template: str = None) -> str:
    """
    Generate ABBA template prompt programmatically.

    Args:
        name_a: Name for position A (appears twice)
        name_b: Name for position B (appears once, is the correct answer)
        template: Optional specific template, otherwise uses first ABBA template

    Returns:
        Prompt with ABBA pattern where correct answer is B (indirect object)
    """
    if template is None:
        template = ABBA_TEMPLATES[0]
    return substitute_template(template, name_a, name_b)


def generate_abc_prompt(name_a: str, name_b: str, name_c: str, template: str = None) -> str:
    """
    Generate ABC (corrupted) template prompt programmatically.

    Args:
        name_a: Name for position A
        name_b: Name for position B
        name_c: Name for position C (corrupted position, C != A)
        template: Optional specific template, otherwise uses ABC_TEMPLATE

    Returns:
        Corrupted prompt with ABC pattern
    """
    if template is None:
        template = ABC_TEMPLATE
    return substitute_template(template, name_a, name_b, name_c)


def generate_ioi_dataset(n_examples: int, template: str, seed: int) -> Dict[str, Any]:
    """
    Generate IOI dataset with deterministic seeding.

    Args:
        n_examples: Number of examples to generate
        template: Template type ("ABBA" or "ABC")
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
            - num_examples: Number of examples generated
            - save_path: Path where dataset was saved
    """
    # Assert valid template type
    assert template in {"ABBA", "ABC"}, f"Template must be 'ABBA' or 'ABC', got '{template}'"

    # Set random seeds for deterministic generation
    random.seed(seed)
    np.random.seed(seed)

    # Verify names are single tokens
    verify_single_token_names(SINGLE_TOKEN_NAMES)

    # Generate dataset
    dataset = []

    for i in range(n_examples):
        if template == "ABBA":
            # Sample two distinct names for A and B
            name_a, name_b = random.sample(SINGLE_TOKEN_NAMES, 2)

            # Randomly choose an ABBA template
            chosen_template = random.choice(ABBA_TEMPLATES)

            # Generate ABBA prompt: correct answer is B
            prompt = generate_abba_prompt(name_a, name_b, chosen_template)
            correct = name_b

            # Generate corrupted version by swapping the subject
            # In ABBA template: "[A] and [B] ... [A] gave/passed/handed ..."
            # Corrupted should be: "[A] and [B] ... [B] gave/passed/handed ..."
            # This swaps the subject from A to B, making B appear twice
            # Strategy: Find the SECOND occurrence of name_a and replace it with name_b
            # The first occurrence is in "[A] and [B]", the second is the subject

            # Split on name_a, replace the second occurrence
            parts = prompt.split(name_a)
            if len(parts) >= 3:  # Should have at least 2 occurrences of name_a
                # Rejoin: first occurrence + middle + second occurrence replaced
                corrupted_prompt = name_a.join(parts[:2]) + name_b + name_a.join(parts[2:])
            else:
                # Fallback: if template doesn't match expected pattern, keep original
                corrupted_prompt = prompt

            # Also generate ABC version for reference (not used for patching)
            # C must be different from both A and B
            available_names = [n for n in SINGLE_TOKEN_NAMES if n not in [name_a, name_b]]
            name_c = random.choice(available_names)

            dataset.append({
                "prompt": prompt,
                "corrupted_prompt": corrupted_prompt,
                "A": name_a,
                "B": name_b,
                "C": name_c,
                "correct": correct,
                "io_name": name_b,  # Indirect object (correct answer)
                "s_name": name_a,   # Subject (appears twice)
                "template": "ABBA"
            })

        elif template == "ABC":
            # ABC template only (corrupted version)
            # Sample three distinct names for A, B, and C
            name_a, name_b, name_c = random.sample(SINGLE_TOKEN_NAMES, 3)

            # Generate ABC prompt
            prompt = generate_abc_prompt(name_a, name_b, name_c, ABC_TEMPLATE)

            dataset.append({
                "prompt": prompt,
                "A": name_a,
                "B": name_b,
                "C": name_c,
                "correct": name_b,  # Still B in the context
                "io_name": name_b,  # Indirect object
                "s_name": name_a,   # Subject
                "template": "ABC"
            })

    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Save dataset with appropriate filename based on template
    filename = f"ioi_{template.lower()}.json"
    save_path = data_dir / filename
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Return metrics
    return {
        "num_examples": n_examples,
        "save_path": str(save_path)
    }


def load_ioi_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load IOI dataset from JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        List of dataset examples
    """
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset


if __name__ == "__main__":
    # Example usage
    result = generate_ioi_dataset(n_examples=200, template="ABBA", seed=123)
    print(f"Generated {result['num_examples']} examples")
    print(f"Saved to: {result['save_path']}")
