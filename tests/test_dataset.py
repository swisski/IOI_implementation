"""
Unit tests for IOI dataset generation.
"""

import json
import pytest
from pathlib import Path
from transformers import GPT2Tokenizer

from src.data.dataset import (
    generate_ioi_dataset,
    generate_abba_prompt,
    generate_abc_prompt,
    verify_single_token_names,
    SINGLE_TOKEN_NAMES,
    ABBA_TEMPLATES,
    ABC_TEMPLATE
)


class TestDatasetGeneration:
    """Test suite for IOI dataset generation."""

    def setup_method(self):
        """Setup for each test method."""
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Clean up any existing test files
        test_path = Path("data/ioi_abba.json")
        if test_path.exists():
            test_path.unlink()

    def test_deterministic_behavior_with_fixed_seed(self):
        """Verify that the same seed produces identical datasets."""
        # Generate dataset twice with same seed
        result1 = generate_ioi_dataset(n_examples=50, template="ABBA", seed=42)
        with open("data/ioi_abba.json", 'r') as f:
            dataset1 = json.load(f)

        result2 = generate_ioi_dataset(n_examples=50, template="ABBA", seed=42)
        with open("data/ioi_abba.json", 'r') as f:
            dataset2 = json.load(f)

        # Datasets should be identical
        assert dataset1 == dataset2, "Same seed should produce identical datasets"

        # Generate with different seed
        result3 = generate_ioi_dataset(n_examples=50, template="ABBA", seed=999)
        with open("data/ioi_abba.json", 'r') as f:
            dataset3 = json.load(f)

        # Datasets should be different
        assert dataset1 != dataset3, "Different seeds should produce different datasets"

    def test_number_of_examples_matches(self):
        """Ensure number of examples matches n_examples parameter."""
        for n in [10, 50, 100, 200]:
            result = generate_ioi_dataset(n_examples=n, template="ABBA", seed=123)

            # Check returned metrics
            assert result["num_examples"] == n, f"Metrics should report {n} examples"

            # Check actual dataset
            with open("data/ioi_abba.json", 'r') as f:
                dataset = json.load(f)
            assert len(dataset) == n, f"Dataset should contain {n} examples"

    def test_prompts_contain_correct_names_in_roles(self):
        """Verify that prompts contain A and B in the correct positions."""
        result = generate_ioi_dataset(n_examples=20, template="ABBA", seed=456)

        with open("data/ioi_abba.json", 'r') as f:
            dataset = json.load(f)

        for example in dataset:
            prompt = example["prompt"]
            name_a = example["A"]
            name_b = example["B"]
            correct = example["correct"]

            # ABBA template: A appears twice, B appears once, correct answer is B
            assert name_a in prompt, f"Name A '{name_a}' should appear in prompt"
            assert name_b in prompt, f"Name B '{name_b}' should appear in prompt"
            assert correct == name_b, f"Correct answer should be B ({name_b})"

            # Check structure: A should appear first, then B, then A again
            first_a_pos = prompt.find(name_a)
            b_pos = prompt.find(name_b)
            second_a_pos = prompt.find(name_a, first_a_pos + 1)

            assert first_a_pos < b_pos, "First A should appear before B"
            assert b_pos < second_a_pos, "B should appear before second A"

            # Verify prompt matches one of the ABBA templates
            matches_template = False
            for template in ABBA_TEMPLATES:
                expected = template.replace("[A]", name_a).replace("[B]", name_b)
                if prompt == expected:
                    matches_template = True
                    break
            assert matches_template, f"Prompt should match one of the ABBA templates: {prompt}"

    def test_all_names_are_single_token(self):
        """Assert that all names in prompts are single-token under GPT2 tokenizer."""
        # First verify our name list
        verify_single_token_names(SINGLE_TOKEN_NAMES)

        # Generate dataset and check all names used
        result = generate_ioi_dataset(n_examples=100, template="ABBA", seed=789)

        with open("data/ioi_abba.json", 'r') as f:
            dataset = json.load(f)

        for example in dataset:
            # Check A, B, and C (if present)
            for name_key in ["A", "B", "C"]:
                if name_key in example:
                    name = example[name_key]
                    tokens = self.tokenizer.encode(name, add_special_tokens=False)
                    assert len(tokens) == 1, f"Name '{name}' should be single token, got {tokens}"

    def test_corrupted_prompts_swap_subject(self):
        """Assert that corrupted prompts swap the subject (A→B) in same template."""
        result = generate_ioi_dataset(n_examples=50, template="ABBA", seed=111)

        with open("data/ioi_abba.json", 'r') as f:
            dataset = json.load(f)

        for example in dataset:
            prompt = example["prompt"]
            corrupted_prompt = example["corrupted_prompt"]
            name_a = example["A"]
            name_b = example["B"]

            # Prompts should be different
            assert prompt != corrupted_prompt, "Clean and corrupted prompts should differ"

            # Clean: A appears twice
            assert prompt.count(name_a) == 2, f"Clean prompt should have 2 occurrences of A ({name_a})"
            assert prompt.count(name_b) == 1, f"Clean prompt should have 1 occurrence of B ({name_b})"

            # Corrupted: B appears twice (subject swapped from A to B)
            assert corrupted_prompt.count(name_a) == 1, f"Corrupted prompt should have 1 occurrence of A ({name_a})"
            assert corrupted_prompt.count(name_b) == 2, f"Corrupted prompt should have 2 occurrences of B ({name_b})"

            # Verify the structure matches an ABBA template with B as subject
            # The second occurrence should be swapped
            # E.g., "When A and B went..., A gave..." → "When A and B went..., B gave..."
            assert name_a in corrupted_prompt, "Corrupted should still contain A"
            assert name_b in corrupted_prompt, "Corrupted should still contain B"

            # Both should end the same way (incomplete sentence ending)
            # The template structure should be preserved
            for template in ABBA_TEMPLATES:
                # Check if clean matches this template
                expected_clean = template.replace("[A]", name_a).replace("[B]", name_b)
                if prompt == expected_clean:
                    # Corrupted should be same template but with second A→B
                    # This is what our new logic does
                    parts = expected_clean.split(name_a)
                    if len(parts) >= 3:
                        expected_corrupted = name_a.join(parts[:2]) + name_b + name_a.join(parts[2:])
                        assert corrupted_prompt == expected_corrupted, \
                            f"Corrupted prompt should swap second A→B in same template"
                        break

    def test_template_functions(self):
        """Test individual template generation functions."""
        # Test ABBA template with first template
        abba = generate_abba_prompt("Alice", "Bob", ABBA_TEMPLATES[0])
        assert abba == "When Alice and Bob went to the store, Alice gave a bottle to"

        # Test ABBA template with second template
        abba2 = generate_abba_prompt("Alice", "Bob", ABBA_TEMPLATES[1])
        assert abba2 == "After Alice and Bob left the house, Alice passed a note to"

        # Test ABC template
        abc = generate_abc_prompt("Alice", "Bob", "Charlie", ABC_TEMPLATE)
        assert abc == "When Alice, Bob, and Charlie met at the store, Charlie gave a bottle to"

    def test_dataset_file_creation(self):
        """Verify that dataset file is created correctly."""
        result = generate_ioi_dataset(n_examples=10, template="ABBA", seed=222)

        # Check file exists
        save_path = Path(result["save_path"])
        assert save_path.exists(), f"Dataset file should exist at {save_path}"

        # Check file is valid JSON
        with open(save_path, 'r') as f:
            dataset = json.load(f)

        assert isinstance(dataset, list), "Dataset should be a list"
        assert len(dataset) > 0, "Dataset should not be empty"

        # Check structure of first example
        example = dataset[0]
        required_fields = ["prompt", "corrupted_prompt", "A", "B", "C", "correct", "template"]
        for field in required_fields:
            assert field in example, f"Example should contain '{field}' field"

    def test_abc_template_only(self):
        """Test generating ABC template only (without ABBA)."""
        result = generate_ioi_dataset(n_examples=30, template="ABC", seed=333)

        # ABC template saves to ioi_abc.json
        with open("data/ioi_abc.json", 'r') as f:
            dataset = json.load(f)

        for example in dataset:
            assert example["template"] == "ABC", "All examples should have ABC template"
            assert "prompt" in example, "Should have prompt field"

            # ABC examples should have A, B, C all different
            name_a = example["A"]
            name_b = example["B"]
            name_c = example["C"]

            assert name_a != name_b, "A and B should be different"
            assert name_b != name_c, "B and C should be different"
            assert name_a != name_c, "A and C should be different"

    def test_invalid_template_raises_error(self):
        """Test that invalid template raises AssertionError."""
        with pytest.raises(AssertionError, match="Template must be"):
            generate_ioi_dataset(n_examples=10, template="INVALID", seed=123)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
