"""
Unit tests for IOI baseline metrics computation.
"""

import json
import pytest
import torch
import tempfile
from pathlib import Path

from src.analysis.ioi_baseline import (
    run_baseline,
    compute_logit_diff,
    get_token_positions,
    get_io_and_s_positions,
    save_baseline_results,
    analyze_errors
)
from src.model.model_loader import load_ioi_model
from src.data.dataset import generate_ioi_dataset


class TestIOIBaseline:
    """Test suite for IOI baseline metrics."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a small sample dataset for testing."""
        # Generate small dataset
        dataset_path = tmp_path / "test_ioi.json"
        generate_ioi_dataset(n_examples=10, template="ABBA", seed=42)

        # Move to tmp location
        import shutil
        shutil.copy("data/ioi_abba.json", dataset_path)

        return str(dataset_path)

    @pytest.fixture
    def model(self):
        """Load model once for all tests."""
        result = load_ioi_model(device="cpu")
        return result["model"]

    def test_get_token_positions(self, model):
        """Test finding token positions in prompt."""
        prompt = "When Alice and Bob went to the store, Alice gave a bottle to"

        # Find Alice positions
        alice_positions = get_token_positions(model, prompt, "Alice")
        assert len(alice_positions) == 2, "Alice should appear twice"

        # Find Bob positions
        bob_positions = get_token_positions(model, prompt, "Bob")
        assert len(bob_positions) == 1, "Bob should appear once"

        # Bob should be between the two Alice occurrences
        assert alice_positions[0] < bob_positions[0] < alice_positions[1]

    def test_get_io_and_s_positions(self, model):
        """Test extracting IO and S positions."""
        prompt = "When Alice and Bob went to the store, Alice gave a bottle to"

        # In ABBA: A=Alice (S), B=Bob (IO)
        io_pos, s_pos = get_io_and_s_positions(model, prompt, "Bob", "Alice")

        # Verify positions are valid
        assert io_pos >= 0, "IO position should be non-negative"
        assert s_pos >= 0, "S position should be non-negative"
        assert io_pos != s_pos, "IO and S positions should be different"

    def test_get_positions_missing_name_raises_error(self, model):
        """Test that missing name raises ValueError."""
        prompt = "When Alice and Bob went to the store"

        # Charlie is not in the prompt
        with pytest.raises(ValueError, match="not found"):
            get_io_and_s_positions(model, prompt, "Charlie", "Alice")

    def test_compute_logit_diff(self):
        """Test logit difference computation."""
        # Create sample logits
        batch_size, seq_len, vocab_size = 1, 10, 50257
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Set specific values for testing
        io_token_id = 100
        s_token_id = 200
        logits[0, -1, io_token_id] = 5.0
        logits[0, -1, s_token_id] = 3.0

        # Compute logit diff
        diff = compute_logit_diff(logits, io_token_id, s_token_id, position=-1)

        # Should be 5.0 - 3.0 = 2.0
        assert abs(diff - 2.0) < 1e-5, "Logit diff should be 2.0"

    def test_compute_logit_diff_negative(self):
        """Test logit diff when S has higher logit than IO."""
        batch_size, seq_len, vocab_size = 1, 10, 50257
        logits = torch.randn(batch_size, seq_len, vocab_size)

        io_token_id = 100
        s_token_id = 200
        logits[0, -1, io_token_id] = 3.0
        logits[0, -1, s_token_id] = 5.0

        diff = compute_logit_diff(logits, io_token_id, s_token_id, position=-1)

        # Should be 3.0 - 5.0 = -2.0
        assert abs(diff - (-2.0)) < 1e-5, "Logit diff should be -2.0"

    def test_run_baseline_structure(self, sample_dataset):
        """Test that run_baseline returns correct structure."""
        results = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=5
        )

        # Check required keys
        required_keys = [
            "accuracy",
            "mean_logit_diff",
            "median_logit_diff",
            "std_logit_diff",
            "num_examples",
            "num_correct",
            "per_example_results"
        ]

        for key in required_keys:
            assert key in results, f"Results should contain '{key}'"

    def test_run_baseline_metrics_range(self, sample_dataset):
        """Test that metrics are in expected ranges."""
        results = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=5
        )

        # Accuracy should be between 0 and 1
        assert 0.0 <= results["accuracy"] <= 1.0, "Accuracy should be in [0, 1]"

        # Number of examples should match
        assert results["num_examples"] == 5, "Should have 5 examples"

        # Num correct should be consistent with accuracy
        expected_correct = int(results["accuracy"] * results["num_examples"])
        assert abs(results["num_correct"] - expected_correct) <= 1

    def test_run_baseline_per_example_structure(self, sample_dataset):
        """Test per-example results structure."""
        results = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=3
        )

        per_ex = results["per_example_results"]
        assert len(per_ex) == 3, "Should have 3 per-example results"

        # Check first example structure
        example = per_ex[0]
        required_fields = [
            "example_idx",
            "prompt",
            "io_name",
            "s_name",
            "io_token_id",
            "s_token_id",
            "logit_diff",
            "io_logit",
            "s_logit",
            "correct",
            "predicted_token_id",
            "top_k_token_ids",
            "top_k_logits"
        ]

        for field in required_fields:
            assert field in example, f"Example should contain '{field}'"

    def test_run_baseline_deterministic(self, sample_dataset):
        """Test that baseline is deterministic with same dataset."""
        results1 = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=5
        )

        results2 = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=5
        )

        # Metrics should be identical
        assert results1["accuracy"] == results2["accuracy"]
        assert results1["mean_logit_diff"] == results2["mean_logit_diff"]
        assert results1["median_logit_diff"] == results2["median_logit_diff"]

    def test_run_baseline_max_examples(self, sample_dataset):
        """Test that max_examples limits evaluation."""
        results = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=3
        )

        assert results["num_examples"] == 3, "Should only process 3 examples"
        assert len(results["per_example_results"]) == 3

    def test_save_baseline_results(self, sample_dataset, tmp_path):
        """Test saving results to JSON."""
        results = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=5
        )

        output_path = tmp_path / "results.json"
        save_baseline_results(results, str(output_path))

        # Check file exists
        assert output_path.exists(), "Results file should exist"

        # Load and verify
        with open(output_path, 'r') as f:
            loaded = json.load(f)

        assert loaded["accuracy"] == results["accuracy"]
        assert loaded["num_examples"] == results["num_examples"]

    def test_analyze_errors(self, sample_dataset):
        """Test error analysis function."""
        results = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=10
        )

        # Run error analysis (should not raise)
        errors = analyze_errors(results, top_n=5)

        # Check structure
        assert isinstance(errors, list), "Should return list of errors"

        # All errors should have correct=False
        for error in errors:
            assert not error["correct"], "All returned examples should be errors"

    def test_baseline_logit_diff_calculation(self, model):
        """Test that logit diff is calculated correctly for a simple case."""
        # Create a simple prompt
        prompt = "When Alice and Bob went to the store, Alice gave a bottle to"

        # Get token IDs
        alice_id = model.to_single_token("Alice")
        bob_id = model.to_single_token("Bob")

        # Run model
        with torch.no_grad():
            logits = model(prompt)

        # Compute logit diff
        diff = compute_logit_diff(logits, bob_id, alice_id, position=-1)

        # Logit diff should be a finite number
        assert isinstance(diff, float), "Logit diff should be float"
        assert not torch.isnan(torch.tensor(diff)), "Logit diff should not be NaN"
        assert not torch.isinf(torch.tensor(diff)), "Logit diff should not be inf"

    def test_baseline_with_different_templates(self, tmp_path):
        """Test baseline with different template variations."""
        # Generate dataset with seed for reproducibility
        generate_ioi_dataset(n_examples=5, template="ABBA", seed=999)
        dataset_path = "data/ioi_abba.json"

        results = run_baseline(
            dataset_path=dataset_path,
            device="cpu",
            max_examples=5
        )

        # Should process all examples
        assert results["num_examples"] == 5

        # All examples should have valid logit diffs
        for example in results["per_example_results"]:
            assert isinstance(example["logit_diff"], float)
            assert not torch.isnan(torch.tensor(example["logit_diff"]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_run_baseline_cuda(self, sample_dataset):
        """Test running baseline on CUDA."""
        results = run_baseline(
            dataset_path=sample_dataset,
            device="cuda",
            max_examples=3
        )

        assert results["num_examples"] == 3
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_top_k_predictions(self, sample_dataset):
        """Test that top-k predictions are captured."""
        results = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=3
        )

        for example in results["per_example_results"]:
            top_k_ids = example["top_k_token_ids"]
            top_k_logits = example["top_k_logits"]

            # Should have 5 predictions
            assert len(top_k_ids) == 5, "Should have top-5 predictions"
            assert len(top_k_logits) == 5, "Should have top-5 logits"

            # Logits should be in descending order
            for i in range(len(top_k_logits) - 1):
                assert top_k_logits[i] >= top_k_logits[i + 1], "Logits should be sorted"

    def test_accuracy_consistency(self, sample_dataset):
        """Test that accuracy matches counted correct predictions."""
        results = run_baseline(
            dataset_path=sample_dataset,
            device="cpu",
            max_examples=10
        )

        # Count correct predictions manually
        num_correct = sum(1 for ex in results["per_example_results"] if ex["correct"])

        assert num_correct == results["num_correct"]
        assert abs(results["accuracy"] - (num_correct / results["num_examples"])) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
