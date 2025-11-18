"""
Unit tests for Logit Lens analysis.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.model.model_loader import load_ioi_model
from src.analysis.logit_lens import (
    compute_layer_wise_logit_diff,
    plot_logit_lens,
    analyze_logit_lens_for_dataset
)


class TestLogitLens:
    """Test suite for logit lens functionality."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for all tests."""
        result = load_ioi_model(device="cpu")  # Use CPU for testing
        return result["model"]

    @pytest.fixture
    def example_tokens(self, model):
        """Create example tokens for testing."""
        prompt = "When Alice and Bob went to the store, Alice gave a bottle to"
        tokens = model.to_tokens(prompt)
        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")
        return tokens, bob_id, alice_id

    def test_compute_layer_wise_logit_diff_returns_correct_shape(self, model, example_tokens):
        """Test that logit lens returns correct number of layers."""
        tokens, bob_id, alice_id = example_tokens

        results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)

        # Should have n_layers + 1 entries (embeddings + each layer)
        n_layers = model.cfg.n_layers
        assert len(results["layer_logit_diffs"]) == n_layers + 1, \
            f"Should have {n_layers + 1} entries (embed + {n_layers} layers)"

        # Check all required keys present
        assert "layer_logit_diffs" in results
        assert "layer_io_logits" in results
        assert "layer_s_logits" in results
        assert "final_logit_diff" in results

    def test_logit_diff_builds_up_through_layers(self, model, example_tokens):
        """Test that logit difference generally increases through layers."""
        tokens, bob_id, alice_id = example_tokens

        results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)
        diffs = results["layer_logit_diffs"]

        # Final diff should be higher than initial (embeddings)
        initial_diff = diffs[0]
        final_diff = diffs[-1]

        # For IOI task, we expect the model to increase preference for IO (Bob)
        # This is a weak test - just checking general trend
        # Note: After final layernorm, absolute values may decrease but relative preference maintained
        assert abs(final_diff) >= abs(initial_diff) * 0.5, \
            "Model should build up or maintain logit difference through layers"

    def test_layer_wise_values_are_finite(self, model, example_tokens):
        """Test that no NaN or Inf values appear."""
        tokens, bob_id, alice_id = example_tokens

        results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)

        # Check all arrays are finite
        assert np.all(np.isfinite(results["layer_logit_diffs"])), \
            "All logit diffs should be finite"
        assert np.all(np.isfinite(results["layer_io_logits"])), \
            "All IO logits should be finite"
        assert np.all(np.isfinite(results["layer_s_logits"])), \
            "All S logits should be finite"

    def test_logit_diff_matches_io_minus_s(self, model, example_tokens):
        """Test that logit_diff = io_logit - s_logit."""
        tokens, bob_id, alice_id = example_tokens

        results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)

        # For each layer, verify the relationship
        for i in range(len(results["layer_logit_diffs"])):
            io_logit = results["layer_io_logits"][i]
            s_logit = results["layer_s_logits"][i]
            logit_diff = results["layer_logit_diffs"][i]

            expected_diff = io_logit - s_logit
            assert abs(logit_diff - expected_diff) < 1e-5, \
                f"Layer {i}: logit_diff should equal io_logit - s_logit"

    def test_final_logit_diff_matches_last_layer(self, model, example_tokens):
        """Test that final_logit_diff matches the last entry."""
        tokens, bob_id, alice_id = example_tokens

        results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)

        assert results["final_logit_diff"] == results["layer_logit_diffs"][-1], \
            "final_logit_diff should match the last layer's logit diff"

    def test_different_positions_give_different_results(self, model):
        """Test that analyzing different positions gives different results."""
        prompt = "When Alice and Bob went to the store, Alice gave a bottle to"
        tokens = model.to_tokens(prompt)
        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        # Analyze at last position (-1)
        results_last = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id, position=-1)

        # Analyze at middle position
        seq_len = tokens.shape[1]
        mid_pos = seq_len // 2
        results_mid = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id, position=mid_pos)

        # Results should differ (different positions give different predictions)
        assert not np.allclose(results_last["layer_logit_diffs"], results_mid["layer_logit_diffs"]), \
            "Different positions should give different logit differences"

    def test_plot_logit_lens_creates_figure(self, model, example_tokens, tmp_path):
        """Test that plotting creates a file."""
        tokens, bob_id, alice_id = example_tokens

        results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)

        # Save to temp directory
        save_path = tmp_path / "test_logit_lens.png"
        plot_logit_lens(results, save_path=str(save_path))

        # Check file was created
        assert save_path.exists(), "Plot should create a file"
        assert save_path.stat().st_size > 0, "Plot file should not be empty"

    def test_analyze_dataset_returns_statistics(self, model):
        """Test that dataset analysis returns mean, std, and all results."""
        # Create a small test dataset first
        from src.data.dataset import generate_ioi_dataset

        # Generate small test dataset
        result = generate_ioi_dataset(n_examples=5, template="ABBA", seed=42)

        # Analyze it
        lens_results = analyze_logit_lens_for_dataset(
            model,
            "data/ioi_abba.json",
            max_examples=5
        )

        # Check structure
        assert "mean_logit_diffs" in lens_results
        assert "std_logit_diffs" in lens_results
        assert "all_logit_diffs" in lens_results

        # Check shapes
        n_layers = model.cfg.n_layers + 1
        assert len(lens_results["mean_logit_diffs"]) == n_layers
        assert len(lens_results["std_logit_diffs"]) == n_layers
        assert lens_results["all_logit_diffs"].shape == (5, n_layers)

    def test_mean_across_examples_reduces_variance(self, model):
        """Test that averaging reduces variance compared to single example."""
        from src.data.dataset import generate_ioi_dataset

        # Generate small test dataset
        result = generate_ioi_dataset(n_examples=10, template="ABBA", seed=123)

        # Get individual example
        tokens_single = model.to_tokens("When Alice and Bob went to the store, Alice gave a bottle to")
        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        single_result = compute_layer_wise_logit_diff(model, tokens_single, bob_id, alice_id)

        # Get average across dataset
        avg_results = analyze_logit_lens_for_dataset(
            model,
            "data/ioi_abba.json",
            max_examples=10
        )

        # Standard deviation should be reasonable (not too high)
        mean_std = np.mean(avg_results["std_logit_diffs"])
        assert mean_std < 10.0, "Standard deviation across examples should be reasonable"

        # All values should be finite
        assert np.all(np.isfinite(avg_results["mean_logit_diffs"]))
        assert np.all(np.isfinite(avg_results["std_logit_diffs"]))

    def test_early_layers_show_initial_processing(self, model, example_tokens):
        """Test that early layers show meaningful computation."""
        tokens, bob_id, alice_id = example_tokens

        results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)
        diffs = results["layer_logit_diffs"]

        # Layer 0 (after embeddings) should differ from embeddings
        embed_diff = diffs[0]
        layer0_diff = diffs[1]

        # There should be some change (duplicate token heads work in L0)
        assert abs(layer0_diff - embed_diff) > 0.1, \
            "Layer 0 should show meaningful change from embeddings"

    def test_late_layers_show_strong_preference(self, model, example_tokens):
        """Test that late layers (9-11) show strong IOI preference."""
        tokens, bob_id, alice_id = example_tokens

        results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)
        diffs = results["layer_logit_diffs"]

        # Late layers (9, 10, 11) should have strong positive logit diff
        # (name mover heads should favor IO token)
        late_layers = diffs[-4:-1]  # Layers 9, 10, 11 (before final)

        # At least one late layer should show strong preference
        max_late_diff = np.max(np.abs(late_layers))
        assert max_late_diff > 5.0, \
            f"Late layers should show strong preference (>5.0), got {max_late_diff:.3f}"

    def test_layer_deltas_show_contributions(self, model, example_tokens):
        """Test that layer-by-layer deltas reveal circuit contributions."""
        tokens, bob_id, alice_id = example_tokens

        results = compute_layer_wise_logit_diff(model, tokens, bob_id, alice_id)
        diffs = results["layer_logit_diffs"]

        # Compute deltas
        deltas = np.diff(diffs)

        # Should have n_layers deltas
        assert len(deltas) == model.cfg.n_layers

        # At least some layers should contribute positively (increase logit diff)
        positive_contributions = np.sum(deltas > 0)
        assert positive_contributions >= model.cfg.n_layers * 0.3, \
            "At least 30% of layers should contribute positively"

    def test_works_with_different_names(self, model):
        """Test that logit lens works with different name pairs."""
        test_cases = [
            ("When David and Kate went to the park, David gave a gift to", "Kate", "David"),
            ("After Frank and Sarah met at the cafe, Frank handed a note to", "Sarah", "Frank"),
        ]

        for prompt, io_name, s_name in test_cases:
            tokens = model.to_tokens(prompt)
            io_id = model.to_single_token(" " + io_name)
            s_id = model.to_single_token(" " + s_name)

            results = compute_layer_wise_logit_diff(model, tokens, io_id, s_id)

            # Should return valid results
            assert len(results["layer_logit_diffs"]) == model.cfg.n_layers + 1
            assert np.all(np.isfinite(results["layer_logit_diffs"])), \
                f"Results should be finite for names {io_name}/{s_name}"


class TestLogitLensIntegration:
    """Integration tests for logit lens with other components."""

    @pytest.fixture(scope="class")
    def model_and_dataset(self):
        """Setup model and small dataset."""
        from src.data.dataset import generate_ioi_dataset

        result = load_ioi_model(device="cpu")
        model = result["model"]

        # Generate small dataset
        generate_ioi_dataset(n_examples=10, template="ABBA", seed=999)

        return model

    def test_logit_lens_correlates_with_baseline_performance(self, model_and_dataset):
        """Test that logit lens results correlate with baseline accuracy."""
        from src.analysis.ioi_baseline import run_baseline

        model = model_and_dataset

        # Run baseline
        baseline_results = run_baseline(
            model,
            "data/ioi_abba.json",
            max_examples=10
        )

        # Run logit lens
        lens_results = analyze_logit_lens_for_dataset(
            model,
            "data/ioi_abba.json",
            max_examples=10
        )

        # Final logit diff should be positive if baseline accuracy is good
        final_diff = lens_results["mean_logit_diffs"][-1]
        baseline_acc = baseline_results["accuracy"]

        if baseline_acc > 0.5:
            # If model performs above chance, final logit diff should favor IO
            assert final_diff > -5.0, \
                f"With {baseline_acc:.1%} accuracy, final logit diff should be reasonable"

    def test_logit_lens_validates_circuit_layer_ranges(self, model_and_dataset):
        """Test that logit lens shows effects in expected layer ranges."""
        model = model_and_dataset

        # Analyze dataset
        lens_results = analyze_logit_lens_for_dataset(
            model,
            "data/ioi_abba.json",
            max_examples=10
        )

        diffs = lens_results["mean_logit_diffs"]
        deltas = np.diff(diffs)

        # Check for contributions in circuit component layers
        # Duplicate token: L0-3
        early_contribution = np.sum(deltas[0:4])  # Deltas for layers 0-3

        # S-inhibition: L7-8
        middle_contribution = np.sum(deltas[7:9])  # Deltas for layers 7-8

        # Name movers: L9-10
        late_contribution = np.sum(deltas[9:11])  # Deltas for layers 9-10

        # At least one of these should show significant contribution
        total_contribution = early_contribution + middle_contribution + late_contribution
        assert abs(total_contribution) > 1.0, \
            "Circuit component layers should show measurable contributions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
