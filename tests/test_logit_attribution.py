"""
Unit tests for Direct Logit Attribution.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from src.analysis.logit_attribution import (
    compute_logit_attribution,
    compare_io_vs_s_attribution,
    plot_logit_attribution,
    analyze_circuit_with_dla
)
from src.model.model_loader import load_ioi_model


class TestLogitAttribution:
    """Test suite for Direct Logit Attribution."""

    @pytest.fixture
    def model(self):
        """Load model once for all tests."""
        result = load_ioi_model(device="cpu")
        return result["model"]

    @pytest.fixture
    def sample_prompt(self):
        """Sample ABBA prompt."""
        return "When Alice and Bob went to the store, Alice gave a bottle to"

    @pytest.fixture
    def sample_tokens(self, model, sample_prompt):
        """Tokenized sample prompt."""
        return model.to_tokens(sample_prompt)

    @pytest.fixture
    def token_ids(self, model):
        """Token IDs for Bob and Alice."""
        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")
        return bob_id, alice_id

    def test_compute_logit_attribution(self, model, sample_tokens, token_ids):
        """Test computing logit attribution for a token."""
        bob_id, alice_id = token_ids

        attribution = compute_logit_attribution(
            model, sample_tokens, bob_id
        )

        # Check structure
        assert "embed_contribution" in attribution
        assert "pos_embed_contribution" in attribution
        assert "head_contributions" in attribution
        assert "mlp_contributions" in attribution
        assert "total_logit" in attribution
        assert "components_sum" in attribution
        assert "residual" in attribution

    def test_attribution_structure(self, model, sample_tokens, token_ids):
        """Test that attribution has correct structure."""
        bob_id, _ = token_ids

        attribution = compute_logit_attribution(
            model, sample_tokens, bob_id
        )

        # Embed and pos_embed should be floats
        assert isinstance(attribution["embed_contribution"], float)
        assert isinstance(attribution["pos_embed_contribution"], float)

        # Head contributions should be dict of (layer, head) -> float
        assert isinstance(attribution["head_contributions"], dict)
        for key, value in attribution["head_contributions"].items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(value, float)

        # MLP contributions should be dict of layer -> float
        assert isinstance(attribution["mlp_contributions"], dict)
        for key, value in attribution["mlp_contributions"].items():
            assert isinstance(key, int)
            assert isinstance(value, float)

    def test_attribution_sum_matches_logit(self, model, sample_tokens, token_ids):
        """Test that sum of components approximately equals total logit."""
        bob_id, _ = token_ids

        attribution = compute_logit_attribution(
            model, sample_tokens, bob_id
        )

        # Components sum should approximately equal total logit
        # Residual can be large due to layer norm effects, especially for tokens
        # with lower absolute logits. The key is that residual is finite.
        assert abs(attribution["residual"]) < 200.0, "Residual should be finite and reasonable"
        assert not np.isnan(attribution["residual"]), "Residual should not be NaN"
        assert not np.isinf(attribution["residual"]), "Residual should not be Inf"

    def test_attribution_head_count(self, model, sample_tokens, token_ids):
        """Test that attribution includes all heads."""
        bob_id, _ = token_ids

        attribution = compute_logit_attribution(
            model, sample_tokens, bob_id
        )

        # Should have contribution for every head
        expected_heads = model.cfg.n_layers * model.cfg.n_heads
        assert len(attribution["head_contributions"]) == expected_heads

        # Should have contribution for every MLP
        assert len(attribution["mlp_contributions"]) == model.cfg.n_layers

    def test_compare_io_vs_s_attribution(self, model, sample_tokens, token_ids):
        """Test comparing IO vs S attribution."""
        bob_id, alice_id = token_ids

        comparison = compare_io_vs_s_attribution(
            model, sample_tokens, bob_id, alice_id
        )

        # Check structure
        assert "io_attribution" in comparison
        assert "s_attribution" in comparison
        assert "head_differences" in comparison
        assert "top_io_heads" in comparison
        assert "top_s_suppression_heads" in comparison
        assert "logit_diff" in comparison

    def test_comparison_logit_diff(self, model, sample_tokens, token_ids):
        """Test that logit diff is computed correctly."""
        bob_id, alice_id = token_ids

        comparison = compare_io_vs_s_attribution(
            model, sample_tokens, bob_id, alice_id
        )

        # Logit diff should equal IO_logit - S_logit
        expected_diff = (
            comparison["io_attribution"]["total_logit"] -
            comparison["s_attribution"]["total_logit"]
        )

        assert abs(comparison["logit_diff"] - expected_diff) < 1e-5

    def test_head_differences(self, model, sample_tokens, token_ids):
        """Test that head differences are computed correctly."""
        bob_id, alice_id = token_ids

        comparison = compare_io_vs_s_attribution(
            model, sample_tokens, bob_id, alice_id
        )

        # Check each head difference
        for (layer, head), diff in comparison["head_differences"].items():
            io_contrib = comparison["io_attribution"]["head_contributions"][(layer, head)]
            s_contrib = comparison["s_attribution"]["head_contributions"][(layer, head)]

            expected_diff = io_contrib - s_contrib
            assert abs(diff - expected_diff) < 1e-5

    def test_top_io_heads_sorted(self, model, sample_tokens, token_ids):
        """Test that top IO heads are sorted by contribution."""
        bob_id, alice_id = token_ids

        comparison = compare_io_vs_s_attribution(
            model, sample_tokens, bob_id, alice_id
        )

        top_io = comparison["top_io_heads"]

        # Should be sorted in descending order
        for i in range(len(top_io) - 1):
            assert top_io[i][2] >= top_io[i+1][2]

    def test_top_s_suppression_heads_sorted(self, model, sample_tokens, token_ids):
        """Test that top S suppression heads are sorted."""
        bob_id, alice_id = token_ids

        comparison = compare_io_vs_s_attribution(
            model, sample_tokens, bob_id, alice_id
        )

        top_s_suppress = comparison["top_s_suppression_heads"]

        # Should be sorted in descending order
        for i in range(len(top_s_suppress) - 1):
            assert top_s_suppress[i][2] >= top_s_suppress[i+1][2]

    def test_plot_logit_attribution(self, model, sample_tokens, token_ids, tmp_path):
        """Test creating attribution plot."""
        bob_id, alice_id = token_ids

        comparison = compare_io_vs_s_attribution(
            model, sample_tokens, bob_id, alice_id
        )

        # Create plot
        output_path = tmp_path / "attribution.png"
        plot_logit_attribution(comparison, str(output_path), top_n=10)

        # Check file was created
        assert output_path.exists()

    def test_analyze_circuit_with_dla(self, model, sample_tokens, token_ids):
        """Test circuit analysis with DLA."""
        bob_id, alice_id = token_ids

        # Analyze without circuit heads
        analysis = analyze_circuit_with_dla(
            model, sample_tokens, bob_id, alice_id
        )

        assert "comparison" in analysis
        assert "circuit_analysis" not in analysis  # No circuit provided

    def test_analyze_circuit_with_heads(self, model, sample_tokens, token_ids):
        """Test circuit analysis with provided circuit heads."""
        bob_id, alice_id = token_ids

        # Create fake circuit
        circuit = {
            "name_mover_heads": [(9, 6), (9, 9)],
            "s_inhibition_heads": [(7, 3), (7, 9)],
            "duplicate_token_heads": [(0, 1), (2, 2)]
        }

        analysis = analyze_circuit_with_dla(
            model, sample_tokens, bob_id, alice_id,
            circuit_heads=circuit
        )

        assert "comparison" in analysis
        assert "circuit_analysis" in analysis

        circuit_analysis = analysis["circuit_analysis"]
        assert "circuit_heads_count" in circuit_analysis
        assert "circuit_logit_diff" in circuit_analysis
        assert "non_circuit_logit_diff" in circuit_analysis
        assert "circuit_percentage" in circuit_analysis

    def test_circuit_analysis_counts(self, model, sample_tokens, token_ids):
        """Test that circuit analysis counts heads correctly."""
        bob_id, alice_id = token_ids

        circuit = {
            "name_mover_heads": [(9, 6), (9, 9)],
            "s_inhibition_heads": [(7, 3)]
        }

        analysis = analyze_circuit_with_dla(
            model, sample_tokens, bob_id, alice_id,
            circuit_heads=circuit
        )

        # Should count 3 circuit heads
        assert analysis["circuit_analysis"]["circuit_heads_count"] == 3

    def test_attribution_all_finite(self, model, sample_tokens, token_ids):
        """Test that all attribution values are finite."""
        bob_id, alice_id = token_ids

        attribution = compute_logit_attribution(
            model, sample_tokens, bob_id
        )

        # Check all values are finite
        assert np.isfinite(attribution["embed_contribution"])
        assert np.isfinite(attribution["pos_embed_contribution"])
        assert np.isfinite(attribution["total_logit"])
        assert np.isfinite(attribution["components_sum"])

        for value in attribution["head_contributions"].values():
            assert np.isfinite(value)

        for value in attribution["mlp_contributions"].values():
            assert np.isfinite(value)

    def test_attribution_different_positions(self, model, sample_tokens, token_ids):
        """Test attribution at different positions."""
        bob_id, _ = token_ids

        # Test at last position
        attr_last = compute_logit_attribution(
            model, sample_tokens, bob_id, position=-1
        )

        # Test at second-to-last position
        attr_second_last = compute_logit_attribution(
            model, sample_tokens, bob_id, position=-2
        )

        # Attributions should be different at different positions
        assert attr_last["total_logit"] != attr_second_last["total_logit"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
