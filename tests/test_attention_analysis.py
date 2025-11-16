"""
Unit tests for attention pattern analysis.
"""

import pytest
import torch
import json
import tempfile

from src.analysis.attention_analysis import (
    get_attention_patterns,
    compute_attention_to_position,
    get_name_positions,
    analyze_duplicate_token_attention,
    analyze_s_inhibition_attention,
    analyze_name_mover_attention,
    find_duplicate_token_heads,
    find_s_inhibition_heads,
    find_name_mover_heads,
    find_all_ioi_heads
)
from src.model.model_loader import load_ioi_model
from src.data.dataset import generate_ioi_dataset


class TestAttentionAnalysis:
    """Test suite for attention pattern analysis."""

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
    def small_dataset(self, tmp_path):
        """Create small dataset for testing."""
        generate_ioi_dataset(n_examples=10, template="ABBA", seed=42)

        # Load the dataset
        with open("data/ioi_abba.json", 'r') as f:
            dataset = json.load(f)

        return dataset[:10]

    def test_get_attention_patterns(self, model, sample_tokens):
        """Test extracting attention patterns for a head."""
        attn = get_attention_patterns(model, sample_tokens, layer=5, head=3)

        # Check shape
        assert len(attn.shape) == 3, "Should be 3D tensor (batch, seq, seq)"
        assert attn.shape[0] == 1, "Batch size should be 1"

        # Sequence dimensions should match
        assert attn.shape[1] == attn.shape[2], "Should be square attention matrix"

        # Attention should sum to ~1 along last dimension
        attn_sums = attn.sum(dim=-1)
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)

    def test_get_attention_patterns_all_layers(self, model, sample_tokens):
        """Test getting patterns from different layers."""
        n_layers = model.cfg.n_layers

        # Get patterns from first, middle, and last layer
        attn_0 = get_attention_patterns(model, sample_tokens, layer=0, head=0)
        attn_mid = get_attention_patterns(model, sample_tokens, layer=n_layers//2, head=0)
        attn_last = get_attention_patterns(model, sample_tokens, layer=n_layers-1, head=0)

        # All should have same shape
        assert attn_0.shape == attn_mid.shape == attn_last.shape

        # Should have different values
        assert not torch.allclose(attn_0, attn_mid, atol=1e-3)

    def test_compute_attention_to_position(self, model, sample_tokens):
        """Test computing attention between specific positions."""
        attn = get_attention_patterns(model, sample_tokens, layer=5, head=3)

        # Get attention from position 5 to position 2
        weight = compute_attention_to_position(attn, source_pos=5, target_pos=2)

        # Should be a float between 0 and 1
        assert isinstance(weight, float)
        assert 0.0 <= weight <= 1.0

    def test_compute_attention_to_position_negative_indices(self, model, sample_tokens):
        """Test using negative indices for positions."""
        attn = get_attention_patterns(model, sample_tokens, layer=5, head=3)

        # Attention from last position to second-to-last
        weight = compute_attention_to_position(attn, source_pos=-1, target_pos=-2)

        assert isinstance(weight, float)
        assert 0.0 <= weight <= 1.0

    def test_get_name_positions(self, model, sample_prompt):
        """Test finding name positions in prompt."""
        positions = get_name_positions(model, sample_prompt, "Alice", "Bob")

        # Should have positions for both names
        assert "A_positions" in positions
        assert "B_positions" in positions

        # Alice should appear twice
        assert len(positions["A_positions"]) == 2

        # Bob should appear once
        assert len(positions["B_positions"]) == 1

        # Alice should appear before and after Bob
        a_first, a_second = positions["A_positions"]
        b_pos = positions["B_positions"][0]
        assert a_first < b_pos < a_second

    def test_analyze_duplicate_token_attention(self, model, sample_prompt):
        """Test analyzing duplicate token attention."""
        # Try different heads
        for layer in [0, 5, 9]:
            for head in [0, 3]:
                score = analyze_duplicate_token_attention(
                    model, sample_prompt, "Alice", layer, head
                )

                # Should be a valid probability
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0

    def test_analyze_s_inhibition_attention(self, model, sample_prompt):
        """Test analyzing S-inhibition attention."""
        for layer in [0, 5, 9]:
            for head in [0, 3]:
                score = analyze_s_inhibition_attention(
                    model, sample_prompt, "Alice", layer, head
                )

                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0

    def test_analyze_name_mover_attention(self, model, sample_prompt):
        """Test analyzing name mover attention."""
        for layer in [0, 5, 9]:
            for head in [0, 3]:
                score = analyze_name_mover_attention(
                    model, sample_prompt, "Bob", layer, head
                )

                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0

    def test_find_duplicate_token_heads(self, model, small_dataset):
        """Test finding duplicate token heads."""
        heads = find_duplicate_token_heads(
            model, small_dataset, threshold=0.3, max_examples=5
        )

        # Should return a list of tuples
        assert isinstance(heads, list)

        # Each element should be (layer, head) tuple
        for layer, head in heads:
            assert isinstance(layer, int)
            assert isinstance(head, int)
            assert 0 <= layer < model.cfg.n_layers
            assert 0 <= head < model.cfg.n_heads

    def test_find_duplicate_token_heads_high_threshold(self, model, small_dataset):
        """Test that high threshold returns fewer heads."""
        heads_low = find_duplicate_token_heads(
            model, small_dataset, threshold=0.2, max_examples=5
        )

        heads_high = find_duplicate_token_heads(
            model, small_dataset, threshold=0.7, max_examples=5
        )

        # High threshold should find fewer or equal heads
        assert len(heads_high) <= len(heads_low)

    def test_find_s_inhibition_heads(self, model, small_dataset):
        """Test finding S-inhibition heads."""
        heads = find_s_inhibition_heads(
            model, small_dataset, threshold=0.2, max_examples=5
        )

        assert isinstance(heads, list)

        for layer, head in heads:
            assert 0 <= layer < model.cfg.n_layers
            assert 0 <= head < model.cfg.n_heads

    def test_find_name_mover_heads(self, model, small_dataset):
        """Test finding name mover heads."""
        heads = find_name_mover_heads(
            model, small_dataset, threshold=0.2, max_examples=5
        )

        assert isinstance(heads, list)

        for layer, head in heads:
            assert 0 <= layer < model.cfg.n_layers
            assert 0 <= head < model.cfg.n_heads

    def test_find_all_ioi_heads(self, model, tmp_path):
        """Test finding all three types of heads."""
        # Create small dataset
        generate_ioi_dataset(n_examples=10, template="ABBA", seed=123)
        dataset_path = "data/ioi_abba.json"

        results = find_all_ioi_heads(
            model,
            dataset_path=dataset_path,
            max_examples=5,
            duplicate_threshold=0.3,
            s_inhibition_threshold=0.2,
            name_mover_threshold=0.2
        )

        # Check structure
        assert "duplicate_token_heads" in results
        assert "s_inhibition_heads" in results
        assert "name_mover_heads" in results

        # All should be lists
        assert isinstance(results["duplicate_token_heads"], list)
        assert isinstance(results["s_inhibition_heads"], list)
        assert isinstance(results["name_mover_heads"], list)

    def test_attention_patterns_are_causal(self, model, sample_tokens):
        """Test that attention patterns respect causal masking."""
        attn = get_attention_patterns(model, sample_tokens, layer=5, head=3)

        seq_len = attn.shape[1]

        # Check that future positions have zero attention
        for pos in range(seq_len):
            for future_pos in range(pos + 1, seq_len):
                weight = attn[0, pos, future_pos].item()
                # Should be essentially zero (or very small due to numerical precision)
                assert weight < 1e-5, f"Position {pos} should not attend to future position {future_pos}"

    def test_attention_sums_to_one(self, model, sample_tokens):
        """Test that attention weights sum to 1 for each query position."""
        for layer in [0, 5, 11]:
            for head in [0, 5, 11]:
                attn = get_attention_patterns(model, sample_tokens, layer, head)

                # Sum over key dimension (last dimension)
                attn_sums = attn.sum(dim=-1)

                # Should all be close to 1
                assert torch.allclose(
                    attn_sums,
                    torch.ones_like(attn_sums),
                    atol=1e-5
                ), f"Attention at layer {layer}, head {head} doesn't sum to 1"

    def test_different_heads_have_different_patterns(self, model, sample_tokens):
        """Test that different heads have different attention patterns."""
        attn_h0 = get_attention_patterns(model, sample_tokens, layer=5, head=0)
        attn_h1 = get_attention_patterns(model, sample_tokens, layer=5, head=1)
        attn_h2 = get_attention_patterns(model, sample_tokens, layer=5, head=2)

        # Should not be identical
        assert not torch.allclose(attn_h0, attn_h1, atol=1e-3)
        assert not torch.allclose(attn_h1, attn_h2, atol=1e-3)

    def test_head_detection_consistency(self, model, small_dataset):
        """Test that running detection twice gives same results."""
        heads1 = find_duplicate_token_heads(
            model, small_dataset, threshold=0.5, max_examples=5
        )

        heads2 = find_duplicate_token_heads(
            model, small_dataset, threshold=0.5, max_examples=5
        )

        # Should be identical
        assert heads1 == heads2

    def test_name_positions_with_missing_name(self, model):
        """Test name positions when name doesn't appear."""
        prompt = "When Alice and Bob went to the store, Alice gave a bottle to"

        positions = get_name_positions(model, prompt, "Alice", "Charlie")

        # Charlie shouldn't appear
        assert len(positions["B_positions"]) == 0

    def test_duplicate_token_with_single_occurrence(self, model):
        """Test duplicate token analysis when name appears only once."""
        prompt = "When Alice and Bob went to the store"

        # Bob appears only once, should return 0.0
        score = analyze_duplicate_token_attention(
            model, prompt, "Bob", layer=0, head=0
        )

        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
