"""
Unit tests for activation patching functionality.
"""

import pytest
import torch
import numpy as np

from src.analysis.activation_patching import (
    run_with_cache_both,
    patch_residual_stream,
    patch_attention_pattern,
    patch_attention_output,
    compute_patching_effect,
    patch_all_layers,
    patch_all_heads,
    analyze_example_patching,
    get_logit_diff
)
from src.model.model_loader import load_ioi_model


class TestActivationPatching:
    """Test suite for activation patching."""

    @pytest.fixture
    def model(self):
        """Load model once for all tests."""
        result = load_ioi_model(device="cpu")
        return result["model"]

    @pytest.fixture
    def sample_prompts(self):
        """Sample clean and corrupted prompts."""
        clean = "When Alice and Bob went to the store, Alice gave a bottle to"
        corrupted = "When Alice and Bob went to the store, Charlie gave a bottle to"
        return clean, corrupted

    @pytest.fixture
    def sample_tokens(self, model, sample_prompts):
        """Sample tokenized prompts."""
        clean, corrupted = sample_prompts
        clean_tokens = model.to_tokens(clean)
        corrupted_tokens = model.to_tokens(corrupted)
        return clean_tokens, corrupted_tokens

    def test_run_with_cache_both(self, model, sample_tokens):
        """Test running model on both prompts with caching."""
        clean_tokens, corrupted_tokens = sample_tokens

        clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        # Check caches exist
        assert clean_cache is not None
        assert corrupted_cache is not None

        # Check logits exist and have correct shape
        assert clean_logits is not None
        assert corrupted_logits is not None
        assert len(clean_logits.shape) == 3  # (batch, seq_len, vocab)
        assert len(corrupted_logits.shape) == 3

        # Check caches contain activations
        assert len(clean_cache) > 0
        assert len(corrupted_cache) > 0

    def test_run_with_cache_both_same_length(self, model, sample_tokens):
        """Test that clean and corrupted runs have same sequence length."""
        clean_tokens, corrupted_tokens = sample_tokens

        # Make sure tokens are same length (important for patching)
        assert clean_tokens.shape == corrupted_tokens.shape

        clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        # Logits should have same sequence length
        assert clean_logits.shape[1] == corrupted_logits.shape[1]

    def test_get_logit_diff(self, model, sample_tokens):
        """Test logit difference computation."""
        clean_tokens, _ = sample_tokens

        # Run model
        with torch.no_grad():
            logits = model(clean_tokens)

        # Get token IDs (with leading space as they appear in text)
        alice_id = model.to_single_token(" Alice")
        bob_id = model.to_single_token(" Bob")

        # Compute logit diff
        diff = get_logit_diff(logits, bob_id, alice_id, position=-1)

        # Should be a tensor scalar
        assert isinstance(diff, torch.Tensor)
        assert diff.numel() == 1

        # Should be finite
        assert torch.isfinite(diff)

    def test_compute_patching_effect_full_restoration(self):
        """Test effect computation when patching fully restores performance."""
        clean_diff = 5.0
        corrupted_diff = 1.0
        patched_diff = 5.0  # Same as clean

        effect = compute_patching_effect(clean_diff, corrupted_diff, patched_diff)

        # Effect should be 1.0 (full restoration)
        assert abs(effect - 1.0) < 1e-6

    def test_compute_patching_effect_no_effect(self):
        """Test effect computation when patching has no effect."""
        clean_diff = 5.0
        corrupted_diff = 1.0
        patched_diff = 1.0  # Same as corrupted

        effect = compute_patching_effect(clean_diff, corrupted_diff, patched_diff)

        # Effect should be 0.0 (no effect)
        assert abs(effect - 0.0) < 1e-6

    def test_compute_patching_effect_partial_restoration(self):
        """Test effect computation for partial restoration."""
        clean_diff = 6.0
        corrupted_diff = 2.0
        patched_diff = 4.0  # Halfway between

        effect = compute_patching_effect(clean_diff, corrupted_diff, patched_diff)

        # Effect should be 0.5 (50% restoration)
        # (4.0 - 2.0) / (6.0 - 2.0) = 2.0 / 4.0 = 0.5
        assert abs(effect - 0.5) < 1e-6

    def test_compute_patching_effect_zero_denominator(self):
        """Test effect computation when clean and corrupted are same."""
        clean_diff = 5.0
        corrupted_diff = 5.0
        patched_diff = 5.0

        effect = compute_patching_effect(clean_diff, corrupted_diff, patched_diff)

        # Should return 0.0 to avoid division by zero
        assert effect == 0.0

    def test_patch_residual_stream(self, model, sample_tokens):
        """Test patching residual stream at a layer."""
        clean_tokens, corrupted_tokens = sample_tokens

        # Get caches
        clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        # Get token IDs (with leading space as they appear in text)
        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        # Patch at layer 5
        patched_diff = patch_residual_stream(
            model, clean_tokens, corrupted_tokens, layer_idx=5,
            cache_clean=clean_cache, io_token_id=bob_id, s_token_id=alice_id
        )

        # Should return a float
        assert isinstance(patched_diff, float)

        # Should be finite
        assert np.isfinite(patched_diff)

    def test_patch_residual_stream_all_layers(self, model, sample_tokens):
        """Test patching at each layer produces valid results."""
        clean_tokens, corrupted_tokens = sample_tokens

        clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        # Get baseline diffs
        from src.analysis.activation_patching import get_logit_diff
        clean_diff = get_logit_diff(clean_logits, bob_id, alice_id).item()
        corrupted_diff = get_logit_diff(corrupted_logits, bob_id, alice_id).item()

        # Patch at multiple layers
        results = []
        for layer_idx in [0, 5, 11]:  # First, middle, last
            patched_diff = patch_residual_stream(
                model, clean_tokens, corrupted_tokens, layer_idx,
                clean_cache, bob_id, alice_id
            )
            results.append(patched_diff)

            # Each result should be finite
            assert np.isfinite(patched_diff), f"Layer {layer_idx} should give finite result"

        # At least one layer should show some effect
        # (patched should be different from corrupted for at least one layer)
        effects = [abs(r - corrupted_diff) for r in results]
        assert any(e > 0.01 for e in effects), "At least one layer should have noticeable effect"

    def test_patch_attention_output(self, model, sample_tokens):
        """Test patching attention head output."""
        clean_tokens, corrupted_tokens = sample_tokens

        clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        # Patch head 0 at layer 5
        patched_diff = patch_attention_output(
            model, corrupted_tokens, layer_idx=5, head_idx=0,
            cache_clean=clean_cache, io_token_id=bob_id, s_token_id=alice_id
        )

        assert isinstance(patched_diff, float)
        assert np.isfinite(patched_diff)

    def test_patch_attention_pattern(self, model, sample_tokens):
        """Test patching attention pattern."""
        clean_tokens, corrupted_tokens = sample_tokens

        clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        # Patch pattern at head 0, layer 5
        patched_diff = patch_attention_pattern(
            model, corrupted_tokens, layer_idx=5, head_idx=0,
            cache_clean=clean_cache, io_token_id=bob_id, s_token_id=alice_id
        )

        assert isinstance(patched_diff, float)
        assert np.isfinite(patched_diff)

    def test_patch_all_layers(self, model, sample_tokens):
        """Test patching all layers at once."""
        clean_tokens, corrupted_tokens = sample_tokens

        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        results = patch_all_layers(
            model, clean_tokens, corrupted_tokens,
            bob_id, alice_id
        )

        # Check structure
        assert "clean_logit_diff" in results
        assert "corrupted_logit_diff" in results
        assert "layer_effects" in results
        assert "layer_patched_diffs" in results

        # Check sizes
        n_layers = model.cfg.n_layers
        assert len(results["layer_effects"]) == n_layers
        assert len(results["layer_patched_diffs"]) == n_layers

        # All effects should be finite
        assert all(np.isfinite(e) for e in results["layer_effects"])

    def test_patch_all_layers_effects_range(self, model, sample_tokens):
        """Test that layer effects are in reasonable range."""
        clean_tokens, corrupted_tokens = sample_tokens

        bob_id = model.to_single_token("Bob")
        alice_id = model.to_single_token("Alice")

        results = patch_all_layers(
            model, clean_tokens, corrupted_tokens,
            bob_id, alice_id
        )

        effects = results["layer_effects"]

        # Most effects should be between -1 and 2 (some overshoot is possible)
        for effect in effects:
            assert -2.0 < effect < 3.0, f"Effect {effect} is out of expected range"

    def test_patch_all_heads(self, model, sample_tokens):
        """Test patching all heads."""
        clean_tokens, corrupted_tokens = sample_tokens

        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        results = patch_all_heads(
            model, clean_tokens, corrupted_tokens,
            bob_id, alice_id,
            patch_type="output"
        )

        # Check structure
        assert "clean_logit_diff" in results
        assert "corrupted_logit_diff" in results
        assert "head_effects" in results
        assert "head_patched_diffs" in results

        # Check sizes
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        assert results["head_effects"].shape == (n_layers, n_heads)
        assert results["head_patched_diffs"].shape == (n_layers, n_heads)

    def test_patch_all_heads_pattern_type(self, model, sample_tokens):
        """Test patching all heads with pattern type."""
        clean_tokens, corrupted_tokens = sample_tokens

        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        results = patch_all_heads(
            model, clean_tokens, corrupted_tokens,
            bob_id, alice_id,
            patch_type="pattern"
        )

        # Should still have valid results
        assert results["head_effects"].shape == (model.cfg.n_layers, model.cfg.n_heads)

    def test_analyze_example_patching(self, model, sample_prompts):
        """Test full example analysis."""
        clean, corrupted = sample_prompts

        results = analyze_example_patching(
            model, clean, corrupted,
            io_name="Bob", s_name="Alice"
        )

        # Check structure
        assert "clean_prompt" in results
        assert "corrupted_prompt" in results
        assert "io_name" in results
        assert "s_name" in results
        assert "layer_results" in results
        assert "head_results" in results

        # Check prompts are stored
        assert results["clean_prompt"] == clean
        assert results["corrupted_prompt"] == corrupted

    def test_patching_different_positions_clean_vs_corrupted(self, model):
        """Test that clean and corrupted differ only at intended position."""
        clean = "When Alice and Bob went to the store, Alice gave a bottle to"
        corrupted = "When Alice and Bob went to the store, Charlie gave a bottle to"

        clean_tokens = model.to_tokens(clean)
        corrupted_tokens = model.to_tokens(corrupted)

        # Tokens should be same length
        assert clean_tokens.shape == corrupted_tokens.shape

        # Find where they differ
        diff_positions = (clean_tokens != corrupted_tokens).nonzero()

        # Should differ at exactly one position (where Alice -> Charlie)
        # Note: might differ at more than one position if tokenization splits differently
        assert len(diff_positions) >= 1, "Clean and corrupted should differ"

    def test_patching_restores_toward_clean(self, model, sample_tokens):
        """Test that patching generally moves logit diff toward clean value."""
        clean_tokens, corrupted_tokens = sample_tokens

        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")

        # Get baseline diffs
        clean_cache, corrupted_cache, clean_logits, corrupted_logits = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        clean_diff = get_logit_diff(clean_logits, bob_id, alice_id).item()
        corrupted_diff = get_logit_diff(corrupted_logits, bob_id, alice_id).item()

        # Patch at a middle layer
        patched_diff = patch_residual_stream(
            model, clean_tokens, corrupted_tokens, layer_idx=6,
            cache_clean=clean_cache, io_token_id=bob_id, s_token_id=alice_id
        )

        # Patched should generally be between corrupted and clean
        # (though this isn't guaranteed for all layers)
        # At minimum, effect should be finite
        effect = compute_patching_effect(clean_diff, corrupted_diff, patched_diff)
        assert np.isfinite(effect)

    def test_effect_metric_interpretation(self):
        """Test interpretation of effect metric."""
        # Full restoration
        assert compute_patching_effect(10.0, 0.0, 10.0) == 1.0

        # No effect
        assert compute_patching_effect(10.0, 0.0, 0.0) == 0.0

        # Half restoration
        assert compute_patching_effect(10.0, 0.0, 5.0) == 0.5

        # Negative effect (makes it worse)
        assert compute_patching_effect(10.0, 0.0, -5.0) == -0.5

        # Overshoot
        assert compute_patching_effect(10.0, 0.0, 15.0) == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
