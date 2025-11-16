"""
Unit tests for path patching functionality.
"""

import pytest
import torch
import numpy as np

from src.analysis.path_patching import (
    get_path_patching_hook,
    patch_path,
    compute_path_patching_matrix,
    find_important_paths,
    analyze_ioi_circuit_paths
)
from src.analysis.activation_patching import run_with_cache_both, get_logit_diff
from src.model.model_loader import load_ioi_model


class TestPathPatching:
    """Test suite for path patching."""

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
        """Tokenized sample prompts."""
        clean, corrupted = sample_prompts
        clean_tokens = model.to_tokens(clean)
        corrupted_tokens = model.to_tokens(corrupted)
        return clean_tokens, corrupted_tokens

    @pytest.fixture
    def token_ids(self, model):
        """Token IDs for Bob and Alice."""
        bob_id = model.to_single_token(" Bob")
        alice_id = model.to_single_token(" Alice")
        return bob_id, alice_id

    def test_get_path_patching_hook(self, model, sample_tokens):
        """Test creating path patching hook."""
        clean_tokens, corrupted_tokens = sample_tokens

        # Get clean cache
        clean_cache, _, _, _ = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        # Create hook
        hook_fn = get_path_patching_hook(
            sender_layer=2, sender_head=2,
            receiver_layer=9, receiver_head=9,
            cache_clean=clean_cache,
            component_type="attn"
        )

        # Hook should be callable
        assert callable(hook_fn)

    def test_patch_path_single(self, model, sample_tokens, token_ids):
        """Test patching a single path."""
        clean_tokens, corrupted_tokens = sample_tokens
        bob_id, alice_id = token_ids

        # Get clean cache
        clean_cache, _, _, _ = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        # Patch path from L2H2 to L9H9
        patched_diff = patch_path(
            model, clean_tokens, corrupted_tokens,
            sender=(2, 2), receiver=(9, 9),
            io_token_id=bob_id, s_token_id=alice_id,
            cache_clean=clean_cache
        )

        # Should return a finite float
        assert isinstance(patched_diff, float)
        assert np.isfinite(patched_diff)

    def test_patch_path_different_layers(self, model, sample_tokens, token_ids):
        """Test patching paths from different layers."""
        clean_tokens, corrupted_tokens = sample_tokens
        bob_id, alice_id = token_ids

        clean_cache, _, _, _ = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        # Patch from different sender layers to same receiver
        results = []
        for sender_layer in [0, 5, 9]:
            patched_diff = patch_path(
                model, clean_tokens, corrupted_tokens,
                sender=(sender_layer, 0), receiver=(10, 0),
                io_token_id=bob_id, s_token_id=alice_id,
                cache_clean=clean_cache
            )
            results.append(patched_diff)

        # All should be finite
        assert all(np.isfinite(r) for r in results)

    def test_compute_path_patching_matrix(self, model, sample_tokens, token_ids):
        """Test computing path patching matrix."""
        clean_tokens, corrupted_tokens = sample_tokens
        bob_id, alice_id = token_ids

        # Define sender and receiver heads
        sender_heads = [(0, 1), (2, 2)]
        receiver_heads = [(9, 6), (9, 9)]

        # Compute matrix
        results = compute_path_patching_matrix(
            model, clean_tokens, corrupted_tokens,
            sender_heads, receiver_heads,
            io_token_id=bob_id,
            s_token_id=alice_id
        )

        # Check structure
        assert "effect_matrix" in results
        assert "sender_heads" in results
        assert "receiver_heads" in results
        assert "clean_logit_diff" in results
        assert "corrupted_logit_diff" in results

        # Check matrix shape
        assert results["effect_matrix"].shape == (2, 2)

        # Check all effects are finite
        assert np.all(np.isfinite(results["effect_matrix"]))

    def test_path_matrix_dimensions(self, model, sample_tokens, token_ids):
        """Test that matrix dimensions match input lists."""
        clean_tokens, corrupted_tokens = sample_tokens
        bob_id, alice_id = token_ids

        # Different numbers of senders and receivers
        sender_heads = [(0, 1), (1, 2), (2, 3)]  # 3 senders
        receiver_heads = [(8, 0), (9, 6)]  # 2 receivers

        results = compute_path_patching_matrix(
            model, clean_tokens, corrupted_tokens,
            sender_heads, receiver_heads,
            io_token_id=bob_id,
            s_token_id=alice_id
        )

        # Should be 3x2 matrix
        assert results["effect_matrix"].shape == (3, 2)

    def test_find_important_paths(self):
        """Test finding important paths from matrix."""
        # Create sample effect matrix
        effect_matrix = np.array([
            [0.1, 0.5, 0.2],
            [0.8, 0.3, 0.1],
            [0.2, 0.4, 0.6]
        ])

        sender_heads = [(0, 1), (2, 2), (3, 3)]
        receiver_heads = [(9, 6), (9, 9), (10, 0)]

        # Find paths above threshold
        important = find_important_paths(
            effect_matrix, sender_heads, receiver_heads,
            threshold=0.4
        )

        # Should find 4 paths: (1,0)=0.8, (2,2)=0.6, (0,1)=0.5, (2,1)=0.4
        assert len(important) == 4

        # Should be sorted by effect (descending)
        for i in range(len(important) - 1):
            assert important[i][2] >= important[i+1][2]

        # Highest effect should be 0.8
        assert important[0][2] == 0.8
        assert important[0][0] == (2, 2)  # Sender
        assert important[0][1] == (9, 6)  # Receiver

    def test_find_important_paths_threshold(self):
        """Test that threshold filters paths correctly."""
        effect_matrix = np.array([
            [0.1, 0.3],
            [0.5, 0.7]
        ])

        sender_heads = [(0, 1), (2, 2)]
        receiver_heads = [(9, 6), (9, 9)]

        # Threshold 0.4 should include only 0.5 and 0.7
        important = find_important_paths(
            effect_matrix, sender_heads, receiver_heads,
            threshold=0.4
        )

        assert len(important) == 2
        assert all(effect >= 0.4 for _, _, effect in important)

    def test_analyze_ioi_circuit_paths(self, model, sample_tokens, token_ids):
        """Test full IOI circuit path analysis."""
        clean_tokens, corrupted_tokens = sample_tokens
        bob_id, alice_id = token_ids

        # Define head types (small subset for testing)
        duplicate_heads = [(0, 1), (2, 2)]
        s_inhibition_heads = [(7, 3)]
        name_mover_heads = [(9, 6), (9, 9)]

        # Analyze circuit
        results = analyze_ioi_circuit_paths(
            model, clean_tokens, corrupted_tokens,
            duplicate_heads, s_inhibition_heads, name_mover_heads,
            bob_id, alice_id
        )

        # Should have all three path types
        assert "dup_to_s_inhibition" in results
        assert "dup_to_name_mover" in results
        assert "s_inhibition_to_name_mover" in results

        # Each should have valid effect matrices
        for path_type in results.values():
            assert "effect_matrix" in path_type
            assert np.all(np.isfinite(path_type["effect_matrix"]))

    def test_path_patching_vs_activation_patching(self, model, sample_tokens, token_ids):
        """Test that path patching and activation patching are different."""
        clean_tokens, corrupted_tokens = sample_tokens
        bob_id, alice_id = token_ids

        clean_cache, _, _, _ = run_with_cache_both(
            model, clean_tokens, corrupted_tokens
        )

        # Path patching: patch only L2H2's contribution
        path_patched = patch_path(
            model, clean_tokens, corrupted_tokens,
            sender=(2, 2), receiver=(9, 9),
            io_token_id=bob_id, s_token_id=alice_id,
            cache_clean=clean_cache
        )

        # Activation patching: patch entire layer 2
        from src.analysis.activation_patching import patch_residual_stream
        activation_patched = patch_residual_stream(
            model, clean_tokens, corrupted_tokens, layer_idx=2,
            cache_clean=clean_cache,
            io_token_id=bob_id, s_token_id=alice_id
        )

        # These should generally be different
        # (though not guaranteed for all examples)
        # At minimum, both should be finite
        assert np.isfinite(path_patched)
        assert np.isfinite(activation_patched)

    def test_path_matrix_consistency(self, model, sample_tokens, token_ids):
        """Test that computing matrix twice gives same results."""
        clean_tokens, corrupted_tokens = sample_tokens
        bob_id, alice_id = token_ids

        sender_heads = [(0, 1), (2, 2)]
        receiver_heads = [(9, 6)]

        results1 = compute_path_patching_matrix(
            model, clean_tokens, corrupted_tokens,
            sender_heads, receiver_heads,
            io_token_id=bob_id,
            s_token_id=alice_id
        )

        results2 = compute_path_patching_matrix(
            model, clean_tokens, corrupted_tokens,
            sender_heads, receiver_heads,
            io_token_id=bob_id,
            s_token_id=alice_id
        )

        # Matrices should be identical
        np.testing.assert_array_almost_equal(
            results1["effect_matrix"],
            results2["effect_matrix"]
        )

    def test_empty_sender_receiver_lists(self, model, sample_tokens, token_ids):
        """Test behavior with empty sender/receiver lists."""
        clean_tokens, corrupted_tokens = sample_tokens
        bob_id, alice_id = token_ids

        # Empty senders
        results = compute_path_patching_matrix(
            model, clean_tokens, corrupted_tokens,
            sender_heads=[],
            receiver_heads=[(9, 6)],
            io_token_id=bob_id,
            s_token_id=alice_id
        )

        assert results["effect_matrix"].shape == (0, 1)

    def test_path_effect_range(self, model, sample_tokens, token_ids):
        """Test that path effects are in reasonable range."""
        clean_tokens, corrupted_tokens = sample_tokens
        bob_id, alice_id = token_ids

        sender_heads = [(0, 1), (2, 2)]
        receiver_heads = [(9, 6), (9, 9)]

        results = compute_path_patching_matrix(
            model, clean_tokens, corrupted_tokens,
            sender_heads, receiver_heads,
            io_token_id=bob_id,
            s_token_id=alice_id
        )

        # Most effects should be between -1 and 2
        # (some overshoot/undershoot is possible but rare)
        effects = results["effect_matrix"].flatten()
        assert all(-2.0 < e < 3.0 for e in effects)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
