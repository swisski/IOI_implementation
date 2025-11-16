"""
Unit tests for circuit discovery functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.analysis.circuit_discovery import (
    discover_ioi_circuit,
    validate_circuit,
    visualize_circuit_structure,
    print_circuit_summary
)
from src.model.model_loader import load_ioi_model
from src.data.dataset import generate_ioi_dataset


class TestCircuitDiscovery:
    """Test suite for circuit discovery."""

    @pytest.fixture
    def model(self):
        """Load model once for all tests."""
        result = load_ioi_model(device="cpu")
        return result["model"]

    @pytest.fixture
    def small_dataset(self, tmp_path):
        """Create small dataset for testing."""
        # Generate dataset
        generate_ioi_dataset(n_examples=20, template="ABBA", seed=42)
        return "data/ioi_abba.json"

    def test_discover_ioi_circuit(self, model, small_dataset):
        """Test discovering IOI circuit."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5,
            head_threshold=0.3,
            path_threshold=0.2
        )

        # Check structure
        assert "duplicate_token_heads" in circuit
        assert "s_inhibition_heads" in circuit
        assert "name_mover_heads" in circuit
        assert "critical_paths" in circuit
        assert "head_effects" in circuit
        assert "metadata" in circuit

        # All should be lists
        assert isinstance(circuit["duplicate_token_heads"], list)
        assert isinstance(circuit["s_inhibition_heads"], list)
        assert isinstance(circuit["name_mover_heads"], list)
        assert isinstance(circuit["critical_paths"], list)

    def test_circuit_metadata(self, model, small_dataset):
        """Test that circuit contains metadata."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5
        )

        metadata = circuit["metadata"]

        assert "head_threshold" in metadata
        assert "path_threshold" in metadata
        assert "max_examples" in metadata
        assert "total_heads_found" in metadata
        assert "total_paths_found" in metadata

    def test_circuit_heads_are_tuples(self, model, small_dataset):
        """Test that heads are (layer, head) tuples."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5
        )

        # Check all head types
        for head_type in ["duplicate_token_heads", "s_inhibition_heads", "name_mover_heads"]:
            for head in circuit[head_type]:
                assert isinstance(head, tuple)
                assert len(head) == 2
                layer, head_idx = head
                assert isinstance(layer, int)
                assert isinstance(head_idx, int)
                assert 0 <= layer < model.cfg.n_layers
                assert 0 <= head_idx < model.cfg.n_heads

    def test_circuit_paths_structure(self, model, small_dataset):
        """Test that paths have correct structure."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5,
            path_threshold=0.2
        )

        for path in circuit["critical_paths"]:
            assert "from" in path
            assert "to" in path
            assert "effect" in path
            assert "type" in path

            # from and to should be tuples
            assert isinstance(path["from"], tuple)
            assert isinstance(path["to"], tuple)

            # effect should be float
            assert isinstance(path["effect"], float)

    def test_validate_circuit(self, model, small_dataset):
        """Test circuit validation."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5
        )

        validation = validate_circuit(
            model,
            dataset_path=small_dataset,
            circuit_dict=circuit,
            max_examples=10
        )

        # Check structure
        assert "full_model_accuracy" in validation
        assert "full_model_logit_diff" in validation
        assert "circuit_heads_count" in validation
        assert "total_heads" in validation
        assert "circuit_percentage" in validation

        # Check values are reasonable
        assert 0.0 <= validation["full_model_accuracy"] <= 1.0
        assert validation["circuit_heads_count"] >= 0
        assert validation["total_heads"] == model.cfg.n_layers * model.cfg.n_heads
        assert 0.0 <= validation["circuit_percentage"] <= 1.0

    def test_visualize_circuit_json(self, model, small_dataset, tmp_path):
        """Test saving circuit as JSON."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5
        )

        output_path = tmp_path / "circuit.json"
        visualize_circuit_structure(circuit, str(output_path), format="json")

        # Check file exists
        assert output_path.exists()

        # Load and verify
        with open(output_path, 'r') as f:
            loaded = json.load(f)

        assert "duplicate_token_heads" in loaded
        assert "s_inhibition_heads" in loaded
        assert "name_mover_heads" in loaded

    def test_visualize_circuit_txt(self, model, small_dataset, tmp_path):
        """Test saving circuit as text."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5
        )

        output_path = tmp_path / "circuit.txt"
        visualize_circuit_structure(circuit, str(output_path), format="txt")

        # Check file exists
        assert output_path.exists()

        # Check content
        with open(output_path, 'r') as f:
            content = f.read()

        assert "DUPLICATE TOKEN HEADS" in content
        assert "S-INHIBITION HEADS" in content
        assert "NAME MOVER HEADS" in content

    def test_print_circuit_summary(self, model, small_dataset, capsys):
        """Test printing circuit summary."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5
        )

        print_circuit_summary(circuit)

        # Capture output
        captured = capsys.readouterr()

        assert "CIRCUIT SUMMARY" in captured.out
        assert "HEAD COUNTS" in captured.out
        assert "CRITICAL PATHS" in captured.out

    def test_circuit_head_effects(self, model, small_dataset):
        """Test that head effects are computed."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5
        )

        head_effects = circuit["head_effects"]

        # Should have effects for some heads
        assert len(head_effects) > 0

        # All effects should be floats
        for head_name, effect in head_effects.items():
            assert isinstance(head_name, str)
            assert isinstance(effect, float)

            # Head name should be in format "L{layer}H{head}"
            assert head_name.startswith("L")
            assert "H" in head_name

    def test_circuit_discovery_with_high_threshold(self, model, small_dataset):
        """Test that high threshold finds fewer heads."""
        circuit_low = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5,
            head_threshold=0.3
        )

        circuit_high = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5,
            head_threshold=0.7
        )

        # High threshold should find fewer or equal heads
        low_total = circuit_low["metadata"]["total_heads_found"]
        high_total = circuit_high["metadata"]["total_heads_found"]

        assert high_total <= low_total

    def test_circuit_paths_have_valid_effects(self, model, small_dataset):
        """Test that path effects are in reasonable range."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5,
            path_threshold=0.1
        )

        for path in circuit["critical_paths"]:
            effect = path["effect"]

            # Should be finite
            assert not (effect != effect)  # Not NaN
            assert abs(effect) < float('inf')  # Not infinite

            # Should be above threshold
            assert effect >= 0.1

    def test_validation_metrics_consistency(self, model, small_dataset):
        """Test that validation metrics are consistent."""
        circuit = discover_ioi_circuit(
            model,
            dataset_path=small_dataset,
            max_examples=5
        )

        validation = validate_circuit(
            model,
            dataset_path=small_dataset,
            circuit_dict=circuit,
            max_examples=10
        )

        # Circuit percentage should match counts
        expected_percentage = validation["circuit_heads_count"] / validation["total_heads"]
        assert abs(validation["circuit_percentage"] - expected_percentage) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
