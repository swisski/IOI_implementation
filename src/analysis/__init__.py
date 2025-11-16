"""Analysis utilities for IOI experiments."""
from .ioi_baseline import (
    run_baseline,
    save_baseline_results,
    analyze_errors,
    compute_logit_diff,
    get_token_positions,
    get_io_and_s_positions
)
from .activation_patching import (
    run_with_cache_both,
    patch_residual_stream,
    patch_attention_pattern,
    patch_attention_output,
    compute_patching_effect,
    patch_all_layers,
    patch_all_heads,
    analyze_example_patching
)
from .attention_analysis import (
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
from .path_patching import (
    get_path_patching_hook,
    patch_path,
    compute_path_patching_matrix,
    find_important_paths,
    analyze_ioi_circuit_paths
)
from .circuit_discovery import (
    discover_ioi_circuit,
    validate_circuit,
    visualize_circuit_structure,
    print_circuit_summary
)
from .logit_attribution import (
    compute_logit_attribution,
    compare_io_vs_s_attribution,
    plot_logit_attribution,
    analyze_circuit_with_dla
)

__all__ = [
    # Baseline
    "run_baseline",
    "save_baseline_results",
    "analyze_errors",
    "compute_logit_diff",
    "get_token_positions",
    "get_io_and_s_positions",
    # Activation Patching
    "run_with_cache_both",
    "patch_residual_stream",
    "patch_attention_pattern",
    "patch_attention_output",
    "compute_patching_effect",
    "patch_all_layers",
    "patch_all_heads",
    "analyze_example_patching",
    # Attention Analysis
    "get_attention_patterns",
    "compute_attention_to_position",
    "get_name_positions",
    "analyze_duplicate_token_attention",
    "analyze_s_inhibition_attention",
    "analyze_name_mover_attention",
    "find_duplicate_token_heads",
    "find_s_inhibition_heads",
    "find_name_mover_heads",
    "find_all_ioi_heads",
    # Path Patching
    "get_path_patching_hook",
    "patch_path",
    "compute_path_patching_matrix",
    "find_important_paths",
    "analyze_ioi_circuit_paths",
    # Circuit Discovery
    "discover_ioi_circuit",
    "validate_circuit",
    "visualize_circuit_structure",
    "print_circuit_summary",
    # Logit Attribution
    "compute_logit_attribution",
    "compare_io_vs_s_attribution",
    "plot_logit_attribution",
    "analyze_circuit_with_dla"
]
