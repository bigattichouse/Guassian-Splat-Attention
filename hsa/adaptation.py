"""
Compatibility module for Hierarchical Splat Attention (HSA) Adaptation.

This module provides backward compatibility with the original adaptation interface
to ensure existing code continues to work with the new modular structure.
"""

# Import from the adaptation submodules
from .adaptation.core import (
    AdaptationType,
    SplatChange,
    AdaptationResult,
    AdaptationMonitor,
    default_monitor,
    create_adaptation_result,
    record_mitosis,
    record_birth,
    record_death,
    record_merge,
    record_adjustment
)

from .adaptation.triggers import (
    should_perform_mitosis,
    identify_empty_regions,
    should_perform_birth,
    calculate_splat_similarity,
    find_merge_candidates,
    check_adaptation_triggers
)

from .adaptation.operations import (
    perform_mitosis,
    perform_birth,
    perform_death,
    perform_merge,
    perform_adjust,
    find_parent_for_level,
    perform_adaptations
)

from .adaptation.reporting import (
    AdaptationReporter,
    AdaptationVisualizer,
    default_reporter,
    default_visualizer,
    generate_report,
    visualize_result,
    visualize_history,
    visualize_splat_changes
)

from .adaptation.metrics import (
    AdaptationMetricsTracker,
    identify_token_clusters,
    identify_empty_regions_advanced,
    analyze_splat_information,
    estimate_optimal_splat_count,
    default_metrics_tracker
)

# Define what gets imported with "from hsa.adaptation import *"
__all__ = [
    # Core types
    'AdaptationType',
    'SplatChange',
    'AdaptationResult',
    'AdaptationMonitor',
    'default_monitor',
    'create_adaptation_result',
    
    # Triggers
    'should_perform_mitosis',
    'identify_empty_regions',
    'should_perform_birth',
    'calculate_splat_similarity',
    'find_merge_candidates',
    'check_adaptation_triggers',
    
    # Operations
    'perform_mitosis',
    'perform_birth',
    'perform_death',
    'perform_merge',
    'perform_adjust',
    'find_parent_for_level',
    'perform_adaptations',
    
    # Reporting
    'AdaptationReporter',
    'AdaptationVisualizer',
    'default_reporter',
    'default_visualizer',
    'generate_report',
    'visualize_result',
    'visualize_history',
    'visualize_splat_changes',
    
    # Metrics
    'AdaptationMetricsTracker',
    'identify_token_clusters',
    'identify_empty_regions_advanced',
    'analyze_splat_information',
    'estimate_optimal_splat_count',
    'default_metrics_tracker',
    'get_splat_metrics'
]
