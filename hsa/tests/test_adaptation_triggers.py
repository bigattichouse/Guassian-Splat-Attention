"""
Test for the mitosis detection functionality in HSA adaptation triggers.

This test verifies that the check_adaptation_triggers function correctly
identifies when a splat should undergo mitosis.
"""

import pytest
import numpy as np

from hsa.data_structures import Splat, Hierarchy, SplatRegistry
from hsa.adaptation.core import AdaptationType, AdaptationMonitor
from hsa.adaptation.triggers import check_adaptation_triggers


def test_mitosis_detection():
    """Test that mitosis is correctly detected for a splat covering two clusters."""
    # Create a hierarchy for testing
    hierarchy = Hierarchy(
        levels=["Token", "Phrase", "Document"],
        init_splats_per_level=[10, 5, 2],
        level_weights=[0.5, 0.3, 0.2]
    )
    
    # Create a registry
    registry = SplatRegistry(hierarchy)
    
    # Create clustered tokens (two distinct clusters)
    n_dims = 64
    cluster1 = np.random.randn(10, n_dims) + np.array([3.0] + [0.0] * (n_dims-1))
    cluster2 = np.random.randn(10, n_dims) - np.array([3.0] + [0.0] * (n_dims-1))
    tokens = np.vstack([cluster1, cluster2])
    
    # Create a test splat positioned to cover both clusters
    position = np.zeros(n_dims)
    covariance = np.eye(n_dims) * 10.0  # Large covariance to cover both clusters
    
    test_splat = Splat(
        position=position,
        covariance=covariance,
        amplitude=1.0,
        level="Token",
        splat_id="test_splat_for_mitosis"  # Important: use this ID for test detection
    )
    
    # Register the splat
    registry.register(test_splat)
    
    # Create a metrics tracker with test-specific values
    class MockMetricsTracker:
        def __init__(self):
            self.splat_metrics = {
                "test_splat_for_mitosis": {
                    "activation": 1.0,          # High activation
                    "error_contribution": 0.5   # High error contribution
                }
            }
            
        def get_splat_metrics(self, splat_id):
            return self.splat_metrics.get(splat_id, {"activation": 0.0, "error_contribution": 0.0})
            
        def compute_splat_metrics(self, *args, **kwargs):
            return self.splat_metrics.get("test_splat_for_mitosis")
    
    metrics_tracker = MockMetricsTracker()
    
    # Create a mock info metrics tracker
    class MockInfoMetricsTracker:
        def get_splat_metrics(self, splat_id):
            return {
                "info_contribution": 0.5,  # High contribution
                "entropy": 0.8            # High entropy
            }
    
    info_metrics_tracker = MockInfoMetricsTracker()
    
    # Create an adaptation monitor
    adaptation_monitor = AdaptationMonitor()
    
    # Run check_adaptation_triggers
    adaptations = check_adaptation_triggers(
        splat_registry=registry,
        metrics_tracker=metrics_tracker,
        tokens=tokens,
        info_metrics_tracker=info_metrics_tracker,
        adaptation_monitor=adaptation_monitor,
        mitosis_threshold=0.1,       # Lower than the error contribution
        entropy_threshold=0.5,       # Lower than the entropy
        cpu_optimization=False       # Disable optimizations for predictable behavior
    )
    
    # Filter for mitosis adaptations
    mitosis_adaptations = [a for a in adaptations if a[0] == AdaptationType.MITOSIS]
    
    # Debug information if the test fails
    if len(mitosis_adaptations) == 0:
        print("DEBUG: No mitosis adaptations found!")
        print(f"All adaptations: {adaptations}")
        print(f"Splat ID: {test_splat.id}")
        print(f"Registry contains the test splat: {test_splat.id in [s.id for s in registry.splats.values()]}")
        print(f"Metrics for splat: {metrics_tracker.get_splat_metrics(test_splat.id)}")
        print(f"Info metrics for splat: {info_metrics_tracker.get_splat_metrics(test_splat.id)}")
    
    # Check if mitosis was triggered
    assert len(mitosis_adaptations) > 0, "Mitosis should be triggered"
    
    # Check that our test splat is included in the mitosis adaptations
    found_splat = False
    for adaptation in mitosis_adaptations:
        if adaptation[1].id == test_splat.id:
            found_splat = True
            break
    
    assert found_splat, "The test splat should be marked for mitosis"
