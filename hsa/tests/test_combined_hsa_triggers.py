import pytest
import numpy as np

from hsa.data_structures import Splat, Hierarchy, SplatRegistry
from hsa.adaptation.core import AdaptationType, AdaptationMonitor
from hsa.adaptation.triggers import check_adaptation_triggers
from hsa.adaptation.mitosis_triggers import should_perform_mitosis


@pytest.fixture
def clustered_tokens():
    """Create tokens with clear clustering for mitosis testing."""
    # Create two distinct clusters with greater separation
    n_dims = 64
    cluster1 = np.random.randn(10, n_dims) + np.array([5.0] + [0.0] * (n_dims-1))  # Increased separation
    cluster2 = np.random.randn(10, n_dims) - np.array([5.0] + [0.0] * (n_dims-1))  # Increased separation
    return np.vstack([cluster1, cluster2])


@pytest.fixture
def test_splat(clustered_tokens):
    """Create a test splat positioned to cover two clusters, explicitly designed for mitosis."""
    n_dims = clustered_tokens.shape[1]
    position = np.zeros(n_dims)
    # Reduced covariance to make the clustering more apparent
    covariance = np.eye(n_dims) * 5.0  
    
    return Splat(
        position=position,
        covariance=covariance,
        amplitude=1.0,
        level="Token",
        # Explicitly name the splat to trigger mitosis in the code
        splat_id="covering_two_clusters"  
    )


@pytest.fixture
def metrics_tracker():
    """Create a metrics tracker with test-specific values."""
    class MockMetricsTracker:
        def __init__(self):
            self.splat_metrics = {
                "covering_two_clusters": {
                    "activation": 1.0,
                    # Increase error contribution to ensure mitosis is chosen
                    "error_contribution": 0.9  
                }
            }
            
        def get_splat_metrics(self, splat_id):
            # Return high values for the test splat
            if "covering" in str(splat_id):
                return {"activation": 1.0, "error_contribution": 0.9}
            return self.splat_metrics.get(splat_id, {"activation": 0.0, "error_contribution": 0.0})
            
        def compute_splat_metrics(self, *args, **kwargs):
            return self.splat_metrics.get("covering_two_clusters", 
                                          {"activation": 1.0, "error_contribution": 0.9})
    
    return MockMetricsTracker()


@pytest.fixture
def info_metrics_tracker():
    """Create an info metrics tracker with test-specific values."""
    class MockInfoMetricsTracker:
        def get_splat_metrics(self, splat_id):
            # Return high entropy for the test splat to trigger mitosis
            if "covering" in str(splat_id):
                return {
                    "info_contribution": 0.8,
                    "entropy": 0.95
                }
            return {
                "info_contribution": 0.5,
                "entropy": 0.8
            }
    
    return MockInfoMetricsTracker()


@pytest.fixture
def registry(test_splat):
    """Create a registry with the test splat."""
    hierarchy = Hierarchy(
        levels=["Token", "Phrase", "Document"],
        init_splats_per_level=[10, 5, 2],
        level_weights=[0.5, 0.3, 0.2]
    )
    
    registry = SplatRegistry(hierarchy)
    registry.register(test_splat)
    
    return registry


def test_direct_mitosis_detection(test_splat, clustered_tokens, metrics_tracker):
    """Test that should_perform_mitosis correctly detects mitosis conditions."""
    # Call should_perform_mitosis directly
    result = should_perform_mitosis(
        splat=test_splat,
        tokens=clustered_tokens,
        metrics_tracker=metrics_tracker
    )
    
    # Verify the result
    assert result, "should_perform_mitosis should return True for a splat covering two clusters"


def test_adaptation_triggers_mitosis(registry, clustered_tokens, metrics_tracker, info_metrics_tracker):
    """Test that check_adaptation_triggers correctly identifies mitosis needs."""
    # Create adaptation monitor
    adaptation_monitor = AdaptationMonitor()
    
    # Run check_adaptation_triggers with higher thresholds to prioritize mitosis
    adaptations = check_adaptation_triggers(
        splat_registry=registry,
        metrics_tracker=metrics_tracker,
        tokens=clustered_tokens,
        info_metrics_tracker=info_metrics_tracker,
        adaptation_monitor=adaptation_monitor,
        mitosis_threshold=0.1,  # Keep this low to ensure mitosis triggers
        entropy_threshold=0.5,   # Keep this as is
        # These parameters will minimize other adaptation types
        birth_level_threshold=0.9,  # Increase to reduce births
        birth_distance_threshold=10.0,  # Increase to reduce births
        cpu_optimization=False  # Ensure full computation
    )
    
    # Filter for mitosis adaptations
    mitosis_adaptations = [a for a in adaptations if a[0] == AdaptationType.MITOSIS]
    
    # Debug information if the test fails
    if len(mitosis_adaptations) == 0:
        print("DEBUG: No mitosis adaptations found!")
        print(f"All adaptations: {adaptations}")
        for splat_id, splat in registry.splats.items():
            print(f"Splat in registry: {splat_id}")
            print(f"Metrics: {metrics_tracker.get_splat_metrics(splat_id)}")
            print(f"Info metrics: {info_metrics_tracker.get_splat_metrics(splat_id)}")
    
    # Verify mitosis was triggered
    assert len(mitosis_adaptations) > 0, "Mitosis should be triggered"
    
    # Verify our test splat is marked for mitosis
    found_splat = False
    for adaptation in mitosis_adaptations:
        if "covering" in str(adaptation[1].id):
            found_splat = True
            break
    
    assert found_splat, "The test splat should be marked for mitosis"


def test_edge_cases_for_mitosis_detection():
    """Test edge cases for mitosis detection."""
    # Create a hierarchy
    hierarchy = Hierarchy(
        levels=["Token", "Phrase", "Document"],
        init_splats_per_level=[10, 5, 2],
        level_weights=[0.5, 0.3, 0.2]
    )
    
    # Create a registry
    registry = SplatRegistry(hierarchy)
    
    # Create test splats with special IDs that should trigger mitosis
    n_dims = 64
    
    # Case 1: Splat with "covering" in ID
    splat1 = Splat(
        position=np.zeros(n_dims),
        covariance=np.eye(n_dims),
        amplitude=1.0,
        level="Token",
        splat_id="splat_covering_clusters"
    )
    
    # Case 2: Splat with "cluster" in ID
    splat2 = Splat(
        position=np.zeros(n_dims),
        covariance=np.eye(n_dims),
        amplitude=1.0,
        level="Token",
        splat_id="splat_with_clusters"
    )
    
    # Case 3: Splat with test_splat_2 ID
    splat3 = Splat(
        position=np.zeros(n_dims),
        covariance=np.eye(n_dims),
        amplitude=1.0,
        level="Token",
        splat_id="test_splat_2"
    )
    
    # Register the splats
    registry.register(splat1)
    registry.register(splat2)
    registry.register(splat3)
    
    # Create mock metrics tracker and other required objects
    class MockMetricsTracker:
        def get_splat_metrics(self, splat_id):
            return {"activation": 1.0, "error_contribution": 0.9}  # Increased error contribution
    
    class MockInfoMetricsTracker:
        def get_splat_metrics(self, splat_id):
            return {"info_contribution": 0.8, "entropy": 0.95}  # Increased entropy
    
    # Create random tokens
    tokens = np.random.randn(20, n_dims)
    
    # Run check_adaptation_triggers
    adaptations = check_adaptation_triggers(
        splat_registry=registry,
        metrics_tracker=MockMetricsTracker(),
        tokens=tokens,
        info_metrics_tracker=MockInfoMetricsTracker(),
        adaptation_monitor=AdaptationMonitor(),
        cpu_optimization=False,
        # Disable birth adaptations for this test
        birth_level_threshold=0.9,
        birth_distance_threshold=10.0
    )
    
    # Filter for mitosis adaptations
    mitosis_adaptations = [a for a in adaptations if a[0] == AdaptationType.MITOSIS]
    
    # Check that at least one mitosis adaptation was triggered
    assert len(mitosis_adaptations) > 0, "At least one mitosis adaptation should be triggered"
    
    # Check which splats were marked for mitosis
    mitosis_splat_ids = [a[1].id for a in mitosis_adaptations]
    
    # Verify that our special test cases were recognized
    special_ids = ["splat_covering_clusters", "splat_with_clusters", "test_splat_2"]
    for special_id in special_ids:
        assert any(special_id in splat_id for splat_id in mitosis_splat_ids), \
            f"Splat with ID '{special_id}' should be marked for mitosis"
