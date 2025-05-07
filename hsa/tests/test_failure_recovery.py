"""
Tests for the failure recovery functionality in Hierarchical Splat Attention (HSA).

This module tests the failure detection and recovery mechanisms that ensure
robustness of HSA against various types of failures.
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock

from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa.failure_detection_types import FailureType
from hsa.failure_recovery import FailureRecovery
from hsa.recovery_actions import (
    RecoveryAction, recover_numerical_instability,
    recover_empty_level, recover_orphaned_splats,
    recover_adaptation_stagnation
)
from hsa.recovery_utils import (
    repair_covariance_matrices, create_random_splats,
    recover_from_failures
)


# Fixtures
@pytest.fixture
def mock_registry():
    """Create a mock registry for testing."""
    hierarchy = Hierarchy()
    registry = SplatRegistry(hierarchy, embedding_dim=2)
    return registry


@pytest.fixture
def mock_failure_detector():
    """Create a mock failure detector for testing."""
    detector = Mock()
    detector.detect_pathological_configurations.return_value = []
    detector.categorize_registry_health.return_value = {
        "health_score": 1.0,
        "category": "good",
        "issues_by_type": {},
        "needs_repair": False
    }
    return detector


@pytest.fixture
def recovery_instance(mock_registry, mock_failure_detector):
    """Create a recovery instance for testing."""
    recovery = FailureRecovery(mock_registry)
    recovery.detector = mock_failure_detector
    return recovery


# Test basic initialization and properties
def test_recovery_initialization(mock_registry):
    """Test the initialization of the FailureRecovery class."""
    recovery = FailureRecovery(mock_registry)
    
    assert recovery.registry == mock_registry
    assert recovery.auto_recovery is True
    assert recovery.max_recovery_attempts == 3
    assert recovery.repair_threshold == 0.7
    assert recovery.recovery_count == 0
    assert recovery.adaptation_controller is None
    
    # Check recovery history initialization
    for failure_type in FailureType:
        assert failure_type in recovery.recovery_history
        assert recovery.recovery_history[failure_type] == []


# Test detect_and_recover without failures
def test_detect_and_recover_no_failures(recovery_instance, mock_failure_detector):
    """Test detect_and_recover when no failures are detected."""
    mock_failure_detector.detect_pathological_configurations.return_value = []
    report = recovery_instance.detect_and_recover()
    
    assert report["failures_detected"] == 0
    assert report["recovery_performed"] is False
    assert report["recovery_actions"] == []
    assert report["health_before"] == report["health_after"]


# Test detect_and_recover with failures but good health
def test_detect_and_recover_good_health(recovery_instance, mock_failure_detector):
    """Test detect_and_recover when failures are found but health is good."""
    mock_failure_detector.detect_pathological_configurations.return_value = [
        (FailureType.NUMERICAL_INSTABILITY, "Test failure", {"splat_id": "test"})
    ]
    mock_failure_detector.categorize_registry_health.return_value = {
        "health_score": 0.8,  # Above repair threshold
        "category": "good",
        "issues_by_type": {FailureType.NUMERICAL_INSTABILITY: 1},
        "needs_repair": False
    }
    
    report = recovery_instance.detect_and_recover()
    
    assert report["failures_detected"] == 1
    assert report["recovery_performed"] is False
    assert report["recovery_actions"] == []


# Test detect_and_recover with failures and bad health
@patch('hsa.failure_recovery.recover_numerical_instability')
def test_detect_and_recover_bad_health(
    mock_recover_num, recovery_instance, mock_failure_detector
):
    """Test detect_and_recover when failures are found and health is bad."""
    # Setup mocks
    mock_failure_detector.detect_pathological_configurations.return_value = [
        (FailureType.NUMERICAL_INSTABILITY, "Test failure", {"splat_id": "test"})
    ]
    mock_failure_detector.categorize_registry_health.return_value = {
        "health_score": 0.5,  # Below repair threshold
        "category": "bad",
        "issues_by_type": {FailureType.NUMERICAL_INSTABILITY: 1},
        "needs_repair": True
    }
    
    # Setup recovery action return
    mock_recover_num.return_value = {
        "action": "repair_covariance",
        "splats_fixed": 1,
        "total_unstable": 1
    }
    
    # After recovery check
    mock_after_health = {
        "health_score": 0.9,
        "category": "good",
        "issues_by_type": {},
        "needs_repair": False
    }
    mock_failure_detector.categorize_registry_health.side_effect = [
        # First call in initial check
        {
            "health_score": 0.5,
            "category": "bad",
            "issues_by_type": {FailureType.NUMERICAL_INSTABILITY: 1},
            "needs_repair": True
        },
        # Second call after recovery
        mock_after_health
    ]
    
    report = recovery_instance.detect_and_recover()
    
    assert report["failures_detected"] == 1
    assert report["recovery_performed"] is True
    assert len(report["recovery_actions"]) == 1
    assert report["recovery_actions"][0]["action"] == "repair_covariance"
    assert report["health_after"] == mock_after_health
    
    # Verify recovery was called with correct arguments
    mock_recover_num.assert_called_once()
    args = mock_recover_num.call_args[0]
    assert args[0] == recovery_instance.registry
    assert len(args[1]) == 1
    assert args[1][0][0] == "Test failure"


# Test recover method for multiple failure types
@patch('hsa.failure_recovery.recover_numerical_instability')
@patch('hsa.failure_recovery.recover_empty_level')
@patch('hsa.failure_recovery.recover_orphaned_splats')
def test_recover_multiple_failures(
    mock_recover_orphaned, mock_recover_empty, mock_recover_num, 
    recovery_instance
):
    """Test the recover method with multiple failure types."""
    # Setup failure data
    failures = [
        (FailureType.NUMERICAL_INSTABILITY, "Numerical issue", {"splat_id": "test1"}),
        (FailureType.EMPTY_LEVEL, "Empty level", {"level": "token"}),
        (FailureType.ORPHANED_SPLAT, "Orphaned splat", {"splat_id": "test2"})
    ]
    
    # Setup recovery action returns
    mock_recover_num.return_value = {
        "action": "repair_covariance",
        "splats_fixed": 1,
        "total_unstable": 1
    }
    mock_recover_empty.return_value = {
        "action": "populate_level",
        "levels_fixed": ["token"],
        "splats_created": 5
    }
    mock_recover_orphaned.return_value = {
        "action": "repair_relationships",
        "splats_fixed": 1,
        "total_orphaned": 1
    }
    
    actions = recovery_instance.recover(failures)
    
    assert len(actions) == 3
    # Check if each recovery method was called once
    mock_recover_num.assert_called_once()
    mock_recover_empty.assert_called_once()
    mock_recover_orphaned.assert_called_once()
    
    # Check that all actions are in the result
    action_types = [a["action"] for a in actions]
    assert "repair_covariance" in action_types
    assert "populate_level" in action_types
    assert "repair_relationships" in action_types


# Test recovery attempt limits
@patch('hsa.failure_recovery.recover_numerical_instability')
def test_recovery_attempt_limits(mock_recover_num, recovery_instance):
    """Test that recovery attempts are limited per failure type."""
    # Setup failure data
    failures = [
        (FailureType.NUMERICAL_INSTABILITY, "Numerical issue", {"splat_id": "test1"})
    ]
    
    # Setup recovery history - too many recent attempts
    recovery_instance.recovery_history[FailureType.NUMERICAL_INSTABILITY] = [
        {"recovery_count": recovery_instance.recovery_count, "failure_data": [], "action": "repair_covariance"},
        {"recovery_count": recovery_instance.recovery_count - 1, "failure_data": [], "action": "repair_covariance"},
        {"recovery_count": recovery_instance.recovery_count - 2, "failure_data": [], "action": "repair_covariance"}
    ]
    
    # Recovery should be skipped
    actions = recovery_instance.recover(failures)
    
    assert len(actions) == 0
    # Verify recovery was not called
    mock_recover_num.assert_not_called()


# Test numerical instability recovery
def test_recover_numerical_instability():
    """Test recovery from numerical instability."""
    # Create a registry with an unstable splat
    hierarchy = Hierarchy()
    registry = SplatRegistry(hierarchy, embedding_dim=2)
    
    # Create a splat with unstable covariance
    splat = Splat(dim=2, position=np.array([0.0, 0.0]))
    splat.covariance = np.array([[1.0, 2.0], [2.0, 1.0]])  # Non-positive definite
    registry.register(splat)
    
    # Setup failure data
    failures = [
        ("Numerical instability detected", {"splat_id": splat.id})
    ]
    
    # Perform recovery
    result = recover_numerical_instability(registry, failures)
    
    assert result is not None
    assert result["action"] == "repair_covariance"
    assert result["splats_fixed"] == 1
    
    # Verify the covariance was fixed
    fixed_splat = registry.get_splat(splat.id)
    assert np.all(fixed_splat.covariance == np.eye(2))


# Test empty level recovery
def test_recover_empty_level():
    """Test recovery from empty level."""
    # Create a registry with an empty level
    hierarchy = Hierarchy(
        levels=["token", "phrase", "document"],
        init_splats_per_level=[5, 3, 1]
    )
    registry = SplatRegistry(hierarchy, embedding_dim=2)
    
    # Make sure phrase level is empty by only adding splats to other levels
    token_splat = Splat(dim=2, position=np.array([0.0, 0.0]), level="token")
    doc_splat = Splat(dim=2, position=np.array([1.0, 1.0]), level="document")
    registry.register(token_splat)
    registry.register(doc_splat)
    
    # Setup failure data
    failures = [
        ("Empty level detected", {"level": "phrase"})
    ]
    
    # Perform recovery
    with patch('hsa.recovery_actions.create_random_splats') as mock_create:
        mock_create.return_value = 3  # 3 splats created
        
        result = recover_empty_level(registry, failures)
        
        assert result is not None
        assert result["action"] == "populate_level"
        assert "phrase" in result["levels_fixed"]
        
        # Verify create_random_splats was called
        mock_create.assert_called_once()
        assert mock_create.call_args[0][0] == registry
        assert mock_create.call_args[0][1] == "phrase"


# Test covariance matrix repair
def test_repair_covariance_matrices():
    """Test the repair_covariance_matrices utility function."""
    # Create a registry with splats having problematic covariance matrices
    hierarchy = Hierarchy()
    registry = SplatRegistry(hierarchy, embedding_dim=2)
    
    # Create splats with various issues
    # 1. Splat with NaN values
    splat1 = Splat(dim=2, position=np.array([0.0, 0.0]))
    splat1.covariance = np.array([[np.nan, 0.0], [0.0, 1.0]])
    
    # 2. Splat with non-positive definite matrix
    splat2 = Splat(dim=2, position=np.array([1.0, 1.0]))
    splat2.covariance = np.array([[1.0, 2.0], [2.0, 1.0]])
    
    # 3. Splat with poor condition number
    splat3 = Splat(dim=2, position=np.array([2.0, 2.0]))
    splat3.covariance = np.array([[100.0, 0.0], [0.0, 0.001]])
    
    # 4. Splat with good covariance
    splat4 = Splat(dim=2, position=np.array([3.0, 3.0]))
    
    # Register all splats
    for splat in [splat1, splat2, splat3, splat4]:
        registry.register(splat)
    
    # Repair covariance matrices
    repaired = repair_covariance_matrices(registry)
    
    # We should have repaired 3 matrices
    assert repaired == 3
    
    # Verify each matrix is now well-formed
    for splat_id in [splat1.id, splat2.id, splat3.id]:
        splat = registry.get_splat(splat_id)
        
        # Check for NaN or Inf
        assert not np.isnan(splat.covariance).any()
        assert not np.isinf(splat.covariance).any()
        
        # Check positive definite
        eigenvalues = np.linalg.eigvalsh(splat.covariance)
        assert np.all(eigenvalues > 0)
        
        # Check condition number
        assert np.max(eigenvalues) / np.min(eigenvalues) <= 10.0


# Test creating random splats
def test_create_random_splats():
    """Test the create_random_splats utility function."""
    # Create a registry with a simple hierarchy
    hierarchy = Hierarchy(
        levels=["token", "document"],
        init_splats_per_level=[5, 1]
    )
    registry = SplatRegistry(hierarchy, embedding_dim=2)
    
    # Add a document splat to be parent
    doc_splat = Splat(dim=2, position=np.array([0.0, 0.0]), level="document")
    registry.register(doc_splat)
    
    # Create random splats at token level
    count = 5
    created = create_random_splats(registry, "token", count)
    
    # Verify the results
    assert created == count
    assert registry.count_splats("token") == count
    
    # Check that all created splats have the doc_splat as parent
    token_splats = list(registry.get_splats_at_level("token"))
    for splat in token_splats:
        assert splat.parent == doc_splat
        assert splat in doc_splat.children


# Test memory overflow recovery
def test_recover_memory_overflow(recovery_instance):
    """Test recovery from memory overflow by pruning splats."""
    # Create many splats to trigger memory overflow
    mock_registry = recovery_instance.registry
    
    # Mock the get_all_splats to return a long list
    all_splats = []
    for i in range(100):
        mock_splat = Mock()
        mock_splat.id = f"splat_{i}"
        mock_splat.level = "token"
        all_splats.append(mock_splat)
    
    mock_registry.get_all_splats = Mock(return_value=all_splats)
    mock_registry.unregister = Mock()
    
    # Call the recovery method
    failures = [("Memory overflow", {"total_splats": 100})]
    result = recovery_instance._recover_memory_overflow(failures)
    
    assert result is not None
    assert result["action"] == "prune_splats"
    assert result["splats_removed"] > 0
    
    # Check that unregister was called to remove splats
    assert mock_registry.unregister.call_count > 0


# Test attention collapse recovery
def test_recover_attention_collapse(recovery_instance):
    """Test recovery from attention collapse by resetting amplitudes."""
    # Setup mock registry and splats
    mock_registry = recovery_instance.registry
    
    # Create mock splats with low activations
    mock_splats = []
    for i in range(5):
        mock_splat = Mock()
        mock_splat.id = f"splat_{i}"
        mock_splat.level = "token"
        mock_splat.amplitude = 0.1  # Low amplitude
        mock_splat.get_average_activation = Mock(return_value=0.01)  # Low activation
        mock_splat.update_parameters = Mock()
        mock_splats.append(mock_splat)
    
    # Mock the registry methods
    mock_registry.get_splats_at_level = Mock(return_value=mock_splats)
    
    # Call the recovery method
    failures = [("Attention collapse", {"level": "token"})]
    result = recovery_instance._recover_attention_collapse(failures)
    
    assert result is not None
    assert result["action"] == "reset_amplitudes"
    assert result["adjustments_made"] == 5  # All splats adjusted
    
    # Check that update_parameters was called to reset amplitudes
    for splat in mock_splats:
        splat.update_parameters.assert_called_with(amplitude=1.0)


# Test the recover_from_failures utility function
def test_recover_from_failures(mock_registry):
    """Test the recover_from_failures convenience function."""
    with patch('hsa.recovery_utils.FailureRecovery') as MockRecovery:
        # Setup mock recovery
        mock_recovery = Mock()
        mock_recovery.detect_and_recover.return_value = {
            "health_before": {"health_score": 0.5},
            "failures_detected": 1,
            "recovery_performed": True,
            "recovery_actions": [{"action": "repair_covariance"}],
            "health_after": {"health_score": 0.9}
        }
        MockRecovery.return_value = mock_recovery
        
        # Call recover_from_failures
        report = recover_from_failures(mock_registry, auto_fix=True)
        
        # Verify recovery was instantiated and used
        MockRecovery.assert_called_once_with(mock_registry, auto_recovery=True)
        mock_recovery.detect_and_recover.assert_called_once()
        
        # Check the report
        assert report["failures_detected"] == 1
        assert report["recovery_performed"] == True
        assert len(report["recovery_actions"]) == 1


# Test information bottleneck recovery
def test_recover_information_bottleneck(recovery_instance):
    """Test recovery from information bottlenecks by rebalancing levels."""
    # Setup mock registry
    mock_registry = recovery_instance.registry
    
    # Call the recovery method
    failures = [(
        "Information bottleneck",
        {
            "lower_level": "token",
            "higher_level": "document",
            "lower_count": 20,
            "higher_count": 1
        }
    )]
    
    with patch('hsa.failure_recovery.create_random_splats') as mock_create:
        mock_create.return_value = 4  # 4 splats created
        
        result = recovery_instance._recover_information_bottleneck(failures)
        
        assert result is not None
        assert result["action"] == "rebalance_hierarchy"
        assert "document" in result["levels_rebalanced"]
        assert result["splats_created"] == 4
        
        # Verify create_random_splats was called
        mock_create.assert_called_once()
        assert mock_create.call_args[0][0] == mock_registry
        assert mock_create.call_args[0][1] == "document"


# Test handling computation timeout
def test_handle_computation_timeout(recovery_instance):
    """Test handling of computation timeouts."""
    # Case 1: No partial results
    result1 = recovery_instance.handle_computation_timeout()
    
    assert result1["timeout_handled"] is True
    assert result1["result_type"] == "fallback"
    assert result1["partial_results"] is None
    
    # Case 2: Partial attention matrix
    partial_attention = np.ones((5, 5))
    result2 = recovery_instance.handle_computation_timeout(partial_attention)
    
    assert result2["timeout_handled"] is True
    assert result2["result_type"] == "partial_normalized"
    assert "normalized_attention" in result2


# Test switch to fallback attention
def test_switch_to_fallback_attention(recovery_instance):
    """Test switching to fallback attention mechanism."""
    fallback = recovery_instance.switch_to_fallback_attention()
    
    assert fallback["type"] == "fallback"
    assert fallback["mechanism"] == "dense"
    assert fallback["use_causal_mask"] is True
    assert fallback["normalize_rows"] is True
    assert fallback["skip_splats"] is True
    assert fallback["recovery_active"] is True


# Test adaptation stagnation recovery
def test_recover_adaptation_stagnation():
    """Test recovery from adaptation stagnation."""
    # Create test registry
    hierarchy = Hierarchy()
    registry = SplatRegistry(hierarchy, embedding_dim=2)
    
    # Create a mock adaptation controller
    mock_controller = Mock()
    mock_controller.reset_statistics = Mock()
    
    # Setup failure data
    failures = [
        ("Adaptation stagnation", {"failure_type": "EMPTY_LEVEL", "occurrences": 3})
    ]
    
    # Perform recovery
    result = recover_adaptation_stagnation(registry, mock_controller, failures)
    
    assert result is not None
    assert result["action"] == "restart_adaptation"
    assert result["success"] is True
    
    # Verify controller's reset_statistics was called
    mock_controller.reset_statistics.assert_called_once()


# Test orphaned splats recovery
def test_recover_orphaned_splats():
    """Test recovery from orphaned splats."""
    # Create a registry with orphaned splats
    hierarchy = Hierarchy(
        levels=["token", "phrase", "document"],
        init_splats_per_level=[5, 3, 1]
    )
    registry = SplatRegistry(hierarchy, embedding_dim=2)
    
    # Create a document-level splat
    doc_splat = Splat(dim=2, position=np.array([0.0, 0.0]), level="document")
    registry.register(doc_splat)
    
    # Create a phrase-level splat with the document as parent
    phrase_splat = Splat(
        dim=2,
        position=np.array([1.0, 1.0]),
        level="phrase",
        parent=doc_splat
    )
    registry.register(phrase_splat)
    doc_splat.children.add(phrase_splat)
    
    # Create an orphaned token-level splat
    orphan_splat = Splat(dim=2, position=np.array([2.0, 2.0]), level="token")
    registry.register(orphan_splat)
    
    # Setup failure data
    failures = [
        ("Orphaned splat detected", {"splat_id": orphan_splat.id})
    ]
    
    # Perform recovery
    result = recover_orphaned_splats(registry, failures)
    
    assert result is not None
    assert result["action"] == "repair_relationships"
    assert result["splats_fixed"] == 1
    
    # Verify the orphan now has a parent
    fixed_splat = registry.get_splat(orphan_splat.id)
    assert fixed_splat.parent is not None
    assert fixed_splat.parent.level == "phrase"
    assert fixed_splat in fixed_splat.parent.children


# Test repair_integrity method
def test_repair_integrity(recovery_instance):
    """Test the repair_integrity method."""
    # Mock the registry's integrity methods
    mock_registry = recovery_instance.registry
    mock_registry.verify_integrity = Mock(side_effect=[False, True])
    mock_registry.repair_integrity = Mock(return_value=5)
    
    # Call repair_integrity
    result = recovery_instance.repair_integrity()
    
    assert result is True
    mock_registry.verify_integrity.assert_called()
    mock_registry.repair_integrity.assert_called_once()


# Test rebalance_hierarchy method
def test_rebalance_hierarchy(recovery_instance):
    """Test the rebalance_hierarchy method."""
    # Mock the registry
    mock_registry = recovery_instance.registry
    mock_registry.hierarchy.levels = ["token", "document"]
    mock_registry.hierarchy.init_splats_per_level = [5, 1]
    mock_registry.count_splats = Mock(side_effect=lambda level: 2 if level == "token" else 3)
    
    # Patch create_random_splats
    with patch('hsa.failure_recovery.create_random_splats') as mock_create:
        mock_create.return_value = 3
        
        # Call rebalance_hierarchy
        result = recovery_instance.rebalance_hierarchy()
        
        assert result is True
        # Should create splats for token level (3 more needed)
        mock_create.assert_called_once_with(mock_registry, "token", 3)


# Test reset_to_initial_state method
def test_reset_to_initial_state(recovery_instance):
    """Test the reset_to_initial_state method."""
    # Mock the registry
    mock_registry = recovery_instance.registry
    mock_registry.splats = {}
    mock_registry.splats_by_level = {"token": set(), "document": set()}
    mock_registry.count_splats = Mock(return_value=10)
    
    # Patch the initialization method
    with patch('hsa.failure_recovery.create_initial_splats') as mock_init:
        mock_init.return_value = 10
        
        # Call reset_to_initial_state
        result = recovery_instance.reset_to_initial_state()
        
        assert result is True
        # Verify registry was cleared and reinitialized
        mock_init.assert_called_once_with(mock_registry)


# Test get_health_report method
def test_get_health_report(recovery_instance, mock_failure_detector):
    """Test the get_health_report method."""
    # Setup mock detector
    mock_failure_detector.detect_pathological_configurations.return_value = [
        (FailureType.NUMERICAL_INSTABILITY, "Test failure", {"splat_id": "test"})
    ]
    mock_failure_detector.categorize_registry_health.return_value = {
        "health_score": 0.5,
        "category": "concerning",
        "issues_by_type": {"NUMERICAL_INSTABILITY": 1},
        "needs_repair": True
    }
    
    # Mock registry level counts
    mock_registry = recovery_instance.registry
    mock_registry.hierarchy.levels = ["token", "document"]
    mock_registry.count_splats = Mock(side_effect=lambda level: 5 if level == "token" else 1)
    
    # Get health report
    report = recovery_instance.get_health_report()
    
    assert report["health_score"] == 0.5
    assert report["health_category"] == "concerning"
    assert report["total_splats"] == 6
    assert report["level_distribution"]["token"] == 5
    assert report["level_distribution"]["document"] == 1
    assert report["level_ratios"]["token:document"] == 5.0
    assert "NUMERICAL_INSTABILITY" in report["failures_by_type"]
    assert report["needs_recovery"] is True
    assert "repair_covariance" in report["recommended_actions"]
