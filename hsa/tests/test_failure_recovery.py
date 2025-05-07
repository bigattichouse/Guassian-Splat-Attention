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
    repair_covariance_matrices, create_random_splats
)


# Fixtures
@pytest.fixture
def mock_registry():
    """Create a mock registry for testing."""
    hierarchy = Hierarchy()
    registry = SplatRegistry(hierarchy, embedding_dim=2)
    return registry


@pytest.fixture
def mock_splat():
    """Create a mock splat for testing."""
    splat = Mock()
    splat.id = "test_id"
    splat.dim = 2
    splat.position = np.array([0.0, 0.0])
    splat.covariance = np.eye(2)
    splat.level = "token"
    splat.parent = None
    splat.children = set()
    return splat


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
        "issues_by_type": {FailureType.NUMERICAL_INSTABILITY.name: 1},
        "needs_repair": False
    }
    
    report = recovery_instance.detect_and_recover()
    
    assert report["failures_detected"] == 1
    assert report["recovery_performed"] is False
    assert report["recovery_actions"] == []


# Test detect_and_recover with failures and bad health
@patch('hsa.failure_recovery.recover_numerical_instability')
def test_detect_and_recover_bad_health(
    mock_recover_num, recovery_instance, mock_failure_detector, mock_splat
):
    """Test detect_and_recover when failures are found and health is bad."""
    # Setup mocks
    mock_failure_detector.detect_pathological_configurations.return_value = [
        (FailureType.NUMERICAL_INSTABILITY, "Test failure", {"splat_id": "test_id"})
    ]
    
    # Mock the registry to return our mock splat
    def mock_get_splat(splat_id):
        if splat_id == "test_id":
            return mock_splat
        raise ValueError(f"Splat with ID {splat_id} not found in registry")
    
    recovery_instance.registry.get_splat = mock_get_splat
    recovery_instance.registry.safe_get_splat = mock_get_splat
    
    # Setup health checks
    mock_failure_detector.categorize_registry_health.side_effect = [
        # First call in initial check
        {
            "health_score": 0.5,  # Below repair threshold
            "category": "bad",
            "issues_by_type": {FailureType.NUMERICAL_INSTABILITY.name: 1},
            "needs_repair": True
        },
        # Second call after recovery
        {
            "health_score": 0.9,
            "category": "good",
            "issues_by_type": {},
            "needs_repair": False
        }
    ]
    
    # Setup recovery action return
    mock_recover_num.return_value = {
        "action": "repair_covariance",
        "splats_fixed": 1,
        "total_unstable": 1
    }
    
    report = recovery_instance.detect_and_recover()
    
    assert report["failures_detected"] == 1
    assert report["recovery_performed"] is True
    assert len(report["recovery_actions"]) == 1
    assert report["recovery_actions"][0]["action"] == "repair_covariance"


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
    
    # Setup mock splats in registry
    mock_splat1 = Mock()
    mock_splat1.id = "test1"
    mock_splat1.dim = 2
    mock_splat1.covariance = np.eye(2)
    
    mock_splat2 = Mock()
    mock_splat2.id = "test2"
    mock_splat2.dim = 2
    mock_splat2.level = "token"
    mock_splat2.parent = None
    
    # Configure get_splat to return the appropriate mock splat
    def mock_get_splat(splat_id):
        if splat_id == "test1":
            return mock_splat1
        elif splat_id == "test2":
            return mock_splat2
        raise ValueError(f"Splat with ID {splat_id} not found in registry")
    
    recovery_instance.registry.get_splat = mock_get_splat
    recovery_instance.registry.safe_get_splat = mock_get_splat
    
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
    # Check that eigenvalues are positive
    eigenvalues = np.linalg.eigvalsh(fixed_splat.covariance)
    assert np.all(eigenvalues > 0)


# Test empty level recovery
@patch('hsa.recovery_actions.create_random_splats')
def test_recover_empty_level(mock_create):
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
    
    # Setup mock for create_random_splats
    mock_create.return_value = 3  # 3 splats created
    
    # Setup failure data
    failures = [
        ("Empty level detected", {"level": "phrase"})
    ]
    
    # Perform recovery
    result = recover_empty_level(registry, failures)
    
    assert result is not None
    assert result["action"] == "populate_level"
    assert "phrase" in result["levels_fixed"]
    
    # Verify create_random_splats was called
    mock_create.assert_called_once()
    mock_create.assert_called_with(registry, "phrase", 3)


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


# Test orphaned splats recovery
def test_recover_orphaned_splats():
    """Test recovery from orphaned splats."""
    # Create a registry with a hierarchical structure
    hierarchy = Hierarchy(
        levels=["token", "document"],
        init_splats_per_level=[5, 1]
    )
    registry = SplatRegistry(hierarchy, embedding_dim=2)
    
    # Create a document splat to be parent
    doc_splat = Splat(dim=2, position=np.array([0.0, 0.0]), level="document")
    registry.register(doc_splat)
    
    # Create an orphaned token splat (no parent)
    orphaned_splat = Splat(dim=2, position=np.array([1.0, 1.0]), level="token")
    registry.register(orphaned_splat)
    
    # Setup failure data
    failures = [
        ("Orphaned splat detected", {"splat_id": orphaned_splat.id})
    ]
    
    # Perform recovery
    result = recover_orphaned_splats(registry, failures)
    
    assert result is not None
    assert result["action"] == "repair_relationships"
    assert result["splats_fixed"] == 1
    
    # Verify that orphaned splat now has a parent
    fixed_splat = registry.get_splat(orphaned_splat.id)
    assert fixed_splat.parent is not None
    assert fixed_splat.parent == doc_splat
    assert fixed_splat in doc_splat.children


# Test memory overflow recovery
@patch('hsa.failure_recovery.FailureRecovery._recover_memory_overflow')
def test_recover_memory_overflow(mock_recover_mem, recovery_instance):
    """Test recovery from memory overflow by pruning splats."""
    # Setup mock for _recover_memory_overflow
    mock_recover_mem.return_value = {
        "action": "prune_splats",
        "splats_removed": 40,
        "by_level": {"token": 40},
        "target_reduction": 40
    }
    
    # Setup failure data
    failures = [
        (FailureType.MEMORY_OVERFLOW, "Memory overflow", {"total_splats": 100})
    ]
    
    # Call recover method
    actions = recovery_instance.recover(failures)
    
    # Verify the result
    assert len(actions) == 1
    assert actions[0]["action"] == "prune_splats"
    assert actions[0]["splats_removed"] == 40
    
    # Verify _recover_memory_overflow was called
    mock_recover_mem.assert_called_once()


# Test adaptation stagnation recovery
@patch('hsa.recovery_actions.recover_adaptation_stagnation')
def test_recover_adaptation_stagnation(mock_recover_adapt, recovery_instance):
    """Test recovery from adaptation stagnation."""
    # Setup mock for recover_adaptation_stagnation
    mock_recover_adapt.return_value = {
        "action": "restart_adaptation",
        "success": True
    }
    
    # Setup failure data
    failures = [
        (FailureType.ADAPTATION_STAGNATION, "Adaptation stagnation", 
         {"failure_type": "EMPTY_LEVEL", "occurrences": 3})
    ]
    
    # Call recover method
    actions = recovery_instance.recover(failures)
    
    # Verify the result
    assert len(actions) == 1
    assert actions[0]["action"] == "restart_adaptation"
    assert actions[0]["success"] is True
    
    # Verify recover_adaptation_stagnation was called
    mock_recover_adapt.assert_called_once()


# Test health report generation
def test_get_health_report(recovery_instance, mock_failure_detector):
    """Test generation of health report."""
    # Setup mocks
    mock_failure_detector.detect_pathological_configurations.return_value = [
        (FailureType.NUMERICAL_INSTABILITY, "Test failure", {"splat_id": "test"}),
        (FailureType.EMPTY_LEVEL, "Empty level", {"level": "token"})
    ]
    
    mock_failure_detector.categorize_registry_health.return_value = {
        "health_score": 0.5,
        "category": "poor",
        "issues_by_type": {
            FailureType.NUMERICAL_INSTABILITY.name: 1,
            FailureType.EMPTY_LEVEL.name: 1
        },
        "needs_repair": True
    }
    
    # Mock hierarchy and levels
    recovery_instance.registry.hierarchy.levels = ["token", "document"]
    recovery_instance.registry.count_splats = Mock(return_value=5)
    
    # Call get_health_report
    report = recovery_instance.get_health_report()
    
    # Verify the report
    assert report["health_score"] == 0.5
    assert report["health_category"] == "poor"
    assert report["needs_recovery"] is True
    assert len(report["recommended_actions"]) > 0
    assert "repair_covariance" in report["recommended_actions"]
    assert "populate_level" in report["recommended_actions"]
