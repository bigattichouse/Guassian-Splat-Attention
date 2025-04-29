"""
Test module for HSA adaptation operations.

This module tests the various adaptation operations (birth, death, merging, mitosis)
and verifies that splat registries remain consistent throughout these operations.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple, Set, Optional

# Import the necessary modules
from hsa.data_structures import Splat, Hierarchy, SplatRegistry
from hsa.adaptation.core import AdaptationType, AdaptationResult
from hsa.adaptation.operations import (
    perform_birth,
    perform_death,
    perform_merge,
    perform_mitosis,
    perform_adaptations,
    find_parent_for_level
)
from hsa.initialization import initialize_splats


@pytest.fixture
def hierarchy():
    """Create a hierarchy for testing."""
    return Hierarchy(
        levels=["Token", "Phrase", "Document"],
        init_splats_per_level=[10, 5, 2],
        level_weights=[0.5, 0.3, 0.2]
    )


@pytest.fixture
def tokens():
    """Create tokens for testing."""
    # Create random tokens in a 3D space
    return np.random.randn(20, 64)


@pytest.fixture
def splat_registry(hierarchy, tokens):
    """Create a splat registry for testing."""
    return initialize_splats(
        tokens=tokens,
        hierarchy_config={
            "levels": hierarchy.levels,
            "init_splats_per_level": hierarchy.init_splats_per_level,
            "level_weights": hierarchy.level_weights
        }
    )


def verify_registry_consistency(registry: SplatRegistry) -> bool:
    """
    Verify that a registry is internally consistent.
    
    This checks for:
    - All splats are properly registered
    - Level counts match
    - Parent-child relationships are valid
    """
    # Check that all splats are in their correct level collections
    for splat_id, splat in registry.splats.items():
        # Splat should be in its level collection
        assert splat in registry.splats_by_level[splat.level], f"Splat {splat_id} missing from level collection"
    
    # Check that level collections only contain splats that exist
    for level, splats in registry.splats_by_level.items():
        for splat in splats:
            assert splat.id in registry.splats, f"Level {level} contains unregistered splat {splat.id}"
    
    # Check parent-child relationships
    for splat_id, splat in registry.splats.items():
        # If splat has parent, parent should have this splat as child
        if splat.parent is not None:
            assert splat in splat.parent.children, f"Parent of {splat_id} doesn't list it as child"
            # Parent should exist in registry
            assert splat.parent.id in registry.splats, f"Parent of {splat_id} not found in registry"
        
        # All children should have this splat as parent
        for child in splat.children:
            assert child.parent is splat, f"Child of {splat_id} doesn't have it as parent"
            # Child should exist in registry
            assert child.id in registry.splats, f"Child of {splat_id} not found in registry"
    
    return True


def test_birth_operation(splat_registry, tokens):
    """Test that birth operations maintain registry consistency."""
    # Get initial counts
    initial_count = len(splat_registry.splats)
    
    # Create a new splat via birth at the Token level
    level = "Token"
    # Use mean of tokens as position
    position = np.mean(tokens, axis=0)
    
    # Create adaptation result for tracking
    result = AdaptationResult()
    
    # Perform birth
    new_splat = perform_birth(
        level=level,
        position=position,
        tokens=tokens,
        splat_registry=splat_registry,
        result=result
    )
    
    # Check that the birth was successful
    assert new_splat is not None, "Birth operation failed"
    
    # Register the new splat
    splat_registry.register(new_splat)
    
    # Check counts after birth
    assert len(splat_registry.splats) == initial_count + 1, "Splat count not increased after birth"
    assert len(splat_registry.get_splats_at_level(level)) > 0, f"No splats at level {level} after birth"
    
    # Verify registry consistency
    assert verify_registry_consistency(splat_registry), "Registry inconsistent after birth"


def test_death_operation(splat_registry, tokens):
    """Test that death operations maintain registry consistency."""
    # Get initial counts
    initial_count = len(splat_registry.splats)
    
    # Select a splat to remove
    if initial_count == 0:
        pytest.skip("No splats available for death test")
    
    splat_to_remove = next(iter(splat_registry.splats.values()))
    
    # Create adaptation result for tracking
    result = AdaptationResult()
    
    # Perform death
    success = perform_death(
        splat=splat_to_remove,
        splat_registry=splat_registry,
        result=result
    )
    
    # Check that the death was successful
    assert success, "Death operation failed"
    
    # Check counts after death
    assert len(splat_registry.splats) == initial_count - 1, "Splat count not decreased after death"
    assert splat_to_remove.id not in splat_registry.splats, "Removed splat still in registry"
    
    # Verify registry consistency
    assert verify_registry_consistency(splat_registry), "Registry inconsistent after death"


def test_merge_operation(splat_registry, tokens):
    """Test that merge operations maintain registry consistency."""
    # Get initial counts
    initial_count = len(splat_registry.splats)
    
    # Need at least 2 splats for merging
    if initial_count < 2:
        pytest.skip("Not enough splats available for merge test")
    
    # Get two splats at the same level for merging
    all_splats = list(splat_registry.splats.values())
    
    # Find splats at the same level
    level_groups = {}
    for splat in all_splats:
        if splat.level not in level_groups:
            level_groups[splat.level] = []
        level_groups[splat.level].append(splat)
    
    # Find a level with at least 2 splats
    merge_level = None
    for level, splats in level_groups.items():
        if len(splats) >= 2:
            merge_level = level
            break
    
    if merge_level is None:
        pytest.skip("No level has at least 2 splats for merge test")
    
    # Get two splats to merge
    target_splat = level_groups[merge_level][0]
    source_splat = level_groups[merge_level][1]
    
    # Create adaptation result for tracking
    result = AdaptationResult()
    
    # Store target's original children
    original_children = set(target_splat.children)
    source_children = set(source_splat.children)
    
    # Perform merge
    success = perform_merge(
        target_splat=target_splat,
        source_splat=source_splat,
        splat_registry=splat_registry,
        result=result
    )
    
    # Check that the merge was successful
    assert success, "Merge operation failed"
    
    # Check counts after merge
    assert len(splat_registry.splats) == initial_count - 1, "Splat count not decreased after merge"
    assert source_splat.id not in splat_registry.splats, "Source splat still in registry after merge"
    assert target_splat.id in splat_registry.splats, "Target splat not in registry after merge"
    
    # Check that source's children are now target's children
    for child in source_children:
        assert child in target_splat.children, "Source child not transferred to target"
        assert child.parent is target_splat, "Child's parent not updated to target"
    
    # Verify registry consistency
    assert verify_registry_consistency(splat_registry), "Registry inconsistent after merge"


def test_mitosis_operation(splat_registry, tokens):
    """Test that mitosis operations maintain registry consistency."""
    # Get initial counts
    initial_count = len(splat_registry.splats)
    
    # Need at least 1 splat for mitosis
    if initial_count < 1:
        pytest.skip("No splats available for mitosis test")
    
    # Select a splat to divide
    splat_to_divide = next(iter(splat_registry.splats.values()))
    
    # Create adaptation result for tracking
    result = AdaptationResult()
    
    # Perform mitosis
    child_splats = perform_mitosis(
        splat=splat_to_divide,
        tokens=tokens,
        result=result
    )
    
    # Check that the mitosis was successful
    assert len(child_splats) == 2, "Mitosis should produce 2 child splats"
    
    # Check properties of child splats
    for child in child_splats:
        assert child.level == splat_to_divide.level, "Child level doesn't match parent"
        assert isinstance(child.position, np.ndarray), "Child position is not an ndarray"
        assert isinstance(child.covariance, np.ndarray), "Child covariance is not an ndarray"
        assert child.amplitude > 0, "Child amplitude should be positive"
    
    # Replace the parent with children in the registry
    splat_registry.replace_splat(splat_to_divide, child_splats)
    
    # Check counts after mitosis
    assert len(splat_registry.splats) == initial_count + 1, "Splat count not increased by 1 after mitosis"
    assert splat_to_divide.id not in splat_registry.splats, "Divided splat still in registry"
    for child in child_splats:
        assert child.id in splat_registry.splats, "Child splat not in registry"
    
    # Verify registry consistency
    assert verify_registry_consistency(splat_registry), "Registry inconsistent after mitosis"


def test_complex_adaptation_sequence(splat_registry, tokens):
    """Test a complex sequence of adaptations to ensure registry remains consistent."""
    # Get initial counts
    initial_count = len(splat_registry.splats)
    
    # Create a set of adaptation operations to test
    adaptations = []
    
    # Find splats for testing, one for each level if possible
    level_splats = {}
    for level in splat_registry.hierarchy.levels:
        level_splats[level] = list(splat_registry.get_splats_at_level(level))
    
    # Add a birth operation
    position = np.mean(tokens, axis=0)
    adaptations.append((AdaptationType.BIRTH, ("Token", position)))
    
    # Add a death operation if we have a Token splat
    if level_splats["Token"]:
        adaptations.append((AdaptationType.DEATH, level_splats["Token"][0]))
    
    # Add a mitosis operation if we have a Phrase splat
    if level_splats["Phrase"]:
        adaptations.append((AdaptationType.MITOSIS, level_splats["Phrase"][0]))
    
    # Add a merge operation if we have at least 2 Document splats
    if len(level_splats["Document"]) >= 2:
        adaptations.append((
            AdaptationType.MERGE, 
            (level_splats["Document"][0], level_splats["Document"][1])
        ))
    
    # Perform the adaptations
    updated_registry, result = perform_adaptations(
        splat_registry=splat_registry,
        adaptations=adaptations,
        tokens=tokens
    )
    
    # Check that the adaptations were tracked
    assert result.birth_count >= 0, "Birth count should be tracked"
    assert result.death_count >= 0, "Death count should be tracked"
    assert result.mitosis_count >= 0, "Mitosis count should be tracked"
    assert result.merge_count >= 0, "Merge count should be tracked"
    
    # Verify that the updated registry is consistent
    assert verify_registry_consistency(updated_registry), "Registry inconsistent after complex adaptations"


def test_find_parent_for_level(splat_registry, tokens):
    """Test that finding a parent for a new splat works correctly."""
    # Create a position
    position = np.mean(tokens, axis=0)
    
    # Try to find a parent for a Token level splat
    token_parent = find_parent_for_level(splat_registry, "Token", position)
    
    # Token level should have Phrase as parent, if Phrase splats exist
    phrase_splats = list(splat_registry.get_splats_at_level("Phrase"))
    if phrase_splats:
        assert token_parent is not None, "Should find a parent for Token level"
        assert token_parent.level == "Phrase", "Parent of Token should be Phrase"
    else:
        assert token_parent is None, "No parent when no Phrase splats exist"
    
    # Try to find a parent for a Document level splat
    document_parent = find_parent_for_level(splat_registry, "Document", position)
    
    # Document is highest level, should have no parent
    assert document_parent is None, "Document level should have no parent"


def test_adaptation_operation_errors(splat_registry, tokens):
    """Test that operations handle errors gracefully."""
    # Test birth with invalid parameters
    invalid_level = "InvalidLevel"
    invalid_birth = perform_birth(
        level=invalid_level,
        position=np.zeros(tokens.shape[1]),
        tokens=tokens,
        splat_registry=splat_registry
    )
    
    # Should return None for invalid level
    assert invalid_birth is None, "Birth with invalid level should return None"
    
    # Test death with invalid splat (not in registry)
    invalid_splat = Splat(
        position=np.zeros(tokens.shape[1]),
        covariance=np.eye(tokens.shape[1]),
        amplitude=1.0,
        level="Token"
    )
    
    invalid_death = perform_death(
        splat=invalid_splat,
        splat_registry=splat_registry
    )
    
    # Should return False for invalid splat
    assert not invalid_death, "Death with invalid splat should return False"
    
    # Test merge with invalid splats
    if len(splat_registry.splats) > 0:
        valid_splat = next(iter(splat_registry.splats.values()))
        
        invalid_merge = perform_merge(
            target_splat=valid_splat,
            source_splat=invalid_splat,
            splat_registry=splat_registry
        )
        
        # Should return False for invalid splat
        assert not invalid_merge, "Merge with invalid splat should return False"


def test_zero_splat_recovery(hierarchy, tokens):
    """Test that the system recovers if all splats are removed."""
    # Create a registry with just one splat
    registry = SplatRegistry(hierarchy)
    splat = Splat(
        position=np.zeros(tokens.shape[1]),
        covariance=np.eye(tokens.shape[1]),
        amplitude=1.0,
        level="Token"
    )
    registry.register(splat)
    
    # Create an adaptation that will remove this splat
    adaptations = [(AdaptationType.DEATH, splat)]
    
    # Perform the adaptation
    updated_registry, result = perform_adaptations(
        splat_registry=registry,
        adaptations=adaptations,
        tokens=tokens
    )
    
    # System should have recovered by creating new splats
    assert len(updated_registry.splats) > 0, "System should recover from zero splats"
    
    # Each level should have at least one splat
    for level in hierarchy.levels:
        level_splats = updated_registry.get_splats_at_level(level)
        assert len(level_splats) > 0, f"Level {level} should have at least one splat after recovery"
    
    # Verify registry consistency
    assert verify_registry_consistency(updated_registry), "Registry inconsistent after recovery"


def test_perform_adaptations_limits(splat_registry, tokens):
    """Test that perform_adaptations applies appropriate limits to operations."""
    # Create many death adaptations
    all_splats = list(splat_registry.splats.values())
    death_adaptations = [(AdaptationType.DEATH, splat) for splat in all_splats]
    
    # Perform the adaptations with low max_death_percentage
    updated_registry, result = perform_adaptations(
        splat_registry=splat_registry,
        adaptations=death_adaptations,
        tokens=tokens,
        max_death_percentage=0.2,  # Only allow 20% of splats to die
        min_level_percentage=0.5   # Maintain at least 50% of initial count per level
    )
    
    # Check that limits were applied
    total_splats = len(splat_registry.splats)
    max_allowed_deaths = int(total_splats * 0.2)
    
    assert result.death_count <= max_allowed_deaths, \
        f"Death count {result.death_count} exceeds limit {max_allowed_deaths}"
    
    # Make sure each level still has splats
    for level in splat_registry.hierarchy.levels:
        level_splats = updated_registry.get_splats_at_level(level)
        assert len(level_splats) > 0, f"Level {level} has no splats after adaptation"
    
    # Verify registry consistency
    assert verify_registry_consistency(updated_registry), "Registry inconsistent after limited adaptations"
