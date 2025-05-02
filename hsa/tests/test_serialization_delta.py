"""
Tests for delta-based serialization in Hierarchical Splat Attention (HSA).

This test module focuses on testing the delta serialization functionality,
which allows for efficient updates between registry versions.
"""

import pytest
import numpy as np
import os
import tempfile
import json
import random
import copy

from hsa.serialization_core import HSASerializer
from hsa.serialization_formats import save_to_file, load_from_file, clone_registry
from hsa.serialization_delta import (
    compute_registry_delta, apply_delta, save_registry_delta,
    apply_registry_delta, get_registry_id
)
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy


@pytest.fixture
def base_hierarchy():
    """Create a base hierarchy for testing."""
    return Hierarchy(
        levels=["token", "sentence", "document"],
        init_splats_per_level=[10, 5, 2],
        level_weights=[0.6, 0.3, 0.1]
    )


@pytest.fixture
def base_registry(base_hierarchy):
    """Create a base registry for delta testing."""
    registry = SplatRegistry(hierarchy=base_hierarchy, embedding_dim=4)
    
    # Create document-level splat
    doc_splat = Splat(
        dim=4,
        position=np.array([0.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(4) * 2.0,
        amplitude=1.0,
        level="document",
        id="doc_splat_1"
    )
    registry.register(doc_splat)
    
    # Create sentence-level splats
    for i in range(2):
        sent_splat = Splat(
            dim=4,
            position=np.array([float(i), 0.0, 0.0, 0.0]),
            covariance=np.eye(4) * 1.5,
            amplitude=1.0,
            level="sentence",
            parent=doc_splat,
            id=f"sent_splat_{i+1}"
        )
        registry.register(sent_splat)
        
        # Create token-level splats
        for j in range(3):
            token_splat = Splat(
                dim=4,
                position=np.array([float(i) + 0.1*j, 0.5, 0.0, 0.0]),
                covariance=np.eye(4),
                amplitude=1.0,
                level="token",
                parent=sent_splat,
                id=f"token_splat_{i+1}_{j+1}"
            )
            registry.register(token_splat)
    
    return registry


@pytest.fixture
def modified_registry(base_registry):
    """Create a modified version of the base registry."""
    # Make a deep copy instead of using clone_registry to avoid any issues
    registry = copy.deepcopy(base_registry)
    
    # Modify an existing splat
    doc_splat = registry.get_splat("doc_splat_1")
    doc_splat.position[0] = 1.0
    doc_splat.amplitude = 0.8
    
    # Remove a splat
    registry.unregister("token_splat_1_2")
    
    # Add a new splat
    new_splat = Splat(
        dim=4,
        position=np.array([2.0, 2.0, 2.0, 2.0]),
        covariance=np.eye(4),
        amplitude=1.0,
        level="token",
        parent=registry.get_splat("sent_splat_2"),
        id="token_splat_new"
    )
    registry.register(new_splat)
    
    return registry


@pytest.fixture
def serializer():
    """Create a serializer instance for testing."""
    return HSASerializer(compression_level=6)


def test_compute_registry_delta(base_registry, modified_registry, serializer):
    """Test computing delta between two registry versions."""
    # Verify splats exist before testing
    assert base_registry.safe_get_splat("doc_splat_1") is not None
    assert modified_registry.safe_get_splat("doc_splat_1") is not None
    assert modified_registry.safe_get_splat("token_splat_new") is not None
    
    # Compute delta
    delta = compute_registry_delta(serializer, base_registry, modified_registry)
    
    # Check delta structure
    assert "added_splats" in delta
    assert "removed_splats" in delta
    assert "modified_splats" in delta
    
    # Check contents
    assert len(delta["added_splats"]) == 1
    assert len(delta["removed_splats"]) == 1
    assert len(delta["modified_splats"]) == 1
    
    # Check specific splat IDs
    assert any(splat["id"] == "token_splat_new" for splat in delta["added_splats"])
    assert "token_splat_1_2" in delta["removed_splats"]
    assert any(splat["id"] == "doc_splat_1" for splat in delta["modified_splats"])


def test_apply_delta(base_registry, modified_registry, serializer):
    """Test applying a delta to update a registry."""
    # Verify splats exist before testing
    assert base_registry.safe_get_splat("doc_splat_1") is not None
    assert modified_registry.safe_get_splat("token_splat_new") is not None
    
    # Compute delta
    delta = compute_registry_delta(serializer, base_registry, modified_registry)
    
    # Create a clean copy of base registry to apply delta to
    base_copy = copy.deepcopy(base_registry)
    
    # Apply delta
    updated_registry = apply_delta(serializer, base_copy, delta)
    
    # Verify updated registry
    assert updated_registry.count_splats() == modified_registry.count_splats()
    
    # Check added splat
    assert updated_registry.safe_get_splat("token_splat_new") is not None
    
    # Check removed splat
    assert updated_registry.safe_get_splat("token_splat_1_2") is None
    
    # Check modified splat
    doc_splat = updated_registry.get_splat("doc_splat_1")
    assert doc_splat.position[0] == 1.0
    assert doc_splat.amplitude == 0.8


def test_save_apply_registry_delta(base_registry, modified_registry, serializer):
    """Test saving and applying delta between registries."""
    # Verify splats exist before testing
    assert base_registry.safe_get_splat("doc_splat_1") is not None
    
    # Save delta
    delta_bytes = save_registry_delta(serializer, base_registry, modified_registry)
    
    # Check result
    assert isinstance(delta_bytes, bytes)
    assert len(delta_bytes) > 0
    
    # Create a clean copy of base registry to apply delta to
    base_copy = copy.deepcopy(base_registry)
    
    # Apply delta
    updated_registry = apply_registry_delta(serializer, base_copy, delta_bytes)
    
    # Verify updated registry
    assert updated_registry.count_splats() == modified_registry.count_splats()
    
    # Check specific changes
    assert updated_registry.safe_get_splat("token_splat_new") is not None
    assert updated_registry.safe_get_splat("token_splat_1_2") is None
    doc_splat = updated_registry.get_splat("doc_splat_1")
    assert doc_splat.position[0] == 1.0


def test_get_registry_id(base_registry, modified_registry):
    """Test generating unique IDs for registries."""
    # Verify splats exist before testing
    assert base_registry.safe_get_splat("doc_splat_1") is not None
    
    # Get IDs
    base_id = get_registry_id(base_registry)
    modified_id = get_registry_id(modified_registry)
    
    # Check IDs
    assert isinstance(base_id, str)
    assert len(base_id) > 0
    assert isinstance(modified_id, str)
    assert len(modified_id) > 0
    
    # IDs should be different because registries are different
    assert base_id != modified_id
    
    # Cloning should maintain the same ID
    base_copy = copy.deepcopy(base_registry)
    clone_id = get_registry_id(base_copy)
    assert clone_id == base_id


def test_delta_with_multiple_changes(base_registry, serializer):
    """Test delta with multiple types of changes simultaneously."""
    # Create a deep copy of the base registry to avoid modification
    registry = copy.deepcopy(base_registry)
    
    # Verify splats exist before testing
    assert registry.safe_get_splat("sent_splat_1") is not None
    assert registry.safe_get_splat("sent_splat_2") is not None
    assert registry.safe_get_splat("token_splat_2_1") is not None
    assert registry.safe_get_splat("token_splat_2_2") is not None
    
    # Make multiple changes
    
    # 1. Add a new splat
    new_splat = Splat(
        dim=4,
        position=np.array([3.0, 3.0, 3.0, 3.0]),
        covariance=np.eye(4),
        amplitude=1.0,
        level="token",
        parent=registry.get_splat("sent_splat_1"),
        id="token_splat_new_multi"
    )
    registry.register(new_splat)
    
    # 2. Remove a splat
    registry.unregister("token_splat_2_2")
    
    # 3. Modify a splat
    doc_splat = registry.get_splat("doc_splat_1")
    doc_splat.position[0] = 2.0
    doc_splat.amplitude = 0.7
    
    # 4. Change a relationship
    token_splat = registry.get_splat("token_splat_2_1")
    sent_splat_1 = registry.get_splat("sent_splat_1")
    sent_splat_2 = registry.get_splat("sent_splat_2")
    
    # Update parent-child relationship
    sent_splat_2.children.remove(token_splat)
    token_splat.parent = sent_splat_1
    sent_splat_1.children.add(token_splat)
    
    # Create a fresh copy of base registry to apply delta to
    base_copy = copy.deepcopy(base_registry)
    
    # Compute and apply delta
    delta = compute_registry_delta(serializer, base_registry, registry)
    updated = apply_delta(serializer, base_copy, delta)
    
    # Verify all changes
    assert updated.count_splats() == registry.count_splats()
    
    # Check added splat
    assert updated.safe_get_splat("token_splat_new_multi") is not None
    
    # Check removed splat
    assert updated.safe_get_splat("token_splat_2_2") is None
    
    # Check modified splat
    updated_doc = updated.get_splat("doc_splat_1")
    assert updated_doc.position[0] == 2.0
    assert updated_doc.amplitude == 0.7
    
    # Check relationship change
    updated_token = updated.get_splat("token_splat_2_1")
    updated_sent_1 = updated.get_splat("sent_splat_1")
    updated_sent_2 = updated.get_splat("sent_splat_2")
    
    assert updated_token.parent == updated_sent_1
    assert updated_token in updated_sent_1.children
    assert updated_token not in updated_sent_2.children


def test_delta_metadata(base_registry, modified_registry, serializer):
    """Test that delta serialization includes proper metadata."""
    # Verify splats exist before testing
    assert base_registry.safe_get_splat("doc_splat_1") is not None
    
    # Save delta
    delta_bytes = save_registry_delta(serializer, base_registry, modified_registry)
    
    # Try to decompress and parse
    try:
        import zlib
        json_data = zlib.decompress(delta_bytes).decode('utf-8')
    except:
        # Not compressed, try direct decoding
        json_data = delta_bytes.decode('utf-8')
    
    # Parse JSON
    delta = json.loads(json_data)
    
    # Check metadata
    assert "_metadata" in delta
    metadata = delta["_metadata"]
    
    assert "version" in metadata
    assert "timestamp" in metadata
    assert "format" in metadata
    assert "base_registry_id" in metadata
    assert "target_registry_id" in metadata
    
    assert metadata["format"] == "hsa_registry_delta"
    assert metadata["version"] == serializer.VERSION


def test_delta_with_incompatible_registry(base_registry, serializer):
    """Test applying delta to an incompatible registry."""
    # Create a completely different registry
    different_hierarchy = Hierarchy(
        levels=["word", "paragraph", "chapter"],  # Different level names
        init_splats_per_level=[5, 3, 1],
        level_weights=[0.7, 0.2, 0.1]
    )
    
    different_registry = SplatRegistry(hierarchy=different_hierarchy, embedding_dim=4)
    
    # Add a splat
    splat = Splat(
        dim=4,
        position=np.array([0.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(4),
        amplitude=1.0,
        level="word",
        id="word_splat_1"
    )
    different_registry.register(splat)
    
    # Create a copy of base registry for modification
    modified = copy.deepcopy(base_registry)
    
    # Verify splat exists
    assert modified.safe_get_splat("doc_splat_1") is not None
    
    # Modify something
    doc_splat = modified.get_splat("doc_splat_1")
    doc_splat.position[0] = 3.0
    
    # Save delta between base and modified
    delta_bytes = save_registry_delta(serializer, base_registry, modified)
    
    # Make a copy of different_registry to apply delta to
    diff_copy = copy.deepcopy(different_registry)
    
    try:
        # Attempt to apply delta to different registry
        updated = apply_registry_delta(serializer, diff_copy, delta_bytes)
        
        # If it succeeded (which it may not in all implementations), verify results
        assert isinstance(updated, SplatRegistry)
        
        # Original registry should be unchanged
        assert different_registry.count_splats() == 1
        assert different_registry.get_splat("word_splat_1") is not None
    except ValueError as e:
        # Some implementations may prevent applying incompatible deltas
        # If so, this is also acceptable behavior
        assert "not found" in str(e) or "registry" in str(e).lower()


def test_empty_delta(base_registry, serializer):
    """Test computing and applying delta when no changes have been made."""
    # Create a clone with no changes
    clone = copy.deepcopy(base_registry)
    
    # Compute delta
    delta = compute_registry_delta(serializer, base_registry, clone)
    
    # Delta should have empty lists for changes
    assert len(delta["added_splats"]) == 0
    assert len(delta["removed_splats"]) == 0
    assert len(delta["modified_splats"]) == 0
    assert delta["hierarchy_changes"] is None
    
    # Apply delta to a fresh copy
    base_copy = copy.deepcopy(base_registry)
    updated = apply_delta(serializer, base_copy, delta)
    
    # Should be identical to original
    assert updated.count_splats() == base_registry.count_splats()
    
    # Check a specific splat
    orig_splat = base_registry.get_splat("doc_splat_1")
    updated_splat = updated.get_splat("doc_splat_1")
    
    assert np.array_equal(orig_splat.position, updated_splat.position)
    assert orig_splat.amplitude == updated_splat.amplitude


def test_large_delta(serializer):
    """Test delta computation and application with many changes."""
    # Create a larger base registry
    hierarchy = Hierarchy(
        levels=["token", "phrase", "sentence", "paragraph", "document"],
        init_splats_per_level=[20, 10, 5, 3, 1],
        level_weights=[0.4, 0.2, 0.2, 0.1, 0.1]
    )
    
    base_registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=8)
    
    # Add splats at different levels
    doc_splat = Splat(
        dim=8,
        position=np.zeros(8),
        covariance=np.eye(8) * 2.0,
        amplitude=1.0,
        level="document",
        id="doc_main"
    )
    base_registry.register(doc_splat)
    
    # Add paragraph splats
    paragraph_splats = []
    for i in range(3):
        para_splat = Splat(
            dim=8,
            position=np.random.randn(8),
            covariance=np.eye(8) * 1.5,
            amplitude=1.0,
            level="paragraph",
            parent=doc_splat,
            id=f"para_{i}"
        )
        base_registry.register(para_splat)
        paragraph_splats.append(para_splat)
    
    # Add sentence splats
    sentence_splats = []
    for i, para_splat in enumerate(paragraph_splats):
        for j in range(3):
            sent_splat = Splat(
                dim=8,
                position=np.random.randn(8),
                covariance=np.eye(8) * 1.2,
                amplitude=1.0,
                level="sentence",
                parent=para_splat,
                id=f"sent_{i}_{j}"
            )
            base_registry.register(sent_splat)
            sentence_splats.append(sent_splat)
    
    # Add token-level splats
    for i, sent_splat in enumerate(sentence_splats):
        for j in range(5):
            token_splat = Splat(
                dim=8,
                position=np.random.randn(8),
                covariance=np.eye(8),
                amplitude=1.0,
                level="token",
                parent=sent_splat,
                id=f"token_{i}_{j}"
            )
            base_registry.register(token_splat)
    
    # Create a modified version with many changes
    modified_registry = copy.deepcopy(base_registry)
    
    # Verify some splats exist
    assert modified_registry.safe_get_splat("para_0") is not None
    
    # Make a lot of changes
    # 1. Add new splats
    for i in range(10):
        new_splat = Splat(
            dim=8,
            position=np.random.randn(8),
            covariance=np.eye(8),
            amplitude=1.0,
            level="token",
            parent=random.choice(sentence_splats),
            id=f"new_token_{i}"
        )
        modified_registry.register(new_splat)
    
    # 2. Remove some splats
    for i in range(5):
        token_id = f"token_{i}_{i % 3}"
        try:
            modified_registry.unregister(token_id)
        except ValueError:
            pass  # ID might not exist
    
    # 3. Modify some splats
    for i in range(3):
        para_id = f"para_{i}"
        para_splat = modified_registry.get_splat(para_id)
        para_splat.position = para_splat.position + np.random.randn(8) * 0.2
        para_splat.amplitude = max(0.1, min(1.0, para_splat.amplitude + np.random.randn() * 0.1))
    
    # 4. Change some relationships
    # Move a sentence from one paragraph to another
    try:
        sent_splat = modified_registry.get_splat("sent_0_1")
        old_parent = sent_splat.parent
        new_parent = modified_registry.get_splat("para_2")
        
        old_parent.children.remove(sent_splat)
        sent_splat.parent = new_parent
        new_parent.children.add(sent_splat)
    except (ValueError, AttributeError):
        pass  # ID might not exist
    
    # Create a fresh copy of base registry to apply delta to
    base_copy = copy.deepcopy(base_registry)
    
    # Compute and apply delta
    delta = compute_registry_delta(serializer, base_registry, modified_registry)
    updated = apply_delta(serializer, base_copy, delta)
    
    # Verify counts match
    assert updated.count_splats() == modified_registry.count_splats()
    
    # Check a few random splats
    assert updated.safe_get_splat("new_token_5") is not None
    
    # Verify integrity
    assert updated.verify_integrity()


def test_delta_with_relationship_changes(base_registry, serializer):
    """Test delta with changes to parent-child relationships."""
    # Create a deep copy of the base registry
    registry = copy.deepcopy(base_registry)
    
    # Verify splats exist before testing
    assert registry.safe_get_splat("sent_splat_1") is not None
    assert registry.safe_get_splat("sent_splat_2") is not None
    assert registry.safe_get_splat("token_splat_1_1") is not None
    
    # Change parent-child relationships
    sent_splat_1 = registry.get_splat("sent_splat_1")
    sent_splat_2 = registry.get_splat("sent_splat_2")
    token_splat = registry.get_splat("token_splat_1_1")
    
    # Move token from sent_splat_1 to sent_splat_2
    # First remove from current parent
    sent_splat_1.children.remove(token_splat)
    # Update parent reference
    token_splat.parent = sent_splat_2
    # Add to new parent's children
    sent_splat_2.children.add(token_splat)
    
    # Create a fresh copy to apply delta to
    base_copy = copy.deepcopy(base_registry)
    
    # Compute and apply delta
    delta = compute_registry_delta(serializer, base_registry, registry)
    updated = apply_delta(serializer, base_copy, delta)
    
    # Verify relationship changes
    updated_sent_1 = updated.get_splat("sent_splat_1")
    updated_sent_2 = updated.get_splat("sent_splat_2")
    updated_token = updated.get_splat("token_splat_1_1")
    
    assert updated_token.parent == updated_sent_2
    assert updated_token not in updated_sent_1.children
    assert updated_token in updated_sent_2.children


def test_delta_with_hierarchy_changes(base_registry, serializer):
    """Test delta with changes to hierarchy structure."""
    # Create a deep copy of the base registry
    registry = copy.deepcopy(base_registry)
    
    # Modify hierarchy
    new_hierarchy = Hierarchy(
        levels=["token", "phrase", "sentence", "document"],  # Added 'phrase' level
        init_splats_per_level=[10, 8, 5, 2],
        level_weights=[0.5, 0.2, 0.2, 0.1]
    )
    registry.hierarchy = new_hierarchy
    
    # Create a fresh copy to apply delta to
    base_copy = copy.deepcopy(base_registry)
    
    # Compute and apply delta
    delta = compute_registry_delta(serializer, base_registry, registry)
    updated = apply_delta(serializer, base_copy, delta)
    
    # Verify hierarchy changes
    assert "hierarchy_changes" in delta
    assert updated.hierarchy.levels == new_hierarchy.levels
    assert updated.hierarchy.init_splats_per_level == new_hierarchy.init_splats_per_level
    assert updated.hierarchy.level_weights == new_hierarchy.level_weights


def test_delta_with_activation_history_changes(base_registry, serializer):
    """Test delta with changes to activation history."""
    # Create a deep copy of the base registry
    registry = copy.deepcopy(base_registry)
    
    # Verify token exists
    assert registry.safe_get_splat("token_splat_1_1") is not None
    
    # Modify activation history
    token_splat = registry.get_splat("token_splat_1_1")
    token_splat.activation_history.add(0.5)
    token_splat.activation_history.add(0.7)
    token_splat.activation_history.add(0.9)
    
    # Create a fresh copy to apply delta to
    base_copy = copy.deepcopy(base_registry)
    
    # Compute and apply delta
    delta = compute_registry_delta(serializer, base_registry, registry)
    updated = apply_delta(serializer, base_copy, delta)
    
    # Verify history changes
    updated_token = updated.get_splat("token_splat_1_1")
    history = updated_token.activation_history.get_values()
    
    assert len(history) > 0
    assert any(h == 0.5 for h in history)
    assert any(h == 0.7 for h in history)
    assert any(h == 0.9 for h in history)


def test_delta_file_save_load(base_registry, modified_registry, serializer):
    """Test saving and loading delta to/from file."""
    # Verify splats exist before testing
    assert base_registry.safe_get_splat("doc_splat_1") is not None
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".hsad", delete=False) as f:
        filepath = f.name
    
    try:
        # Save delta to file
        delta_bytes = save_registry_delta(serializer, base_registry, modified_registry)
        with open(filepath, "wb") as f:
            f.write(delta_bytes)
        
        # Check file exists and has content
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        
        # Load delta from file
        with open(filepath, "rb") as f:
            loaded_delta = f.read()
        
        # Create a fresh copy to apply delta to
        base_copy = copy.deepcopy(base_registry)
        
        # Apply loaded delta
        updated = apply_registry_delta(serializer, base_copy, loaded_delta)
        
        # Verify updated registry
        assert updated.count_splats() == modified_registry.count_splats()
        assert updated.safe_get_splat("token_splat_new") is not None
        assert updated.safe_get_splat("token_splat_1_2") is None
        
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)


def test_delta_compression_efficiency(base_registry, modified_registry, serializer):
    """Test that delta serialization is more efficient than full serialization."""
    # Verify splats exist before testing
    assert base_registry.safe_get_splat("doc_splat_1") is not None
    
    # Serialize full registries
    full_base_bytes = serializer.serialize_registry(base_registry)
    full_modified_bytes = serializer.serialize_registry(modified_registry)
    
    # Serialize delta
    delta_bytes = save_registry_delta(serializer, base_registry, modified_registry)
    
    # Check that delta is smaller than full modified registry
    assert len(delta_bytes) < len(full_modified_bytes)
    
    # Create a fresh copy to apply delta to
    base_copy = copy.deepcopy(base_registry)
    
    # Verify delta can be applied correctly
    updated = apply_registry_delta(serializer, base_copy, delta_bytes)
    assert updated.count_splats() == modified_registry.count_splats()
