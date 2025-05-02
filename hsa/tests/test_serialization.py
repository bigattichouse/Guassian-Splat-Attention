"""
Tests for serialization functionality in Hierarchical Splat Attention (HSA).

This test module verifies that we can properly serialize a SplatRegistry to
various formats and deserialize it back with all relationships intact.
"""

import pytest
import numpy as np
import os
import tempfile
from io import BytesIO

from hsa.serialization_core import HSASerializer
from hsa.serialization_formats import save_to_file, load_from_file, compress_registry, clone_registry
from hsa.serialization_delta import compute_registry_delta, apply_delta
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy


@pytest.fixture
def simple_hierarchy():
    """Create a simple hierarchy for testing."""
    return Hierarchy(
        levels=["token", "sentence", "document"],
        init_splats_per_level=[10, 5, 2],
        level_weights=[0.6, 0.3, 0.1]
    )


@pytest.fixture
def simple_registry(simple_hierarchy):
    """Create a simple registry with a few splats for testing."""
    registry = SplatRegistry(hierarchy=simple_hierarchy, embedding_dim=4)
    
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
    
    # Create sentence-level splat with parent
    sent_splat = Splat(
        dim=4,
        position=np.array([1.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(4) * 1.5,
        amplitude=1.0,
        level="sentence",
        parent=doc_splat,
        id="sent_splat_1"
    )
    registry.register(sent_splat)
    
    # Create token-level splats with parent
    token_splat1 = Splat(
        dim=4,
        position=np.array([0.5, 0.5, 0.0, 0.0]),
        covariance=np.eye(4),
        amplitude=1.0,
        level="token",
        parent=sent_splat,
        id="token_splat_1"
    )
    registry.register(token_splat1)
    
    token_splat2 = Splat(
        dim=4,
        position=np.array([1.5, 0.5, 0.0, 0.0]),
        covariance=np.eye(4),
        amplitude=1.0,
        level="token",
        parent=sent_splat,
        id="token_splat_2"
    )
    registry.register(token_splat2)
    
    return registry


@pytest.fixture
def serializer():
    """Create a serializer instance for testing."""
    return HSASerializer(compression_level=6)


def test_serializer_initialization():
    """Test that the serializer can be initialized with different compression levels."""
    serializer = HSASerializer()
    assert serializer.compression_level == 6  # Default
    
    serializer = HSASerializer(compression_level=0)
    assert serializer.compression_level == 0
    
    serializer = HSASerializer(compression_level=9)
    assert serializer.compression_level == 9


def test_serialize_registry(simple_registry, serializer):
    """Test that a registry can be serialized to bytes."""
    # Serialize registry
    data = serializer.serialize_registry(simple_registry)
    
    # Check result
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_deserialize_registry(simple_registry, serializer):
    """Test that a registry can be deserialized from bytes."""
    # Serialize registry
    data = serializer.serialize_registry(simple_registry)
    
    # Deserialize registry
    registry = serializer.deserialize_registry(data)
    
    # Check result
    assert isinstance(registry, SplatRegistry)
    assert registry.embedding_dim == simple_registry.embedding_dim
    assert registry.hierarchy.levels == simple_registry.hierarchy.levels
    assert registry.count_splats() == simple_registry.count_splats()


def test_serialization_preserves_relationships(simple_registry, serializer):
    """Test that serialization preserves parent-child relationships between splats."""
    # Get original relationships
    doc_splat = simple_registry.get_splat("doc_splat_1")
    sent_splat = simple_registry.get_splat("sent_splat_1")
    token_splat1 = simple_registry.get_splat("token_splat_1")
    token_splat2 = simple_registry.get_splat("token_splat_2")
    
    # Verify original relationships
    assert doc_splat.parent is None
    assert sent_splat.parent == doc_splat
    assert token_splat1.parent == sent_splat
    assert token_splat2.parent == sent_splat
    assert sent_splat in doc_splat.children
    assert token_splat1 in sent_splat.children
    assert token_splat2 in sent_splat.children
    
    # Serialize and deserialize
    data = serializer.serialize_registry(simple_registry)
    registry = serializer.deserialize_registry(data)
    
    # Get deserialized splats
    doc_splat_new = registry.get_splat("doc_splat_1")
    sent_splat_new = registry.get_splat("sent_splat_1")
    token_splat1_new = registry.get_splat("token_splat_1")
    token_splat2_new = registry.get_splat("token_splat_2")
    
    # Verify deserialized relationships
    assert doc_splat_new.parent is None
    assert sent_splat_new.parent == doc_splat_new
    assert token_splat1_new.parent == sent_splat_new
    assert token_splat2_new.parent == sent_splat_new
    assert sent_splat_new in doc_splat_new.children
    assert token_splat1_new in sent_splat_new.children
    assert token_splat2_new in sent_splat_new.children


def test_serialization_preserves_parameters(simple_registry, serializer):
    """Test that serialization preserves splat parameters."""
    # Get original parameters
    doc_splat = simple_registry.get_splat("doc_splat_1")
    original_pos = doc_splat.position.copy()
    original_cov = doc_splat.covariance.copy()
    original_amp = doc_splat.amplitude
    
    # Serialize and deserialize
    data = serializer.serialize_registry(simple_registry)
    registry = serializer.deserialize_registry(data)
    
    # Get deserialized parameters
    doc_splat_new = registry.get_splat("doc_splat_1")
    new_pos = doc_splat_new.position
    new_cov = doc_splat_new.covariance
    new_amp = doc_splat_new.amplitude
    
    # Verify deserialized parameters
    assert np.array_equal(new_pos, original_pos)
    assert np.array_equal(new_cov, original_cov)
    assert new_amp == original_amp


def test_save_load_file_binary(simple_registry, serializer):
    """Test saving and loading registry to/from a binary file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        filepath = f.name
    
    try:
        # Save registry to file
        save_to_file(serializer, simple_registry, filepath, format="binary")
        
        # Check file exists and has content
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        
        # Load registry from file
        registry = load_from_file(serializer, filepath)
        
        # Check loaded registry
        assert isinstance(registry, SplatRegistry)
        assert registry.embedding_dim == simple_registry.embedding_dim
        assert registry.count_splats() == simple_registry.count_splats()
        
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_load_file_json(simple_registry, serializer):
    """Test saving and loading registry to/from a JSON file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name
    
    try:
        # Save registry to file
        save_to_file(serializer, simple_registry, filepath, format="json")
        
        # Check file exists and has content
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        
        # Load registry from file
        registry = load_from_file(serializer, filepath)
        
        # Check loaded registry
        assert isinstance(registry, SplatRegistry)
        assert registry.embedding_dim == simple_registry.embedding_dim
        assert registry.count_splats() == simple_registry.count_splats()
        
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)


def test_compress_registry(simple_registry, serializer):
    """Test compressing a registry to a minimal representation."""
    # Compress registry
    data = compress_registry(serializer, simple_registry)
    
    # Check result
    assert isinstance(data, bytes)
    assert len(data) > 0
    
    # Deserialize compressed data
    registry = serializer.deserialize_registry(data)
    
    # Check result
    assert isinstance(registry, SplatRegistry)
    assert registry.embedding_dim == simple_registry.embedding_dim
    assert registry.count_splats() == simple_registry.count_splats()


def test_clone_registry(simple_registry):
    """Test creating a deep copy of a registry."""
    # Clone registry
    cloned = clone_registry(simple_registry)
    
    # Check cloned registry
    assert isinstance(cloned, SplatRegistry)
    assert cloned.embedding_dim == simple_registry.embedding_dim
    assert cloned.count_splats() == simple_registry.count_splats()
    
    # Verify they are not the same object
    assert cloned is not simple_registry
    
    # Modify original registry
    doc_splat = simple_registry.get_splat("doc_splat_1")
    doc_splat.position[0] = 5.0
    
    # Verify cloned registry was not affected
    cloned_doc_splat = cloned.get_splat("doc_splat_1")
    assert cloned_doc_splat.position[0] != 5.0


def test_compute_apply_delta(simple_registry, serializer):
    """Test computing and applying deltas between registry versions."""
    # Clone the registry for modification
    modified = clone_registry(simple_registry)
    
    # Modify the clone
    doc_splat = modified.get_splat("doc_splat_1")
    doc_splat.position[0] = 5.0
    
    # Add a new splat
    new_splat = Splat(
        dim=4,
        position=np.array([2.0, 2.0, 2.0, 2.0]),
        covariance=np.eye(4),
        amplitude=1.0,
        level="token",
        parent=modified.get_splat("sent_splat_1"),
        id="token_splat_3"
    )
    modified.register(new_splat)
    
    # Compute delta
    delta = compute_registry_delta(serializer, simple_registry, modified)
    
    # Check delta structure
    assert "added_splats" in delta
    assert "modified_splats" in delta
    assert len(delta["added_splats"]) == 1
    assert len(delta["modified_splats"]) == 1
    
    # Apply delta
    updated = apply_delta(serializer, simple_registry, delta)
    
    # Verify updated registry
    assert updated.count_splats() == modified.count_splats()
    
    # Check modified splat
    updated_doc_splat = updated.get_splat("doc_splat_1")
    assert updated_doc_splat.position[0] == 5.0
    
    # Check added splat
    updated_new_splat = updated.get_splat("token_splat_3")
    assert updated_new_splat is not None
    assert np.array_equal(updated_new_splat.position, np.array([2.0, 2.0, 2.0, 2.0]))
    assert updated_new_splat.parent == updated.get_splat("sent_splat_1")


def test_serialization_with_history(simple_registry, serializer):
    """Test that serialization preserves activation history."""
    # Add some activation history
    token_splat = simple_registry.get_splat("token_splat_1")
    token_splat.activation_history.add(0.5)
    token_splat.activation_history.add(0.7)
    token_splat.activation_history.add(0.9)
    
    # Verify original history
    original_history = token_splat.activation_history.get_values()
    assert len(original_history) == 3
    
    # Serialize and deserialize
    data = serializer.serialize_registry(simple_registry)
    registry = serializer.deserialize_registry(data)
    
    # Get deserialized splat
    token_splat_new = registry.get_splat("token_splat_1")
    
    # Verify deserialized history
    new_history = token_splat_new.activation_history.get_values()
    assert len(new_history) == len(original_history)
    assert all(a == b for a, b in zip(new_history, original_history))


def test_version_compatibility(simple_registry):
    """Test version compatibility checking."""
    # Create serializers with different versions
    serializer_10 = HSASerializer()
    serializer_10.VERSION = "1.0.0"
    
    serializer_11 = HSASerializer()
    serializer_11.VERSION = "1.1.0"
    
    serializer_20 = HSASerializer()
    serializer_20.VERSION = "2.0.0"
    
    # Serialize with 1.0.0
    data_10 = serializer_10.serialize_registry(simple_registry)
    
    # Should deserialize with 1.1.0 (backward compatible)
    registry_11 = serializer_11.deserialize_registry(data_10)
    assert isinstance(registry_11, SplatRegistry)
    
    # Should not deserialize with 2.0.0 (major version change)
    with pytest.raises(Exception):
        serializer_20.deserialize_registry(data_10)


def test_serialization_large_registry(serializer):
    """Test serialization with a larger registry."""
    # Create a larger registry with many splats
    hierarchy = Hierarchy(
        levels=["token", "phrase", "sentence", "paragraph", "document"],
        init_splats_per_level=[50, 20, 10, 5, 2],
        level_weights=[0.4, 0.3, 0.2, 0.07, 0.03]
    )
    
    registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=8)
    
    # Add some splats at document level
    for i in range(2):
        doc_splat = Splat(
            dim=8,
            position=np.random.randn(8),
            covariance=np.eye(8) * 2.0,
            amplitude=1.0,
            level="document",
            id=f"doc_splat_{i}"
        )
        registry.register(doc_splat)
        
        # Add some paragraph splats
        for j in range(3):
            para_splat = Splat(
                dim=8,
                position=np.random.randn(8),
                covariance=np.eye(8) * 1.5,
                amplitude=1.0,
                level="paragraph",
                parent=doc_splat,
                id=f"para_splat_{i}_{j}"
            )
            registry.register(para_splat)
            
            # Add some sentence splats
            for k in range(4):
                sent_splat = Splat(
                    dim=8,
                    position=np.random.randn(8),
                    covariance=np.eye(8) * 1.2,
                    amplitude=1.0,
                    level="sentence",
                    parent=para_splat,
                    id=f"sent_splat_{i}_{j}_{k}"
                )
                registry.register(sent_splat)
    
    # Register a few more splats to get to about 50
    for i in range(20):
        token_splat = Splat(
            dim=8,
            position=np.random.randn(8),
            covariance=np.eye(8),
            amplitude=1.0,
            level="token",
            id=f"token_splat_{i}"
        )
        registry.register(token_splat)
    
    # Verify we have a decent number of splats
    assert registry.count_splats() > 40
    
    # Serialize and deserialize
    data = serializer.serialize_registry(registry)
    deserialized = serializer.deserialize_registry(data)
    
    # Verify
    assert deserialized.count_splats() == registry.count_splats()
    assert deserialized.hierarchy.levels == registry.hierarchy.levels
    
    # Check if registry integrity is maintained
    assert deserialized.verify_integrity()
