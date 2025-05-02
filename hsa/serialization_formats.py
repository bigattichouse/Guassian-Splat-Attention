"""
Format-specific serialization logic for Hierarchical Splat Attention (HSA).

This module provides utilities for handling different serialization formats
(binary, JSON, pickle) and file operations.
"""

import os
import json
import zlib
import pickle
import logging
from typing import Dict, List, Optional, Any, Union, BinaryIO

from .splat import Splat
from .registry import SplatRegistry
from .hierarchy import Hierarchy
from .serialization_core import HSASerializer

# Configure logging
logger = logging.getLogger(__name__)


def compress_data(data: bytes, compression_level: int = 6) -> bytes:
    """Compress data using zlib.
    
    Args:
        data: Data to compress
        compression_level: Compression level (0-9)
        
    Returns:
        Compressed data
    """
    return zlib.compress(data, compression_level)


def decompress_data(data: bytes) -> str:
    """Decompress data using zlib.
    
    Args:
        data: Compressed data
        
    Returns:
        Decompressed data as string
        
    Raises:
        zlib.error: If decompression fails
    """
    return zlib.decompress(data).decode('utf-8')


def save_to_file(
    serializer: HSASerializer,
    registry: SplatRegistry,
    path: str,
    format: str = "binary"
) -> None:
    """Save a registry to a file.
    
    Args:
        serializer: HSASerializer instance
        registry: SplatRegistry to save
        path: File path to save to
        format: File format (binary, json, or pickle)
        
    Raises:
        ValueError: If format is invalid
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    if format == "binary":
        # Serialize to binary
        data = serializer.serialize_registry(registry)
        
        # Write to file
        with open(path, 'wb') as f:
            f.write(data)
    elif format == "json":
        # Convert to dictionary
        data = serializer._registry_to_dict(registry)
        
        # Add metadata
        from datetime import datetime
        data["_metadata"] = {
            "version": serializer.VERSION,
            "timestamp": datetime.now().isoformat(),
            "format": "hsa_registry"
        }
        
        # Write to file
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == "pickle":
        # Write to file
        with open(path, 'wb') as f:
            pickle.dump(registry, f)
    else:
        raise ValueError(f"Invalid format: {format}")


def load_from_file(
    serializer: HSASerializer,
    path: str
) -> SplatRegistry:
    """Load a registry from a file.
    
    Args:
        serializer: HSASerializer instance
        path: File path to load from
        
    Returns:
        Loaded SplatRegistry
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Try to determine format from extension
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.json':
        # Read JSON file
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Check metadata
        metadata = data.get("_metadata", {})
        format_type = metadata.get("format", "")
        
        if format_type != "hsa_registry":
            raise ValueError(f"Invalid format: {format_type}")
        
        # Convert to registry
        return serializer._dict_to_registry(data)
    elif ext == '.pickle' or ext == '.pkl':
        # Read pickle file
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        # Assume binary format
        with open(path, 'rb') as f:
            data = f.read()
        
        return serializer.deserialize_registry(data)


def compress_registry(
    serializer: HSASerializer,
    registry: SplatRegistry
) -> bytes:
    """Compress a registry to a minimal representation.
    
    Args:
        serializer: HSASerializer instance
        registry: SplatRegistry to compress
        
    Returns:
        Compressed registry bytes
    """
    # Use maximum compression level
    old_level = serializer.compression_level
    serializer.compression_level = 9
    
    # Serialize with max compression
    data = serializer.serialize_registry(registry)
    
    # Restore original compression level
    serializer.compression_level = old_level
    
    return data


def clone_registry(registry: SplatRegistry) -> SplatRegistry:
    """Create a deep copy of a registry.
    
    Args:
        registry: SplatRegistry to clone
        
    Returns:
        Cloned SplatRegistry
    """
    # Create new registry with same hierarchy
    hierarchy = Hierarchy(
        levels=registry.hierarchy.levels.copy(),
        init_splats_per_level=registry.hierarchy.init_splats_per_level.copy(),
        level_weights=registry.hierarchy.level_weights.copy()
    )
    
    new_registry = SplatRegistry(
        hierarchy=hierarchy,
        embedding_dim=registry.embedding_dim
    )
    
    # Copy stats
    new_registry.registered_count = registry.registered_count
    new_registry.unregistered_count = registry.unregistered_count
    new_registry.recovery_count = registry.recovery_count
    
    # Copy all splats without relationships
    splats_by_id = {}
    for splat in registry.get_all_splats():
        new_splat = splat.clone()
        new_registry.register(new_splat)
        splats_by_id[splat.id] = new_splat
    
    # Restore relationships
    for splat in registry.get_all_splats():
        new_splat = splats_by_id[splat.id]
        
        # Set parent
        if splat.parent and splat.parent.id in splats_by_id:
            new_splat.parent = splats_by_id[splat.parent.id]
        
        # Set children
        for child in splat.children:
            if child.id in splats_by_id:
                new_splat.children.add(splats_by_id[child.id])
    
    return new_registry


# Add extension methods to HSASerializer
def add_format_methods(HSASerializer):
    """Add format-related methods to HSASerializer class."""
    
    def save_to_file_method(self, registry, path, format="binary"):
        return save_to_file(self, registry, path, format)
    
    def load_from_file_method(self, path):
        return load_from_file(self, path)
    
    def compress_registry_method(self, registry):
        return compress_registry(self, registry)
    
    # Add methods to HSASerializer
    HSASerializer.save_to_file = save_to_file_method
    HSASerializer.load_from_file = load_from_file_method
    HSASerializer.compress_registry = compress_registry_method

# Add methods when module is imported
add_format_methods(HSASerializer)
