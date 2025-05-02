"""
Serialization core for Hierarchical Splat Attention (HSA).

This module provides the core functionality for serializing and deserializing HSA 
data structures, acting as the primary interface for persistence operations.
"""

import json
import numpy as np
import base64
import logging
from typing import Dict, List, Optional, Any, Union

from .splat import Splat
from .registry import SplatRegistry
from .hierarchy import Hierarchy

# Configure logging
logger = logging.getLogger(__name__)


class HSASerializer:
    """Base serializer class for HSA data structures."""
    
    # Version info
    VERSION = "1.0.0"
    
    def __init__(self, compression_level: int = 6):
        """Initialize serializer.
        
        Args:
            compression_level: Compression level (0-9, 0=none, 9=max)
        """
        self.compression_level = compression_level
    
    def serialize_registry(self, registry: SplatRegistry) -> bytes:
        """Serialize a registry to bytes.
        
        Args:
            registry: SplatRegistry to serialize
            
        Returns:
            Serialized registry as bytes
        """
        # Import here to avoid circular imports
        from .serialization_formats import compress_data
        
        # Convert registry to dictionary
        data = self._registry_to_dict(registry)
        
        # Add metadata
        from datetime import datetime
        data["_metadata"] = {
            "version": self.VERSION,
            "timestamp": datetime.now().isoformat(),
            "format": "hsa_registry"
        }
        
        # Convert to JSON
        json_data = json.dumps(data)
        
        # Compress if requested
        if self.compression_level > 0:
            return compress_data(json_data.encode('utf-8'), self.compression_level)
        else:
            return json_data.encode('utf-8')
    
    def deserialize_registry(self, data: bytes) -> SplatRegistry:
        """Deserialize a registry from bytes.
        
        Args:
            data: Serialized registry bytes
            
        Returns:
            Deserialized SplatRegistry
            
        Raises:
            ValueError: If data format is invalid
        """
        # Import here to avoid circular imports
        from .serialization_formats import decompress_data
        
        # Try to decompress
        try:
            json_data = decompress_data(data)
        except Exception as e:
            logger.warning(f"Decompression failed: {e}. Trying direct decoding.")
            # Not compressed, try direct decoding
            json_data = data.decode('utf-8')
        
        # Parse JSON
        data = json.loads(json_data)
        
        # Check metadata
        metadata = data.get("_metadata", {})
        format_type = metadata.get("format", "")
        
        if format_type != "hsa_registry":
            raise ValueError(f"Invalid format: {format_type}")
        
        version = metadata.get("version", "")
        if not self._check_version_compatibility(version):
            logger.warning(f"Version mismatch: {version} vs {self.VERSION}")
        
        # Convert to registry
        return self._dict_to_registry(data)
    
    def _check_version_compatibility(self, version: str) -> bool:
        """Check if a serialized version is compatible with this serializer.
        
        Args:
            version: Version string to check
            
        Returns:
            True if compatible, False otherwise
        """
        if version == self.VERSION:
            return True
            
        # Simple version compatibility (major.minor.patch)
        try:
            ver_parts = list(map(int, version.split('.')))
            current_parts = list(map(int, self.VERSION.split('.')))
            
            # Check major version
            if ver_parts[0] != current_parts[0]:
                return False
                
            # Minor version can be lower
            if ver_parts[1] > current_parts[1]:
                return False
                
            return True
        except:
            # If parsing fails, assume incompatible
            return False
    
    def _registry_to_dict(self, registry: SplatRegistry) -> Dict[str, Any]:
        """Convert a registry to a dictionary for serialization.
        
        Args:
            registry: SplatRegistry to convert
            
        Returns:
            Dictionary representation
        """
        data = {
            "embedding_dim": registry.embedding_dim,
            "hierarchy": self._hierarchy_to_dict(registry.hierarchy),
            "splats": [],
            "stats": {
                "registered_count": registry.registered_count,
                "unregistered_count": registry.unregistered_count,
                "recovery_count": registry.recovery_count
            }
        }
        
        # Process all splats
        for splat in registry.get_all_splats():
            splat_dict = {
                "id": splat.id,
                "level": splat.level,
                "position": self._serialize_array(splat.position),
                "covariance": self._serialize_array(splat.covariance),
                "amplitude": splat.amplitude,
                "lifetime": splat.lifetime,
                "info_contribution": splat.info_contribution,
                "activation_history": splat.activation_history.get_values(),
                "parent_id": splat.parent.id if splat.parent else None,
                "children_ids": [child.id for child in splat.children]
            }
            
            data["splats"].append(splat_dict)
        
        return data
    
    def _hierarchy_to_dict(self, hierarchy: Hierarchy) -> Dict[str, Any]:
        """Convert a hierarchy to a dictionary.
        
        Args:
            hierarchy: Hierarchy to convert
            
        Returns:
            Dictionary representation
        """
        return {
            "levels": hierarchy.levels,
            "init_splats_per_level": hierarchy.init_splats_per_level,
            "level_weights": hierarchy.level_weights
        }
    
    def _dict_to_registry(self, data: Dict[str, Any]) -> SplatRegistry:
        """Convert a dictionary to a registry.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Converted SplatRegistry
        """
        # Create hierarchy
        hierarchy = self._dict_to_hierarchy(data["hierarchy"])
        
        # Create registry
        registry = SplatRegistry(
            hierarchy=hierarchy,
            embedding_dim=data["embedding_dim"]
        )
        
        # Restore stats if available
        if "stats" in data:
            stats = data["stats"]
            registry.registered_count = stats.get("registered_count", 0)
            registry.unregistered_count = stats.get("unregistered_count", 0)
            registry.recovery_count = stats.get("recovery_count", 0)
        
        # First pass: Create all splats
        splats_by_id = {}
        for splat_data in data["splats"]:
            position = self._deserialize_array(splat_data["position"])
            covariance = self._deserialize_array(splat_data["covariance"])
            
            splat = Splat(
                dim=registry.embedding_dim,
                position=position,
                covariance=covariance,
                amplitude=splat_data["amplitude"],
                level=splat_data["level"],
                id=splat_data["id"]
            )
            
            # Restore additional properties
            if "lifetime" in splat_data:
                splat.lifetime = splat_data["lifetime"]
            
            if "info_contribution" in splat_data:
                splat.info_contribution = splat_data["info_contribution"]
            
            if "activation_history" in splat_data:
                for value in splat_data["activation_history"]:
                    splat.activation_history.add(value)
            
            # Store for relationship resolution
            splats_by_id[splat.id] = splat
            
            # Add to registry
            registry.register(splat)
        
        # Second pass: Resolve relationships
        for splat_data in data["splats"]:
            splat_id = splat_data["id"]
            
            if splat_id not in splats_by_id:
                continue
            
            splat = splats_by_id[splat_id]
            
            # Set parent
            parent_id = splat_data.get("parent_id")
            if parent_id and parent_id in splats_by_id:
                splat.parent = splats_by_id[parent_id]
            
            # Set children
            children_ids = splat_data.get("children_ids", [])
            for child_id in children_ids:
                if child_id in splats_by_id:
                    splat.children.add(splats_by_id[child_id])
        
        # Fix any relationship issues
        registry.repair_integrity()
        
        return registry
    
    def _dict_to_hierarchy(self, data: Dict[str, Any]) -> Hierarchy:
        """Convert a dictionary to a hierarchy.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Converted Hierarchy
        """
        return Hierarchy(
            levels=data["levels"],
            init_splats_per_level=data["init_splats_per_level"],
            level_weights=data["level_weights"]
        )
    
    def _serialize_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Serialize a numpy array to a dictionary.
        
        Args:
            arr: NumPy array to serialize
            
        Returns:
            Dictionary representation
        """
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "data": base64.b64encode(arr.tobytes()).decode('ascii')
        }
    
    def _deserialize_array(self, data: Dict[str, Any]) -> np.ndarray:
        """Deserialize a numpy array from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Deserialized NumPy array
        """
        shape = tuple(data["shape"])
        dtype = np.dtype(data["dtype"])
        buffer = base64.b64decode(data["data"])
        
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)
