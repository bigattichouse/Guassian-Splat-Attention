"""
Serialization core for Hierarchical Splat Attention (HSA).

This module provides functionality for serializing and deserializing HSA data
structures, enabling persistence and loading of trained models.
"""

import json
import numpy as np
import base64
import zlib
import os
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union, BinaryIO
import pickle
from datetime import datetime

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
        # Convert registry to dictionary
        data = self._registry_to_dict(registry)
        
        # Add metadata
        data["_metadata"] = {
            "version": self.VERSION,
            "timestamp": datetime.now().isoformat(),
            "format": "hsa_registry"
        }
        
        # Convert to JSON
        json_data = json.dumps(data)
        
        # Compress if requested
        if self.compression_level > 0:
            return zlib.compress(json_data.encode('utf-8'), self.compression_level)
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
        # Try to decompress (will fail if not compressed)
        try:
            json_data = zlib.decompress(data).decode('utf-8')
        except zlib.error:
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
    
    def save_to_file(
        self,
        registry: SplatRegistry,
        path: str,
        format: str = "binary"
    ) -> None:
        """Save a registry to a file.
        
        Args:
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
            data = self.serialize_registry(registry)
            
            # Write to file
            with open(path, 'wb') as f:
                f.write(data)
        elif format == "json":
            # Convert to dictionary
            data = self._registry_to_dict(registry)
            
            # Add metadata
            data["_metadata"] = {
                "version": self.VERSION,
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
    
    def load_from_file(self, path: str) -> SplatRegistry:
        """Load a registry from a file.
        
        Args:
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
            return self._dict_to_registry(data)
        elif ext == '.pickle' or ext == '.pkl':
            # Read pickle file
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            # Assume binary format
            with open(path, 'rb') as f:
                data = f.read()
            
            return self.deserialize_registry(data)
    
    def save_registry_delta(
        self,
        old_registry: SplatRegistry,
        new_registry: SplatRegistry
    ) -> bytes:
        """Save the delta between two registries.
        
        Args:
            old_registry: Old SplatRegistry state
            new_registry: New SplatRegistry state
            
        Returns:
            Serialized delta as bytes
        """
        # Compute delta
        delta = self._compute_registry_delta(old_registry, new_registry)
        
        # Add metadata
        delta["_metadata"] = {
            "version": self.VERSION,
            "timestamp": datetime.now().isoformat(),
            "format": "hsa_registry_delta",
            "base_registry_id": self._get_registry_id(old_registry),
            "target_registry_id": self._get_registry_id(new_registry)
        }
        
        # Convert to JSON
        json_data = json.dumps(delta)
        
        # Compress if requested
        if self.compression_level > 0:
            return zlib.compress(json_data.encode('utf-8'), self.compression_level)
        else:
            return json_data.encode('utf-8')
    
    def apply_registry_delta(
        self,
        base_registry: SplatRegistry,
        delta: bytes
    ) -> SplatRegistry:
        """Apply a delta to a base registry.
        
        Args:
            base_registry: Base SplatRegistry
            delta: Serialized delta bytes
            
        Returns:
            Updated SplatRegistry
            
        Raises:
            ValueError: If delta format is invalid or base registry doesn't match
        """
        # Try to decompress (will fail if not compressed)
        try:
            json_data = zlib.decompress(delta).decode('utf-8')
        except zlib.error:
            # Not compressed, try direct decoding
            json_data = delta.decode('utf-8')
        
        # Parse JSON
        delta_data = json.loads(json_data)
        
        # Check metadata
        metadata = delta_data.get("_metadata", {})
        format_type = metadata.get("format", "")
        
        if format_type != "hsa_registry_delta":
            raise ValueError(f"Invalid format: {format_type}")
        
        # Check base registry ID
        base_id = metadata.get("base_registry_id", "")
        current_id = self._get_registry_id(base_registry)
        
        if base_id != current_id:
            logger.warning(
                f"Base registry mismatch: {base_id} vs {current_id}. " +
                "Delta may not apply correctly."
            )
        
        # Apply delta
        return self._apply_delta(base_registry, delta_data)
    
    def _get_registry_id(self, registry: SplatRegistry) -> str:
        """Generate a unique ID for a registry based on its content.
        
        Args:
            registry: SplatRegistry to identify
            
        Returns:
            Unique ID string
        """
        # Get all splats
        splats = registry.get_all_splats()
        
        # Generate hash based on splat IDs and parameters
        splat_data = []
        for splat in splats:
            splat_data.append((
                splat.id,
                splat.level,
                hash(tuple(splat.position.flatten().tolist())),
                hash(tuple(splat.covariance.flatten().tolist())),
                splat.amplitude
            ))
        
        # Sort for consistency
        splat_data.sort()
        
        # Convert to string and hash
        return str(hash(str(splat_data)))
    
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
    
    def _compute_registry_delta(
        self,
        old_registry: SplatRegistry,
        new_registry: SplatRegistry
    ) -> Dict[str, Any]:
        """Compute delta between two registries.
        
        Args:
            old_registry: Old SplatRegistry state
            new_registry: New SplatRegistry state
            
        Returns:
            Delta dictionary
        """
        delta = {
            "embedding_dim": new_registry.embedding_dim,
            "added_splats": [],
            "removed_splats": [],
            "modified_splats": [],
            "hierarchy_changes": None
        }
        
        # Check if hierarchy changed
        if (old_registry.hierarchy.levels != new_registry.hierarchy.levels or
            old_registry.hierarchy.init_splats_per_level != new_registry.hierarchy.init_splats_per_level or
            old_registry.hierarchy.level_weights != new_registry.hierarchy.level_weights):
            delta["hierarchy_changes"] = self._hierarchy_to_dict(new_registry.hierarchy)
        
        # Get all splats from both registries
        old_splats = {splat.id: splat for splat in old_registry.get_all_splats()}
        new_splats = {splat.id: splat for splat in new_registry.get_all_splats()}
        
        # Find removed splats
        for splat_id in old_splats:
            if splat_id not in new_splats:
                delta["removed_splats"].append(splat_id)
        
        # Find added and modified splats
        for splat_id, new_splat in new_splats.items():
            if splat_id not in old_splats:
                # Added splat
                delta["added_splats"].append(self._splat_to_dict(new_splat))
            else:
                # Check if modified
                old_splat = old_splats[splat_id]
                
                # Check for modifications
                if (not np.array_equal(old_splat.position, new_splat.position) or
                    not np.array_equal(old_splat.covariance, new_splat.covariance) or
                    old_splat.amplitude != new_splat.amplitude or
                    old_splat.level != new_splat.level or
                    (old_splat.parent.id if old_splat.parent else None) != 
                    (new_splat.parent.id if new_splat.parent else None) or
                    set(c.id for c in old_splat.children) != 
                    set(c.id for c in new_splat.children)):
                    
                    # Modified splat
                    delta["modified_splats"].append(self._splat_to_dict(new_splat))
        
        return delta
    
    def _splat_to_dict(self, splat: Splat) -> Dict[str, Any]:
        """Convert a splat to a dictionary.
        
        Args:
            splat: Splat to convert
            
        Returns:
            Dictionary representation
        """
        return {
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
    
    def _apply_delta(
        self,
        base_registry: SplatRegistry,
        delta: Dict[str, Any]
    ) -> SplatRegistry:
        """Apply a delta to a base registry.
        
        Args:
            base_registry: Base SplatRegistry
            delta: Delta dictionary
            
        Returns:
            Updated SplatRegistry
        """
        # Create a copy of the base registry
        registry = self._clone_registry(base_registry)
        
        # Update hierarchy if changed
        if delta.get("hierarchy_changes"):
            registry.hierarchy = self._dict_to_hierarchy(delta["hierarchy_changes"])
        
        # Remove splats
        for splat_id in delta.get("removed_splats", []):
            try:
                registry.unregister(splat_id)
            except ValueError:
                logger.warning(f"Failed to remove splat {splat_id}: not found")
        
        # Add new splats
        splats_by_id = {splat.id: splat for splat in registry.get_all_splats()}
        
        for splat_data in delta.get("added_splats", []):
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
            
            # Add to registry and store for relationship resolution
            registry.register(splat)
            splats_by_id[splat.id] = splat
        
        # Modify existing splats
        for splat_data in delta.get("modified_splats", []):
            splat_id = splat_data["id"]
            
            try:
                splat = registry.get_splat(splat_id)
                
                # Update properties
                position = self._deserialize_array(splat_data["position"])
                covariance = self._deserialize_array(splat_data["covariance"])
                
                splat.update_parameters(
                    position=position,
                    covariance=covariance,
                    amplitude=splat_data["amplitude"]
                )
                
                # Update level if changed
                if splat.level != splat_data["level"]:
                    registry.change_splat_level(splat, splat_data["level"])
                
                # Update other properties
                if "lifetime" in splat_data:
                    splat.lifetime = splat_data["lifetime"]
                
                if "info_contribution" in splat_data:
                    splat.info_contribution = splat_data["info_contribution"]
                
                if "activation_history" in splat_data:
                    # Clear current history
                    splat.activation_history = type(splat.activation_history)(10)
                    
                    # Add new values
                    for value in splat_data["activation_history"]:
                        splat.activation_history.add(value)
                
                # Store for relationship resolution
                splats_by_id[splat_id] = splat
                
            except ValueError:
                logger.warning(f"Failed to modify splat {splat_id}: not found")
        
        # Update relationships for added and modified splats
        for splat_data in delta.get("added_splats", []) + delta.get("modified_splats", []):
            splat_id = splat_data["id"]
            
            if splat_id not in splats_by_id:
                continue
            
            splat = splats_by_id[splat_id]
            
            # Update parent
            parent_id = splat_data.get("parent_id")
            
            if parent_id:
                if parent_id in splats_by_id:
                    splat.parent = splats_by_id[parent_id]
                    splats_by_id[parent_id].children.add(splat)
                else:
                    logger.warning(f"Parent {parent_id} not found for splat {splat_id}")
            else:
                # No parent
                if splat.parent:
                    splat.parent.children.remove(splat)
                    splat.parent = None
            
            # Update children
            current_children = set(splat.children)
            new_children_ids = set(splat_data.get("children_ids", []))
            
            # Remove old children
            for child in list(current_children):
                if child.id not in new_children_ids:
                    splat.children.remove(child)
                    if child.parent == splat:
                        child.parent = None
            
            # Add new children
            for child_id in new_children_ids:
                if child_id in splats_by_id and splats_by_id[child_id] not in splat.children:
                    child = splats_by_id[child_id]
                    splat.children.add(child)
                    child.parent = splat
        
        # Fix any relationship issues
        registry.repair_integrity()
        
        return registry
    
    def _clone_registry(self, registry: SplatRegistry) -> SplatRegistry:
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
    
    def compress_registry(self, registry: SplatRegistry) -> bytes:
        """Compress a registry to a minimal representation.
        
        Args:
            registry: SplatRegistry to compress
            
        Returns:
            Compressed registry bytes
        """
        # Use maximum compression level
        old_level = self.compression_level
        self.compression_level = 9
        
        # Serialize with max compression
        data = self.serialize_registry(registry)
        
        # Restore original compression level
        self.compression_level = old_level
        
        return data
    
    def migrate_to_version(
        self,
        registry: SplatRegistry,
        target_version: str
    ) -> SplatRegistry:
        """Migrate a registry to a specific version.
        
        Args:
            registry: SplatRegistry to migrate
            target_version: Target version string
            
        Returns:
            Migrated SplatRegistry
            
        Raises:
            ValueError: If migration is not possible
        """
        # Simple version checking
        current_v = list(map(int, self.VERSION.split('.')))
        target_v = list(map(int, target_version.split('.')))
        
        # Check if migration is needed
        if current_v == target_v:
            return registry
        
        # Check if migration is possible
        if current_v[0] != target_v[0]:
            raise ValueError(
                f"Cannot migrate between major versions: {self.VERSION} to {target_version}"
            )
        
        if current_v[1] > target_v[1]:
            raise ValueError(
                f"Cannot downgrade from {self.VERSION} to {target_version}"
            )
        
        # For now, just return the registry as-is
        # In a real implementation, perform necessary migrations
        logger.warning(
            f"Migration from {self.VERSION} to {target_version} not implemented. " +
            "Returning registry as-is."
        )
        
        return registry
