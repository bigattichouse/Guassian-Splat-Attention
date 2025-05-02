"""
Delta-based serialization for Hierarchical Splat Attention (HSA).

This module provides functionality for computing and applying deltas between
registry versions, enabling efficient updates and migrations.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from .splat import Splat
from .registry import SplatRegistry
from .hierarchy import Hierarchy
from .serialization_core import HSASerializer
from .serialization_formats import decompress_data, compress_data, clone_registry

# Configure logging
logger = logging.getLogger(__name__)


def compute_registry_delta(
    serializer: HSASerializer,
    old_registry: SplatRegistry,
    new_registry: SplatRegistry
) -> Dict[str, Any]:
    """Compute delta between two registries.
    
    Args:
        serializer: HSASerializer instance
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
        delta["hierarchy_changes"] = serializer._hierarchy_to_dict(new_registry.hierarchy)
    
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
            delta["added_splats"].append(splat_to_dict(serializer, new_splat))
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
                delta["modified_splats"].append(splat_to_dict(serializer, new_splat))
    
    return delta


def splat_to_dict(serializer: HSASerializer, splat: Splat) -> Dict[str, Any]:
    """Convert a splat to a dictionary.
    
    Args:
        serializer: HSASerializer instance
        splat: Splat to convert
        
    Returns:
        Dictionary representation
    """
    return {
        "id": splat.id,
        "level": splat.level,
        "position": serializer._serialize_array(splat.position),
        "covariance": serializer._serialize_array(splat.covariance),
        "amplitude": splat.amplitude,
        "lifetime": splat.lifetime,
        "info_contribution": splat.info_contribution,
        "activation_history": splat.activation_history.get_values(),
        "parent_id": splat.parent.id if splat.parent else None,
        "children_ids": [child.id for child in splat.children]
    }


def apply_delta(
    serializer: HSASerializer,
    base_registry: SplatRegistry,
    delta: Dict[str, Any]
) -> SplatRegistry:
    """Apply a delta to a base registry.
    
    Args:
        serializer: HSASerializer instance
        base_registry: Base SplatRegistry
        delta: Delta dictionary
        
    Returns:
        Updated SplatRegistry
    """
    # Import numpy here to avoid circular imports
    import numpy as np
    
    # Create a copy of the base registry
    registry = clone_registry(base_registry)
    
    # Update hierarchy if changed
    if delta.get("hierarchy_changes"):
        registry.hierarchy = serializer._dict_to_hierarchy(delta["hierarchy_changes"])
    
    # Remove splats
    for splat_id in delta.get("removed_splats", []):
        try:
            registry.unregister(splat_id)
        except ValueError:
            logger.warning(f"Failed to remove splat {splat_id}: not found")
    
    # Add new splats
    splats_by_id = {splat.id: splat for splat in registry.get_all_splats()}
    
    for splat_data in delta.get("added_splats", []):
        position = serializer._deserialize_array(splat_data["position"])
        covariance = serializer._deserialize_array(splat_data["covariance"])
        
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
            position = serializer._deserialize_array(splat_data["position"])
            covariance = serializer._deserialize_array(splat_data["covariance"])
            
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


def save_registry_delta(
    serializer: HSASerializer,
    old_registry: SplatRegistry,
    new_registry: SplatRegistry
) -> bytes:
    """Save the delta between two registries.
    
    Args:
        serializer: HSASerializer instance
        old_registry: Old SplatRegistry state
        new_registry: New SplatRegistry state
        
    Returns:
        Serialized delta as bytes
    """
    # Compute delta
    delta = compute_registry_delta(serializer, old_registry, new_registry)
    
    # Add metadata
    from datetime import datetime
    delta["_metadata"] = {
        "version": serializer.VERSION,
        "timestamp": datetime.now().isoformat(),
        "format": "hsa_registry_delta",
        "base_registry_id": get_registry_id(old_registry),
        "target_registry_id": get_registry_id(new_registry)
    }
    
    # Convert to JSON
    json_data = json.dumps(delta)
    
    # Compress if requested
    if serializer.compression_level > 0:
        return compress_data(json_data.encode('utf-8'), serializer.compression_level)
    else:
        return json_data.encode('utf-8')


def apply_registry_delta(
    serializer: HSASerializer,
    base_registry: SplatRegistry,
    delta: bytes
) -> SplatRegistry:
    """Apply a delta to a base registry.
    
    Args:
        serializer: HSASerializer instance
        base_registry: Base SplatRegistry
        delta: Serialized delta bytes
        
    Returns:
        Updated SplatRegistry
        
    Raises:
        ValueError: If delta format is invalid or base registry doesn't match
    """
    # Try to decompress
    try:
        json_data = decompress_data(delta)
    except Exception as e:
        logger.warning(f"Decompression failed: {e}. Trying direct decoding.")
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
    current_id = get_registry_id(base_registry)
    
    if base_id != current_id:
        logger.warning(
            f"Base registry mismatch: {base_id} vs {current_id}. " +
            "Delta may not apply correctly."
        )
    
    # Apply delta
    return apply_delta(serializer, base_registry, delta_data)


def get_registry_id(registry: SplatRegistry) -> str:
    """Generate a unique ID for a registry based on its content.
    
    Args:
        registry: SplatRegistry to identify
        
    Returns:
        Unique ID string
    """
    # Import numpy here to avoid circular imports
    import numpy as np
    
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

def migrate_to_version(
    serializer: HSASerializer,
    registry: SplatRegistry,
    target_version: str
) -> SplatRegistry:
    """Migrate a registry to a specific version.
    
    Args:
        serializer: HSASerializer instance
        registry: SplatRegistry to migrate
        target_version: Target version string
        
    Returns:
        Migrated SplatRegistry
        
    Raises:
        ValueError: If migration is not possible
    """
    # Simple version checking
    current_v = list(map(int, serializer.VERSION.split('.')))
    target_v = list(map(int, target_version.split('.')))
    
    # Check if migration is needed
    if current_v == target_v:
        return registry
    
    # Check if migration is possible
    if current_v[0] != target_v[0]:
        raise ValueError(
            f"Cannot migrate between major versions: {serializer.VERSION} to {target_version}"
        )
    
    if current_v[1] > target_v[1]:
        raise ValueError(
            f"Cannot downgrade from {serializer.VERSION} to {target_version}"
        )
    
    # For now, just return the registry as-is
    # In a real implementation, perform necessary migrations
    logger.warning(
        f"Migration from {serializer.VERSION} to {target_version} not implemented. " +
        "Returning registry as-is."
    )
    
    return registry


# Add delta methods to HSASerializer
def add_delta_methods(HSASerializer):
    """Add delta-related methods to HSASerializer class."""
    
    def save_registry_delta_method(self, old_registry, new_registry):
        return save_registry_delta(self, old_registry, new_registry)
    
    def apply_registry_delta_method(self, base_registry, delta):
        return apply_registry_delta(self, base_registry, delta)
    
    def migrate_to_version_method(self, registry, target_version):
        return migrate_to_version(self, registry, target_version)
    
    # Add methods to HSASerializer
    HSASerializer.save_registry_delta = save_registry_delta_method
    HSASerializer.apply_registry_delta = apply_registry_delta_method
    HSASerializer.migrate_to_version = migrate_to_version_method

# Add methods when module is imported
add_delta_methods(HSASerializer)
