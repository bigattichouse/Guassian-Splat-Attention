"""
Registry operations for Hierarchical Splat Attention (HSA).

This module provides extended operations for the SplatRegistry class,
including batch operations, filtering, selection strategies, and special
management functions.
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Any, Iterator, Union
import numpy as np
import random
import logging

from .splat import Splat
from .registry import SplatRegistry
from .hierarchy import Hierarchy

# Configure logging
logger = logging.getLogger(__name__)


class RegistryOperations:
    """
    Extended operations for SplatRegistry to perform batch operations,
    filtering, selection, and specialized management functions.
    """
    
    @staticmethod
    def batch_register(registry: SplatRegistry, splats: List[Splat]) -> int:
        """Register multiple splats in the registry.
        
        Args:
            registry: SplatRegistry to operate on
            splats: List of splats to register
            
        Returns:
            Number of splats successfully registered
        """
        success_count = 0
        for splat in splats:
            try:
                registry.register(splat)
                success_count += 1
            except ValueError as e:
                logger.warning(f"Failed to register splat {splat.id}: {e}")
                
        return success_count
    
    @staticmethod
    def batch_unregister(registry: SplatRegistry, splats: List[Union[Splat, str]]) -> int:
        """Unregister multiple splats from the registry.
        
        Args:
            registry: SplatRegistry to operate on
            splats: List of splats or splat IDs to unregister
            
        Returns:
            Number of splats successfully unregistered
        """
        success_count = 0
        for splat in splats:
            try:
                registry.unregister(splat)
                success_count += 1
            except ValueError as e:
                splat_id = splat if isinstance(splat, str) else splat.id
                logger.warning(f"Failed to unregister splat {splat_id}: {e}")
                
        return success_count
    
    @staticmethod
    def filter_splats_by_level(registry: SplatRegistry, level: str) -> List[Splat]:
        """Get all splats at a specific level.
        
        This is a convenience method that wraps registry.get_splats_at_level().
        
        Args:
            registry: SplatRegistry to query
            level: Hierarchical level to filter by
            
        Returns:
            List of splats at the specified level
            
        Raises:
            ValueError: If level is not valid
        """
        return list(registry.get_splats_at_level(level))
    
    @staticmethod
    def filter_splats_by_activation(
        registry: SplatRegistry, 
        min_activation: float = 0.0, 
        max_activation: float = 1.0
    ) -> List[Splat]:
        """Filter splats by their activation levels.
        
        Args:
            registry: SplatRegistry to query
            min_activation: Minimum activation threshold (inclusive)
            max_activation: Maximum activation threshold (inclusive)
            
        Returns:
            List of splats with activation in the specified range
        """
        return [
            splat for splat in registry.get_all_splats()
            if min_activation <= splat.get_average_activation() <= max_activation
        ]
    
    @staticmethod
    def filter_splats_by_lifetime(
        registry: SplatRegistry, 
        min_lifetime: int = 0, 
        max_lifetime: Optional[int] = None
    ) -> List[Splat]:
        """Filter splats by their lifetime.
        
        Args:
            registry: SplatRegistry to query
            min_lifetime: Minimum lifetime threshold (inclusive)
            max_lifetime: Maximum lifetime threshold (inclusive, None for no upper limit)
            
        Returns:
            List of splats with lifetime in the specified range
        """
        if max_lifetime is None:
            return [
                splat for splat in registry.get_all_splats()
                if splat.lifetime >= min_lifetime
            ]
        
        return [
            splat for splat in registry.get_all_splats()
            if min_lifetime <= splat.lifetime <= max_lifetime
        ]
    
    @staticmethod
    def filter_splats_by_position(
        registry: SplatRegistry,
        center: np.ndarray,
        radius: float
    ) -> List[Splat]:
        """Filter splats by their proximity to a position in embedding space.
        
        Args:
            registry: SplatRegistry to query
            center: Center position in embedding space
            radius: Maximum distance from center
            
        Returns:
            List of splats within the specified radius
        """
        return [
            splat for splat in registry.get_all_splats()
            if np.linalg.norm(splat.position - center) <= radius
        ]
    
    @staticmethod
    def filter_splats_by_custom(
        registry: SplatRegistry,
        predicate: Callable[[Splat], bool]
    ) -> List[Splat]:
        """Filter splats using a custom predicate function.
        
        Args:
            registry: SplatRegistry to query
            predicate: Function that takes a splat and returns a boolean
            
        Returns:
            List of splats for which the predicate returns True
        """
        return [
            splat for splat in registry.get_all_splats()
            if predicate(splat)
        ]
    
    @staticmethod
    def select_random_splats(
        registry: SplatRegistry, 
        count: int, 
        level: Optional[str] = None
    ) -> List[Splat]:
        """Select random splats from the registry.
        
        Args:
            registry: SplatRegistry to select from
            count: Number of splats to select
            level: Optional level to restrict selection to
            
        Returns:
            List of randomly selected splats (may be fewer than count if not enough available)
        """
        if level is not None:
            candidates = list(registry.get_splats_at_level(level))
        else:
            candidates = registry.get_all_splats()
        
        if not candidates:
            return []
        
        # Ensure we don't try to select more than available
        count = min(count, len(candidates))
        
        return random.sample(candidates, count)
    
    @staticmethod
    def select_top_splats_by_activation(
        registry: SplatRegistry, 
        count: int, 
        level: Optional[str] = None
    ) -> List[Splat]:
        """Select splats with highest activation from the registry.
        
        Args:
            registry: SplatRegistry to select from
            count: Number of splats to select
            level: Optional level to restrict selection to
            
        Returns:
            List of splats with highest activation
        """
        if level is not None:
            candidates = list(registry.get_splats_at_level(level))
        else:
            candidates = registry.get_all_splats()
        
        if not candidates:
            return []
        
        # Sort by activation (highest first)
        sorted_splats = sorted(
            candidates,
            key=lambda s: s.get_average_activation(),
            reverse=True
        )
        
        # Ensure we don't try to select more than available
        count = min(count, len(sorted_splats))
        
        return sorted_splats[:count]
    
    @staticmethod
    def select_splats_near_position(
        registry: SplatRegistry,
        position: np.ndarray,
        count: int,
        level: Optional[str] = None
    ) -> List[Splat]:
        """Select splats closest to a position in embedding space.
        
        Args:
            registry: SplatRegistry to select from
            position: Position in embedding space
            count: Number of splats to select
            level: Optional level to restrict selection to
            
        Returns:
            List of splats closest to the specified position
        """
        if level is not None:
            candidates = list(registry.get_splats_at_level(level))
        else:
            candidates = registry.get_all_splats()
        
        if not candidates:
            return []
        
        # Sort by distance to position (closest first)
        sorted_splats = sorted(
            candidates,
            key=lambda s: np.linalg.norm(s.position - position)
        )
        
        # Ensure we don't try to select more than available
        count = min(count, len(sorted_splats))
        
        return sorted_splats[:count]
    
    @staticmethod
    def reorganize_hierarchy(
        registry: SplatRegistry,
        new_hierarchy: Hierarchy
    ) -> bool:
        """Reorganize the registry to use a new hierarchy structure.
        
        This is a complex operation that tries to map splats from the old
        hierarchy to the new one. It may require removing or reassigning splats.
        
        Args:
            registry: SplatRegistry to reorganize
            new_hierarchy: New hierarchy structure
            
        Returns:
            True if successful, False if failed
        """
        old_hierarchy = registry.hierarchy
        
        # Create mapping from old levels to new levels
        level_mapping = {}
        
        # First, try to map levels with the same name
        for old_level in old_hierarchy.levels:
            if old_level in new_hierarchy.levels:
                level_mapping[old_level] = old_level
        
        # Then, try to map levels by position (lowest to highest)
        for i, old_level in enumerate(old_hierarchy.levels):
            if old_level not in level_mapping:
                if i < len(new_hierarchy.levels):
                    level_mapping[old_level] = new_hierarchy.levels[i]
                else:
                    # Map to the highest level in the new hierarchy
                    level_mapping[old_level] = new_hierarchy.levels[-1]
        
        logger.info(f"Level mapping: {level_mapping}")
        
        # Store the current splats to avoid modification during iteration
        current_splats = list(registry.get_all_splats())
        
        # Update registry hierarchy
        registry.hierarchy = new_hierarchy
        
        # Ensure all new levels exist in splats_by_level
        for level in new_hierarchy.levels:
            if level not in registry.splats_by_level:
                registry.splats_by_level[level] = set()
        
        # Track which splats need to be moved
        splat_moves = []
        for splat in current_splats:
            old_level = splat.level
            new_level = level_mapping.get(old_level)
            
            if new_level is None:
                logger.warning(f"No mapping found for level {old_level}, removing splat {splat.id}")
                try:
                    registry.unregister(splat)
                except ValueError:
                    logger.error(f"Failed to unregister splat {splat.id}")
                continue
            
            # If level has changed, mark for update
            if old_level != new_level:
                splat_moves.append((splat, old_level, new_level))
        
        # Execute level changes after hierarchy is updated and new levels exist
        for splat, old_level, new_level in splat_moves:
            try:
                # Remove from old level set
                if splat in registry.splats_by_level.get(old_level, set()):
                    registry.splats_by_level[old_level].remove(splat)
                
                # Update level attribute
                splat.level = new_level
                
                # Add to new level set
                registry.splats_by_level[new_level].add(splat)
                
                logger.info(f"Moved splat {splat.id} from {old_level} to {new_level}")
            except Exception as e:
                logger.error(f"Failed to update level for splat {splat.id}: {e}")
                # Continue anyway to process other splats
        
        # Clean up old levels that are no longer in the hierarchy
        for level in list(registry.splats_by_level.keys()):
            if level not in new_hierarchy.levels:
                del registry.splats_by_level[level]
        
        # Verify and fix any relationship issues
        registry.repair_integrity()
        
        # Verify again after repair
        return registry.verify_integrity()
    
    @staticmethod
    def redistribution_strategy_evenly(
        registry: SplatRegistry,
        splats: List[Splat],
        target_levels: List[str],
        target_counts: List[int]
    ) -> Dict[str, List[Splat]]:
        """Redistribute splats evenly across target levels.
        
        Args:
            registry: SplatRegistry for context
            splats: List of splats to redistribute
            target_levels: List of levels to distribute to
            target_counts: List of target counts for each level
            
        Returns:
            Dictionary mapping level names to lists of splats
        """
        if len(target_levels) != len(target_counts):
            raise ValueError("target_levels and target_counts must have the same length")
        
        # Validate target levels
        for level in target_levels:
            if not registry.hierarchy.is_valid_level(level):
                raise ValueError(f"Level '{level}' is not valid in current hierarchy")
        
        # Check if we have enough splats
        total_target = sum(target_counts)
        if len(splats) < total_target:
            logger.warning(
                f"Not enough splats ({len(splats)}) to meet target counts ({total_target})"
            )
        
        # Initialize result dictionary
        result = {level: [] for level in target_levels}
        
        # Sort target levels by hierarchy order (low to high)
        sorted_indices = sorted(
            range(len(target_levels)),
            key=lambda i: registry.hierarchy.get_level_index(target_levels[i])
        )
        
        sorted_levels = [target_levels[i] for i in sorted_indices]
        sorted_counts = [target_counts[i] for i in sorted_indices]
        
        # Shuffle splats for random distribution
        shuffled_splats = splats.copy()
        random.shuffle(shuffled_splats)
        
        # Distribute splats
        start_idx = 0
        for level, count in zip(sorted_levels, sorted_counts):
            end_idx = min(start_idx + count, len(shuffled_splats))
            result[level].extend(shuffled_splats[start_idx:end_idx])
            start_idx = end_idx
        
        return result
    
    @staticmethod
    def redistribute_splats(
        registry: SplatRegistry,
        splats: List[Splat],
        target_levels: List[str],
        target_counts: List[int]
    ) -> bool:
        """Redistribute splats across target levels and update registry.
        
        Args:
            registry: SplatRegistry to update
            splats: List of splats to redistribute
            target_levels: List of levels to distribute to
            target_counts: List of target counts for each level
            
        Returns:
            True if successful, False if failed
        """
        try:
            # Get distribution plan
            distribution = RegistryOperations.redistribution_strategy_evenly(
                registry, splats, target_levels, target_counts
            )
            
            # Execute the plan
            for level, level_splats in distribution.items():
                for splat in level_splats:
                    old_level = splat.level
                    
                    # Skip if already at target level
                    if old_level == level:
                        continue
                    
                    # Change level in registry
                    try:
                        registry.change_splat_level(splat, level)
                    except ValueError as e:
                        logger.error(f"Failed to change level for splat {splat.id}: {e}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to redistribute splats: {e}")
            return False
    
    @staticmethod
    def redistribute_by_position_clustering(
        registry: SplatRegistry,
        target_levels: List[str],
        target_counts: List[int],
        tokens: Optional[np.ndarray] = None
    ) -> bool:
        """Redistribute splats across levels using position-based clustering.
        
        Args:
            registry: SplatRegistry to update
            target_levels: List of levels to distribute to
            target_counts: List of target counts for each level
            tokens: Optional token embeddings to influence clustering
            
        Returns:
            True if successful, False if failed
        """
        try:
            # Get all splats
            all_splats = registry.get_all_splats()
            
            # Extract positions
            positions = np.array([splat.position for splat in all_splats])
            
            # Perform simple clustering based on distances
            # This is a basic implementation - for production, consider using 
            # k-means or other clustering algorithms
            
            # Compute mean position (optional: use tokens to influence this)
            if tokens is not None and tokens.shape[0] > 0:
                # Use token positions to influence the mean
                token_mean = np.mean(tokens, axis=0)
                mean_position = (np.mean(positions, axis=0) + token_mean) / 2
            else:
                # Use only splat positions
                mean_position = np.mean(positions, axis=0)
            
            # Compute distance from mean
            distances = [np.linalg.norm(splat.position - mean_position) for splat in all_splats]
            
            # Sort splats by distance (closest first)
            sorted_splats = [x for _, x in sorted(zip(distances, all_splats))]
            
            # Execute redistribution
            return RegistryOperations.redistribute_splats(
                registry, sorted_splats, target_levels, target_counts
            )
            
        except Exception as e:
            logger.error(f"Failed to redistribute by clustering: {e}")
            return False
    
    @staticmethod
    def balance_levels(registry: SplatRegistry) -> bool:
        """Balance the number of splats across levels according to hierarchy settings.
        
        Args:
            registry: SplatRegistry to balance
            
        Returns:
            True if successful, False if failed
        """
        try:
            hierarchy = registry.hierarchy
            
            # Get target counts from hierarchy
            target_levels = hierarchy.levels
            target_counts = hierarchy.init_splats_per_level
            
            # Get current counts
            current_counts = {
                level: registry.count_splats(level)
                for level in target_levels
            }
            
            # Check if rebalancing is needed
            imbalance = False
            for level, target in zip(target_levels, target_counts):
                current = current_counts[level]
                if abs(current - target) > max(2, target * 0.1):  # 10% tolerance or at least 2
                    imbalance = True
                    break
            
            if not imbalance:
                return True  # Already balanced
            
            # Get all splats for redistribution
            all_splats = registry.get_all_splats()
            
            # Use position-based clustering for intelligent redistribution
            return RegistryOperations.redistribute_by_position_clustering(
                registry, target_levels, target_counts
            )
            
        except Exception as e:
            logger.error(f"Failed to balance levels: {e}")
            return False
    
    @staticmethod
    def merge_registries(
        registry_a: SplatRegistry,
        registry_b: SplatRegistry,
        strategy: str = "replace"
    ) -> SplatRegistry:
        """Merge two registries into a new one.
        
        Args:
            registry_a: First SplatRegistry
            registry_b: Second SplatRegistry
            strategy: Merge strategy ('replace', 'keep_a', or 'keep_b' for ID conflicts)
            
        Returns:
            New SplatRegistry with merged splats
            
        Raises:
            ValueError: If registries are incompatible or strategy is invalid
        """
        # Validate strategy
        valid_strategies = ["replace", "keep_a", "keep_b"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid merge strategy '{strategy}'. Must be one of {valid_strategies}")
        
        # Check compatibility
        if registry_a.embedding_dim != registry_b.embedding_dim:
            raise ValueError(
                f"Registry embedding dimensions don't match: " +
                f"{registry_a.embedding_dim} vs {registry_b.embedding_dim}"
            )
        
        # Check hierarchies for compatibility (not restricting to exact match)
        hierarchy_a = registry_a.hierarchy
        hierarchy_b = registry_b.hierarchy
        
        # Combine hierarchies (use A as base but keep all levels from both)
        combined_levels = list(hierarchy_a.levels)
        for level in hierarchy_b.levels:
            if level not in combined_levels:
                combined_levels.append(level)
        
        # Initialize splats per level (use maximum of both)
        combined_splats_per_level = []
        for level in combined_levels:
            count_a = hierarchy_a.get_num_init_splats(level) if level in hierarchy_a.levels else 0
            count_b = hierarchy_b.get_num_init_splats(level) if level in hierarchy_b.levels else 0
            combined_splats_per_level.append(max(count_a, count_b))
        
        # Initialize weights (use average of both where available)
        combined_weights = []
        for level in combined_levels:
            weight_a = hierarchy_a.get_level_weight(level) if level in hierarchy_a.levels else 0.0
            weight_b = hierarchy_b.get_level_weight(level) if level in hierarchy_b.levels else 0.0
            
            if level in hierarchy_a.levels and level in hierarchy_b.levels:
                # Average weights if both have the level
                combined_weights.append((weight_a + weight_b) / 2)
            else:
                # Use the weight from whichever hierarchy has this level
                combined_weights.append(weight_a or weight_b)
        
        # Create combined hierarchy
        combined_hierarchy = Hierarchy(
            levels=combined_levels,
            init_splats_per_level=combined_splats_per_level,
            level_weights=combined_weights
        )
        
        # Create new registry
        new_registry = SplatRegistry(
            hierarchy=combined_hierarchy,
            embedding_dim=registry_a.embedding_dim
        )
        
        # Helper to clone splat with proper parent reference
        def clone_with_parent_update(splat, id_mapping):
            clone = splat.clone()
            # Update parent reference if parent was cloned
            if splat.parent is not None and splat.parent.id in id_mapping:
                parent_id = id_mapping[splat.parent.id]
                clone.parent = new_registry.safe_get_splat(parent_id)
            return clone
        
        # Collect splats from both registries
        splats_a = registry_a.get_all_splats()
        splats_b = registry_b.get_all_splats()
        
        # Track ID mappings for parent updates
        id_mapping = {}
        
        # Add splats from registry A
        for splat in splats_a:
            cloned = clone_with_parent_update(splat, id_mapping)
            new_registry.register(cloned)
            id_mapping[splat.id] = cloned.id
        
        # Add splats from registry B (handling conflicts)
        for splat in splats_b:
            existing = new_registry.safe_get_splat(splat.id)
            
            if existing is None:
                # No conflict - add normally
                cloned = clone_with_parent_update(splat, id_mapping)
                new_registry.register(cloned)
                id_mapping[splat.id] = cloned.id
            else:
                # Handle ID conflict according to strategy
                if strategy == "replace":
                    # Replace with B's splat
                    new_registry.unregister(existing)
                    cloned = clone_with_parent_update(splat, id_mapping)
                    new_registry.register(cloned)
                    id_mapping[splat.id] = cloned.id
                elif strategy == "keep_a":
                    # Keep A's splat (do nothing)
                    pass
                elif strategy == "keep_b":
                    # Replace with B's splat
                    new_registry.unregister(existing)
                    cloned = clone_with_parent_update(splat, id_mapping)
                    new_registry.register(cloned)
                    id_mapping[splat.id] = cloned.id
        
        # Fix any remaining parent-child relationship issues
        new_registry.repair_integrity()
        
        return new_registry
    
    @staticmethod
    def export_to_dict(registry: SplatRegistry) -> Dict[str, Any]:
        """Export registry to a dictionary for serialization.
        
        Args:
            registry: SplatRegistry to export
            
        Returns:
            Dictionary representation of registry
        """
        result = {
            "embedding_dim": registry.embedding_dim,
            "hierarchy": registry.hierarchy.to_dict(),
            "splats": []
        }
        
        # Export all splats
        for splat in registry.get_all_splats():
            # Capture basic splat properties
            splat_dict = {
                "id": splat.id,
                "level": splat.level,
                "position": splat.position.tolist(),
                "covariance": splat.covariance.tolist(),
                "amplitude": splat.amplitude,
                "lifetime": splat.lifetime,
                "info_contribution": splat.info_contribution,
                "activation_history": splat.activation_history.get_values(),
                "parent_id": splat.parent.id if splat.parent else None,
                "children_ids": [child.id for child in splat.children] if splat.children else []
            }
            
            result["splats"].append(splat_dict)
        
        # Add registry statistics
        result["stats"] = {
            "registered_count": registry.registered_count,
            "unregistered_count": registry.unregistered_count,
            "recovery_count": registry.recovery_count
        }
        
        return result
    
    @staticmethod
    def import_from_dict(data: Dict[str, Any]) -> SplatRegistry:
        """Import registry from a dictionary representation.
        
        Args:
            data: Dictionary representation of registry
            
        Returns:
            New SplatRegistry instance
            
        Raises:
            ValueError: If data is invalid or incomplete
        """
        # Validate required fields
        required_fields = ["embedding_dim", "hierarchy", "splats"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' in import data")
        
        # Create hierarchy
        hierarchy = Hierarchy.from_dict(data["hierarchy"])
        
        # Create registry
        registry = SplatRegistry(
            hierarchy=hierarchy,
            embedding_dim=data["embedding_dim"]
        )
        
        # First pass: Create all splats without relationships
        splats_by_id = {}
        for splat_data in data["splats"]:
            try:
                position = np.array(splat_data["position"])
                covariance = np.array(splat_data["covariance"])
                
                splat = Splat(
                    dim=registry.embedding_dim,
                    position=position,
                    covariance=covariance,
                    amplitude=splat_data["amplitude"],
                    level=splat_data["level"],
                    id=splat_data["id"]
                )
                
                # Restore non-relationship properties
                splat.lifetime = splat_data.get("lifetime", 0)
                splat.info_contribution = splat_data.get("info_contribution", 0.0)
                
                # Restore activation history
                activation_history = splat_data.get("activation_history", [])
                for value in activation_history:
                    splat.activation_history.add(value)
                
                # Add to registry
                registry.register(splat)
                
                # Store for relationship resolution
                splats_by_id[splat.id] = splat
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to import splat: {e}")
        
        # Second pass: Resolve relationships
        for splat_data in data["splats"]:
            splat_id = splat_data["id"]
            
            # Skip if splat wasn't imported
            if splat_id not in splats_by_id:
                continue
            
            splat = splats_by_id[splat_id]
            
            # Resolve parent relationship
            parent_id = splat_data.get("parent_id")
            if parent_id is not None and parent_id in splats_by_id:
                splat.parent = splats_by_id[parent_id]
            
            # Resolve children relationships
            children_ids = splat_data.get("children_ids", [])
            for child_id in children_ids:
                if child_id in splats_by_id:
                    child = splats_by_id[child_id]
                    splat.children.add(child)
        
        # Optional: Restore registry statistics if available
        if "stats" in data:
            stats = data["stats"]
            registry.registered_count = stats.get("registered_count", registry.registered_count)
            registry.unregistered_count = stats.get("unregistered_count", registry.unregistered_count)
            registry.recovery_count = stats.get("recovery_count", registry.recovery_count)
        
        # Fix any relationship issues
        registry.repair_integrity()
        
        return registry
