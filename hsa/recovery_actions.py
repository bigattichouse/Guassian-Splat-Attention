"""
Recovery actions for Hierarchical Splat Attention (HSA).

This module provides implementations of specific recovery actions to address
various failure modes in the HSA structure.
"""

from enum import Enum, auto
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

from .splat import Splat
from .registry import SplatRegistry
from .failure_detection_types import FailureType
from .recovery_utils import create_random_splats

# Configure logging
logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions that can be performed."""
    REPAIR_COVARIANCE = auto()         # Fix numerical issues in covariance
    POPULATE_LEVEL = auto()            # Add splats to empty level
    REPAIR_RELATIONSHIPS = auto()      # Fix parent-child relationships
    REBALANCE_HIERARCHY = auto()       # Redistribute splats across levels
    RESTART_ADAPTATION = auto()        # Reset adaptation parameters
    PRUNE_SPLATS = auto()              # Remove problematic splats
    MERGE_SIMILAR = auto()             # Merge similar splats
    SPLIT_COMPLEX = auto()             # Split overly complex splats
    RESET_AMPLITUDES = auto()          # Reset attention amplitudes


def recover_numerical_instability(
    registry: SplatRegistry, 
    failures: List[Tuple[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Recover from numerical instabilities.
    
    Args:
        registry: SplatRegistry to recover
        failures: List of (message, data) tuples
        
    Returns:
        Recovery action report or None if no action taken
    """
    # Extract splat IDs with numerical issues
    unstable_ids = []
    for _, data in failures:
        if isinstance(data, dict) and "splat_id" in data:
            unstable_ids.append(data["splat_id"])
    
    if not unstable_ids:
        return None
    
    # Perform recovery
    fixed_count = 0
    for splat_id in unstable_ids:
        try:
            # Get the splat
            splat = registry.get_splat(splat_id)
            
            # Create a stabilized covariance matrix
            # We'll use identity matrix scaled by trace as a safe fallback
            old_cov = splat.covariance
            trace = max(1e-3, np.trace(old_cov))
            dim = splat.dim
            
            # Create a well-conditioned covariance
            new_cov = np.eye(dim) * (trace / dim)
            
            # Update splat parameters
            splat.update_parameters(covariance=new_cov)
            
            # Verify fix worked
            try:
                _ = np.linalg.inv(splat.covariance)
                fixed_count += 1
            except np.linalg.LinAlgError:
                logger.error(f"Failed to fix covariance for splat {splat_id}")
                
        except Exception as e:
            logger.error(f"Error recovering from numerical instability: {e}")
    
    if fixed_count > 0:
        return {
            "action": "repair_covariance",
            "splats_fixed": fixed_count,
            "total_unstable": len(unstable_ids)
        }
    
    return None


def recover_empty_level(
    registry: SplatRegistry, 
    failures: List[Tuple[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Recover from empty levels by creating new splats.
    
    Args:
        registry: SplatRegistry to update
        failures: List of (message, data) tuples
        
    Returns:
        Recovery action report or None if no action taken
    """
    # Extract empty levels
    empty_levels = []
    for _, data in failures:
        if isinstance(data, dict) and "level" in data:
            empty_levels.append(data["level"])
    
    if not empty_levels:
        return None
    
    # Import birth module for creating new splats
    try:
        from .birth import create_initial_splats
        
        # Perform recovery
        total_created = 0
        levels_fixed = []
        
        for level in empty_levels:
            # Create new splats at this level
            # First, get parent level
            parent_level = registry.hierarchy.get_parent_level(level)
            
            # Get initial splat count from hierarchy
            init_count = registry.hierarchy.get_num_init_splats(level)
            
            # Create initial splats
            created = 0
            
            if parent_level:
                # Get parent splats
                parent_splats = list(registry.get_splats_at_level(parent_level))
                
                if parent_splats:
                    # Create splats with parents
                    dim = registry.embedding_dim
                    
                    # Calculate splats per parent
                    splats_per_parent = max(1, init_count // len(parent_splats))
                    scale = 0.5  # Smaller covariance for child splats
                    
                    for parent in parent_splats:
                        for _ in range(splats_per_parent):
                            try:
                                # Generate position near parent
                                offset = np.random.normal(0, 0.3, dim)
                                position = parent.position + offset
                                
                                # Create covariance (smaller than parent)
                                covariance = np.eye(dim) * scale
                                
                                # Create splat
                                splat = Splat(
                                    dim=dim,
                                    position=position,
                                    covariance=covariance,
                                    amplitude=1.0,
                                    level=level,
                                    parent=parent
                                )
                                
                                registry.register(splat)
                                created += 1
                            except Exception as e:
                                logger.error(f"Failed to create child splat: {e}")
                else:
                    # No parents available - create random splats
                    created = create_random_splats(registry, level, init_count)
            else:
                # Top level - create random splats
                created = create_random_splats(registry, level, init_count)
            
            if created > 0:
                total_created += created
                levels_fixed.append(level)
        
        if total_created > 0:
            return {
                "action": "populate_level",
                "levels_fixed": levels_fixed,
                "splats_created": total_created
            }
        
    except ImportError:
        logger.error("Could not import birth module for level recovery")
        
        # Fallback: use simpler approach
        for level in empty_levels:
            create_random_splats(registry, level, 5)  # Create 5 splats
            
        return {
            "action": "populate_level",
            "levels_fixed": empty_levels,
            "splats_created": 5 * len(empty_levels)
        }
        
    return None


def recover_orphaned_splats(
    registry: SplatRegistry, 
    failures: List[Tuple[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Recover from orphaned splats by fixing parent-child relationships.
    
    Args:
        registry: SplatRegistry to update
        failures: List of (message, data) tuples
        
    Returns:
        Recovery action report or None if no action taken
    """
    # Extract orphaned splat IDs
    orphaned_ids = []
    for _, data in failures:
        if isinstance(data, dict) and "splat_id" in data:
            orphaned_ids.append(data["splat_id"])
    
    if not orphaned_ids:
        return None
    
    # Perform recovery
    fixed_count = 0
    for splat_id in orphaned_ids:
        try:
            # Get the splat
            splat = registry.get_splat(splat_id)
            
            # Skip if at highest level (these don't need parents)
            level_idx = registry.hierarchy.get_level_index(splat.level)
            if level_idx == len(registry.hierarchy.levels) - 1:
                continue
            
            # Find a suitable parent
            parent_level = registry.hierarchy.get_parent_level(splat.level)
            if not parent_level:
                continue
                
            parent_candidates = list(registry.get_splats_at_level(parent_level))
            
            if not parent_candidates:
                # No parents available at parent level
                continue
            
            # Find nearest parent
            best_parent = None
            min_distance = float('inf')
            
            for candidate in parent_candidates:
                distance = np.linalg.norm(splat.position - candidate.position)
                if distance < min_distance:
                    min_distance = distance
                    best_parent = candidate
            
            if best_parent:
                # Update parent reference
                if splat.parent:
                    # Remove from old parent's children
                    if splat in splat.parent.children:
                        splat.parent.children.remove(splat)
                
                # Set new parent
                splat.parent = best_parent
                
                # Add to new parent's children
                best_parent.children.add(splat)
                
                fixed_count += 1
                
        except Exception as e:
            logger.error(f"Error recovering orphaned splat {splat_id}: {e}")
    
    if fixed_count > 0:
        return {
            "action": "repair_relationships",
            "splats_fixed": fixed_count,
            "total_orphaned": len(orphaned_ids)
        }
    
    return None


def recover_adaptation_stagnation(
    registry: SplatRegistry,
    adaptation_controller,
    failures: List[Tuple[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Recover from adaptation stagnation by resetting adaptation state.
    
    Args:
        registry: SplatRegistry to update
        adaptation_controller: AdaptationController instance
        failures: List of (message, data) tuples
        
    Returns:
        Recovery action report or None if no action taken
    """
    # Check if adaptation controller is available
    if adaptation_controller is None:
        return {
            "action": "restart_adaptation",
            "success": False,
            "reason": "Adaptation controller not available"
        }
    
    # Reset adaptation state
    try:
        # Reset statistics
        adaptation_controller.reset_statistics()
        
        # Perform forced adaptations to kickstart the process
        # Force a few births to introduce new splats
        for level in registry.hierarchy.levels:
            # Create one new splat per level
            adapted = False
            
            level_splats = list(registry.get_splats_at_level(level))
            if not level_splats:
                continue
            
            # Try to perform mitosis on a random splat
            random_idx = np.random.randint(0, len(level_splats))
            random_splat = level_splats[random_idx]
            
            try:
                # Import mitosis module
                from .mitosis import perform_mitosis
                result = perform_mitosis(registry, random_splat.id)
                if result:
                    adapted = True
            except Exception:
                pass
            
            # If mitosis failed, try death followed by births
            if not adapted:
                try:
                    from .birth import perform_birth
                    # Create a new splat
                    perform_birth(registry, level)
                    adapted = True
                except Exception:
                    pass
        
        return {
            "action": "restart_adaptation",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error recovering from adaptation stagnation: {e}")
        
        return {
            "action": "restart_adaptation",
            "success": False,
            "reason": f"Error: {str(e)}"
        }
