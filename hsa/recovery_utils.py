"""
Utility functions for failure recovery in Hierarchical Splat Attention (HSA).

This module provides helper functions used by the main recovery mechanisms.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any

from .splat import Splat
from .registry import SplatRegistry

# Configure logging
logger = logging.getLogger(__name__)


def repair_covariance_matrices(
    registry: SplatRegistry,
    target_condition_number: float = 10.0
) -> int:
    """Repair ill-conditioned covariance matrices.
    
    Args:
        registry: SplatRegistry to repair
        target_condition_number: Target condition number for repaired matrices
        
    Returns:
        Number of matrices repaired
    """
    repaired_count = 0
    
    for splat in registry.get_all_splats():
        try:
            # Check for NaN or Inf values
            if np.isnan(splat.covariance).any() or np.isinf(splat.covariance).any():
                # Replace with identity matrix scaled by trace
                trace = max(1e-3, np.trace(np.abs(np.nan_to_num(splat.covariance))))
                dim = splat.dim
                splat.update_parameters(covariance=np.eye(dim) * (trace / dim))
                repaired_count += 1
                continue
                
            # Check condition number
            eigenvalues = np.linalg.eigvalsh(splat.covariance)
            min_eig = np.min(eigenvalues)
            max_eig = np.max(eigenvalues)
            
            if min_eig <= 0:
                # Non-positive definite - replace with identity
                dim = splat.dim
                splat.update_parameters(covariance=np.eye(dim))
                repaired_count += 1
            elif max_eig / min_eig > target_condition_number:
                # Ill-conditioned - regularize eigenvalues
                eigvals, eigvecs = np.linalg.eigh(splat.covariance)
                
                # Adjust eigenvalues to improve condition number
                min_target = max_eig / target_condition_number
                eigvals = np.maximum(eigvals, min_target)
                
                # Reconstruct covariance
                new_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Ensure symmetry (handle numerical errors)
                new_cov = 0.5 * (new_cov + new_cov.T)
                
                splat.update_parameters(covariance=new_cov)
                repaired_count += 1
                
        except Exception as e:
            logger.error(f"Error repairing covariance for splat {splat.id}: {e}")
    
    return repaired_count


def create_random_splats(registry: SplatRegistry, level: str, count: int) -> int:
    """Create random splats at a specific level.
    
    Args:
        registry: SplatRegistry to update
        level: Level to create splats at
        count: Number of splats to create
        
    Returns:
        Number of splats created
    """
    dim = registry.embedding_dim
    created = 0
    
    # Get parent level
    parent_level = registry.hierarchy.get_parent_level(level)
    parent_splats = []
    
    if parent_level:
        parent_splats = list(registry.get_splats_at_level(parent_level))
    
    # Scale for covariance based on level
    level_idx = registry.hierarchy.get_level_index(level)
    scale = 1.0 + level_idx * 0.5  # Higher levels get larger covariance
    
    for i in range(count):
        try:
            # Generate random position
            position = np.random.normal(0, 1.0, dim)
            
            # Create covariance
            covariance = np.eye(dim) * scale
            
            # Select parent if available
            parent = None
            if parent_splats:
                parent = parent_splats[i % len(parent_splats)]
            
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
            logger.error(f"Failed to create random splat: {e}")
    
    return created


def recover_from_failures(
    registry: SplatRegistry,
    auto_fix: bool = True
) -> Dict[str, Any]:
    """Detect and recover from failures in the registry.
    
    This is a convenience function that wraps FailureRecovery for simple usage.
    
    Args:
        registry: SplatRegistry to recover
        auto_fix: Whether to automatically fix detected issues
        
    Returns:
        Recovery report dictionary
    """
    from .failure_recovery import FailureRecovery
    recovery = FailureRecovery(registry, auto_recovery=auto_fix)
    report = recovery.detect_and_recover()
    
    return report
