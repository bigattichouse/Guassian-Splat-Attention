"""
Compatibility module for Hierarchical Splat Attention (HSA).

This module provides backward compatibility with the original attention.py interface
to ensure existing code continues to work with the new modular structure.

Import classes and functions from this module exactly as before.
"""

# Import everything from the new modular structure
from .attention.base import (
    AttentionComputer as BaseAttentionComputer,
    mahalanobis_batch,
    gauss_kernel_batch,
    find_relevant_tokens
)

from .attention.implementations import (
    DenseAttentionComputer as DenseAttentionComputerImpl,
    SparseAttentionComputer as SparseAttentionComputerImpl,
    SpatialAttentionComputer as SpatialAttentionComputerImpl
)

from .attention.metrics import (
    SplatAttentionMetrics,
    compute_pairwise_attention_correlation,
    compute_attention_similarity,
    analyze_attention_patterns
)

from .attention.pytorch import (
    HSAMultiheadAttention,
    HSAAttentionFunction,
    hsa_attention
)

from .attention.factory import (
    create_attention_computer as _factory_create_attention_computer,
    create_splat_registry,
    create_hierarchy,
    create_preset_config,
    estimate_optimal_params,
    adjust_config_for_task,
    create_hierarchy_for_data
)

# Re-export important classes and functions with their original names
# to maintain backward compatibility

# Main attention computer class (backward compatibility)
class AttentionComputer(BaseAttentionComputer):
    """
    Compatibility wrapper for AttentionComputer that delegates to the appropriate
    implementation based on parameters.
    """
    
    def __init__(self, hierarchy, sparse_topk=64, max_splat_radius=3.0, efficient=True, use_spatial=False):
        """Initialize the attention computer with compatibility parameters."""
        super().__init__(hierarchy, sparse_topk, max_splat_radius)
        self.efficient = efficient
        self.use_spatial = use_spatial
        
        # Create the appropriate implementation based on parameters
        if not efficient:
            self._impl = DenseAttentionComputerImpl(hierarchy, sparse_topk, max_splat_radius)
        elif use_spatial:
            self._impl = SpatialAttentionComputerImpl(hierarchy, sparse_topk, max_splat_radius)
        else:
            self._impl = SparseAttentionComputerImpl(hierarchy, sparse_topk, max_splat_radius)
    
    def compute_attention(self, tokens, splat_registry):
        """Delegate to the appropriate implementation."""
        return self._impl.compute_attention(tokens, splat_registry)
    
    def compute_splat_attention_map(self, tokens, splat, max_tokens=200):
        """Delegate to the appropriate implementation."""
        return self._impl.compute_splat_attention_map(tokens, splat, max_tokens)
    
    # Expose internal implementation methods for compatibility with tests
    
    def _compute_attention_sparse_optimized(self, tokens, splat_registry):
        """Compatibility method that delegates to the sparse implementation."""
        if hasattr(self._impl, '_compute_attention_sparse_optimized'):
            return self._impl._compute_attention_sparse_optimized(tokens, splat_registry)
        elif isinstance(self._impl, SparseAttentionComputerImpl):
            # SparseAttentionComputer has this method
            return self._impl._compute_attention_sparse_optimized(tokens, splat_registry)
        else:
            # Fallback to regular compute_attention
            return self._impl.compute_attention(tokens, splat_registry)
    
    def _compute_attention_dense(self, tokens, splat_registry):
        """Compatibility method that delegates to the dense implementation."""
        if hasattr(self._impl, '_compute_attention_dense'):
            return self._impl._compute_attention_dense(tokens, splat_registry)
        elif isinstance(self._impl, DenseAttentionComputerImpl):
            # DenseAttentionComputer has this method directly
            return self._impl.compute_attention(tokens, splat_registry)
        else:
            # Try to find a dense method in the implementation
            if hasattr(self._impl, '_compute_attention_dense'):
                return self._impl._compute_attention_dense(tokens, splat_registry)
            # Final fallback
            return self._impl.compute_attention(tokens, splat_registry)
    
    def _compute_level_attention_sparse(self, tokens, splats):
        """Compatibility method that delegates to the implementation."""
        if hasattr(self._impl, '_compute_level_attention_sparse'):
            return self._impl._compute_level_attention_sparse(tokens, splats)
        # No good fallback for this one
        raise NotImplementedError(
            "_compute_level_attention_sparse not available in the current implementation"
        )
    
    def _apply_topk_sparsity(self, attention_matrix):
        """Compatibility method that delegates to the base implementation."""
        return super()._apply_topk_sparsity(attention_matrix)
    
    def _apply_topk_sparsity_sparse(self, attention_matrix):
        """Compatibility method that delegates to the base implementation."""
        return super()._apply_topk_sparsity_sparse(attention_matrix)

# Define the actual implementation classes for external use with proper inheritance
class DenseAttentionComputer(AttentionComputer):
    """Dense implementation of AttentionComputer."""
    
    def __init__(self, hierarchy, sparse_topk=64, max_splat_radius=3.0):
        """Initialize DenseAttentionComputer."""
        AttentionComputer.__init__(self, hierarchy, sparse_topk, max_splat_radius, efficient=False)

class SparseAttentionComputer(AttentionComputer):
    """Sparse implementation of AttentionComputer."""
    
    def __init__(self, hierarchy, sparse_topk=64, max_splat_radius=3.0):
        """Initialize SparseAttentionComputer."""
        AttentionComputer.__init__(self, hierarchy, sparse_topk, max_splat_radius, efficient=True, use_spatial=False)

class SpatialAttentionComputer(AttentionComputer):
    """Spatial implementation of AttentionComputer."""
    
    def __init__(self, hierarchy, sparse_topk=64, max_splat_radius=3.0):
        """Initialize SpatialAttentionComputer."""
        AttentionComputer.__init__(self, hierarchy, sparse_topk, max_splat_radius, efficient=True, use_spatial=True)
        
        # Add direct access to the implementation's attributes
        self.token_index = None
        self.splat_indices = {}
    
    def _build_token_index(self, tokens):
        """Build a spatial index for tokens for efficient nearest neighbor queries."""
        if isinstance(self._impl, SpatialAttentionComputerImpl):
            result = self._impl._build_token_index(tokens)
            # Update our own token_index for tests that check this directly
            self.token_index = self._impl.token_index
            return result
        else:
            raise NotImplementedError("_build_token_index not available in non-spatial implementation")
    
    def _build_splat_index(self, splats, level):
        """Build a spatial index for splats at a specific level."""
        if isinstance(self._impl, SpatialAttentionComputerImpl):
            result = self._impl._build_splat_index(splats, level)
            # Update our own splat_indices for tests that check this directly
            self.splat_indices = self._impl.splat_indices
            return result
        else:
            raise NotImplementedError("_build_splat_index not available in non-spatial implementation")

# Factory function that uses the original interface
def create_attention_computer(
    hierarchy, 
    sparse_topk=64,
    efficient=True,
    use_spatial=False,
    max_splat_radius=3.0,
    device="cpu"
) -> AttentionComputer:
    """
    Create an appropriate attention computer based on configuration.
    
    Args:
        hierarchy: Hierarchy configuration for attention
        sparse_topk: Number of top-k connections to keep per token
        efficient: Whether to use optimized implementations
        use_spatial: Whether to use the spatial indexing implementation
        max_splat_radius: Maximum radius of influence for splats
        device: Compute device ("cpu" or "cuda")
        
    Returns:
        AttentionComputer instance appropriate for the configuration
    """
    # Create the appropriate implementation
    if not efficient:
        return DenseAttentionComputer(hierarchy, sparse_topk, max_splat_radius)
    elif use_spatial:
        return SpatialAttentionComputer(hierarchy, sparse_topk, max_splat_radius)
    else:
        return SparseAttentionComputer(hierarchy, sparse_topk, max_splat_radius)

# Maintain backward compatibility for important functions
def compute_attention(tokens, splat_registry, sparse_topk=64, efficient=True, use_spatial=False):
    """Compatibility function to compute attention without creating a class instance."""
    computer = create_attention_computer(
        hierarchy=splat_registry.hierarchy,
        sparse_topk=sparse_topk,
        efficient=efficient,
        use_spatial=use_spatial
    )
    return computer.compute_attention(tokens, splat_registry)

# Additional compatibility functions that might have been used in the original interface
def initialize_splats(tokens, hierarchy_config, n_neighbors=10, affinity='nearest_neighbors', random_seed=None):
    """Compatibility function for initializing splats."""
    from .attention.initialization import initialize_splats as init_splats
    return init_splats(tokens, hierarchy_config, n_neighbors, affinity, random_seed)

def reinitialize_splat(splat, data_points):
    """Compatibility function for reinitializing a splat."""
    from .attention.initialization import reinitialize_splat as reinit_splat
    return reinit_splat(splat, data_points)

def check_adaptation_triggers(splat_registry, metrics_tracker, **kwargs):
    """Compatibility function for checking adaptation triggers."""
    from .attention.adaptation import check_adaptation_triggers as check_triggers
    return check_triggers(splat_registry, metrics_tracker, **kwargs)

def perform_adaptations(splat_registry, adaptations, tokens, **kwargs):
    """Compatibility function for performing adaptations."""
    from .attention.adaptation import perform_adaptations as perform_adapt
    return perform_adapt(splat_registry, adaptations, tokens, **kwargs)

def replace_attention_with_hsa(model, hsa_config, attention_layer_pattern="attention", replace_in_place=True):
    """Compatibility function for replacing attention in a model."""
    from .attention.model_integration import replace_attention_with_hsa as replace_attn
    return replace_attn(model, hsa_config, attention_layer_pattern, replace_in_place)

# For completeness, re-export all relevant types
__all__ = [
    # Main classes
    'AttentionComputer',
    'SplatAttentionMetrics',
    'HSAMultiheadAttention',
    
    # Implementation variants
    'DenseAttentionComputer',
    'SparseAttentionComputer',
    'SpatialAttentionComputer',
    
    # Factory functions
    'create_attention_computer',
    'create_splat_registry',
    'create_hierarchy',
    'create_preset_config',
    
    # Core functions
    'compute_attention',
    'initialize_splats',
    'reinitialize_splat',
    'check_adaptation_triggers',
    'perform_adaptations',
    'replace_attention_with_hsa',
    
    # Utility functions
    'mahalanobis_batch',
    'gauss_kernel_batch',
    'find_relevant_tokens',
    'compute_pairwise_attention_correlation',
    'compute_attention_similarity',
    'analyze_attention_patterns',
]
