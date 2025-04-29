"""
Analysis module for Hierarchical Splat Attention (HSA) initialization.

This module provides functions to analyze token distributions, embedding space,
and splat effectiveness to improve initialization and adaptation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _analyze_token_coverage(
    tokens: np.ndarray, 
    registry: Any  # SplatRegistry
) -> float:
    """
    Analyze how well the initialized splats cover the token space.
    
    Args:
        tokens: Token embeddings
        registry: The initialized splat registry
        
    Returns:
        Coverage percentage (0-100)
    """
    # Skip if no tokens
    if len(tokens) == 0:
        return 0.0
        
    # Track which tokens are covered by any splat
    covered = np.zeros(len(tokens), dtype=bool)
    
    # Check coverage from each splat
    for splat in registry.splats.values():
        # Extract splat parameters
        pos = splat.position
        cov_inv = splat.covariance_inverse
        
        # Calculate distances from tokens to this splat
        diffs = tokens - pos
        try:
            distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
        except:
            # Fallback to Euclidean distance if Mahalanobis fails
            distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Mark tokens as covered if within 3 standard deviations
        covered = covered | (distances < 3.0)
    
    # Calculate coverage percentage
    coverage = 100.0 * np.mean(covered)
    return coverage

def analyze_embedding_space(
    tokens: np.ndarray,
    method: str = 'pca',
    n_components: int = 2
) -> Dict[str, Any]:
    """
    Analyze the embedding space to guide splat placement.
    
    Args:
        tokens: Token embeddings
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of components for the projection
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing embedding space using {method}")
    results = {
        "method": method,
        "projections": None,
        "clusters": None,
        "density": None
    }
    
    try:
        # Project to lower dimension for visualization and analysis
        if method == 'pca':
            from sklearn.decomposition import PCA
            projector = PCA(n_components=n_components)
            projections = projector.fit_transform(tokens)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            projector = TSNE(n_components=n_components, perplexity=min(30, tokens.shape[0] // 2))
            projections = projector.fit_transform(tokens)
        else:
            logger.warning(f"Unknown projection method: {method}")
            return results
        
        results["projections"] = projections
        
        # Try clustering in the projected space to identify natural clusters
        try:
            # First try to determine optimal number of clusters
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            max_clusters = min(10, tokens.shape[0] // 5)
            best_score = -1
            best_n_clusters = 2
            
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
                labels = kmeans.fit_predict(projections)
                
                try:
                    score = silhouette_score(projections, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                except:
                    continue
            
            # Cluster with the best number of clusters
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(projections)
            
            # Calculate cluster centers
            cluster_centers = np.zeros((best_n_clusters, tokens.shape[1]))
            for i in range(best_n_clusters):
                cluster_points = tokens[labels == i]
                if len(cluster_points) > 0:
                    cluster_centers[i] = np.mean(cluster_points, axis=0)
            
            results["clusters"] = {
                "labels": labels,
                "centers": cluster_centers,
                "n_clusters": best_n_clusters,
                "silhouette_score": best_score
            }
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
        
        # Calculate density information
        try:
            from scipy.spatial import distance
            
            # Calculate pairwise distances
            distances = distance.pdist(projections)
            results["density"] = {
                "mean_distance": np.mean(distances),
                "min_distance": np.min(distances),
                "max_distance": np.max(distances),
                "std_distance": np.std(distances)
            }
        except Exception as e:
            logger.warning(f"Density analysis failed: {e}")
    
    except Exception as e:
        logger.warning(f"Embedding space analysis failed: {e}")
    
    return results

def analyze_splat_distribution(
    registry: Any,  # SplatRegistry
    tokens: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Analyze the distribution and effectiveness of splats.
    
    Args:
        registry: The splat registry
        tokens: Optional token embeddings for coverage analysis
        
    Returns:
        Dictionary of analysis results
    """
    results = {
        "level_counts": {},
        "level_coverage": {},
        "avg_distances": {},
        "avg_amplitudes": {},
        "avg_covariance_traces": {}
    }
    
    # Count splats by level
    for level in registry.hierarchy.levels:
        level_splats = list(registry.get_splats_at_level(level))
        results["level_counts"][level] = len(level_splats)
        
        # Calculate average amplitude
        if level_splats:
            avg_amplitude = np.mean([splat.amplitude for splat in level_splats])
            avg_cov_trace = np.mean([np.trace(splat.covariance) for splat in level_splats])
            
            results["avg_amplitudes"][level] = avg_amplitude
            results["avg_covariance_traces"][level] = avg_cov_trace
        else:
            results["avg_amplitudes"][level] = 0.0
            results["avg_covariance_traces"][level] = 0.0
    
    # Calculate coverage by level if tokens provided
    if tokens is not None:
        for level in registry.hierarchy.levels:
            level_splats = list(registry.get_splats_at_level(level))
            
            # Skip if no splats at this level
            if not level_splats:
                results["level_coverage"][level] = 0.0
                continue
            
            # Create a temporary registry with only this level
            from ..data_structures import SplatRegistry
            temp_registry = SplatRegistry(registry.hierarchy)
            for splat in level_splats:
                temp_registry.register(splat)
            
            # Calculate coverage
            coverage = _analyze_token_coverage(tokens, temp_registry)
            results["level_coverage"][level] = coverage
            
            # Calculate average distances between splats
            if len(level_splats) > 1:
                distances = []
                for i in range(len(level_splats)):
                    for j in range(i+1, len(level_splats)):
                        distance = np.linalg.norm(
                            level_splats[i].position - level_splats[j].position
                        )
                        distances.append(distance)
                
                results["avg_distances"][level] = np.mean(distances)
            else:
                results["avg_distances"][level] = 0.0
    
    return results

def identify_coverage_gaps(
    tokens: np.ndarray,
    registry: Any,  # SplatRegistry
    coverage_threshold: float = 3.0
) -> List[np.ndarray]:
    """
    Identify regions in the embedding space with insufficient splat coverage.
    
    Args:
        tokens: Token embeddings
        registry: The splat registry
        coverage_threshold: Distance threshold to consider a token covered
        
    Returns:
        List of positions for potential new splats
    """
    # Track which tokens are covered by any splat
    covered = np.zeros(len(tokens), dtype=bool)
    
    # Check coverage from each splat
    for splat in registry.splats.values():
        # Extract splat parameters
        pos = splat.position
        cov_inv = splat.covariance_inverse
        
        # Calculate distances from tokens to this splat
        diffs = tokens - pos
        try:
            distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
        except:
            # Fallback to Euclidean distance if Mahalanobis fails
            distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Mark tokens as covered if within threshold
        covered = covered | (distances < coverage_threshold)
    
    # Get uncovered tokens
    uncovered_indices = np.where(~covered)[0]
    uncovered_tokens = tokens[uncovered_indices]
    
    # If few uncovered tokens, return them directly
    if len(uncovered_tokens) <= 5:
        return [token for token in uncovered_tokens]
    
    # For many uncovered tokens, cluster them
    from sklearn.cluster import KMeans
    
    n_clusters = min(5, len(uncovered_tokens) // 5)
    if n_clusters < 1:
        n_clusters = 1
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    kmeans.fit(uncovered_tokens)
    
    # Return cluster centers as potential new splat positions
    return [center for center in kmeans.cluster_centers_]

def analyze_hierarchy_effectiveness(
    registry: Any,  # SplatRegistry
    tokens: np.ndarray
) -> Dict[str, float]:
    """
    Analyze the effectiveness of the hierarchical structure.
    
    Args:
        registry: The splat registry
        tokens: Token embeddings
        
    Returns:
        Dictionary of effectiveness metrics
    """
    # Calculate coverage by level
    level_coverage = {}
    for level in registry.hierarchy.levels:
        level_splats = list(registry.get_splats_at_level(level))
        
        # Skip if no splats at this level
        if not level_splats:
            level_coverage[level] = 0.0
            continue
        
        # Create a temporary registry with only this level
        from ..data_structures import SplatRegistry
        temp_registry = SplatRegistry(registry.hierarchy)
        for splat in level_splats:
            temp_registry.register(splat)
        
        # Calculate coverage
        coverage = _analyze_token_coverage(tokens, temp_registry)
        level_coverage[level] = coverage
    
    # Calculate parent-child relationship effectiveness
    relationship_metrics = {}
    
    # For each level (except top level), calculate how well parents cover their children
    for level_idx, level in enumerate(registry.hierarchy.levels[:-1]):  # Skip top level
        child_splats = list(registry.get_splats_at_level(level))
        
        # Skip if no splats at this level
        if not child_splats:
            relationship_metrics[f"{level}_coverage_by_parents"] = 0.0
            continue
        
        # Count children with parents
        children_with_parents = sum(1 for splat in child_splats if splat.parent is not None)
        parent_coverage_ratio = children_with_parents / len(child_splats)
        
        # Calculate average distance from child to parent
        child_parent_distances = []
        for child in child_splats:
            if child.parent is not None:
                distance = np.linalg.norm(child.position - child.parent.position)
                child_parent_distances.append(distance)
        
        if child_parent_distances:
            avg_distance = np.mean(child_parent_distances)
        else:
            avg_distance = 0.0
        
        relationship_metrics[f"{level}_parent_coverage"] = parent_coverage_ratio
        relationship_metrics[f"{level}_avg_parent_distance"] = avg_distance
    
    # Combine metrics
    metrics = {
        "overall_coverage": _analyze_token_coverage(tokens, registry),
        **{f"{level}_coverage": cov for level, cov in level_coverage.items()},
        **relationship_metrics
    }
    
    return metrics

def analyze_initialization_quality(
    registry: Any,  # SplatRegistry
    tokens: np.ndarray
) -> Dict[str, Any]:
    """
    Comprehensive analysis of initialization quality.
    
    Args:
        registry: The splat registry
        tokens: Token embeddings
        
    Returns:
        Dictionary of quality metrics and recommendations
    """
    # Analyze splat distribution
    distribution = analyze_splat_distribution(registry, tokens)
    
    # Analyze hierarchy effectiveness
    hierarchy_metrics = analyze_hierarchy_effectiveness(registry, tokens)
    
    # Identify coverage gaps
    coverage_gaps = identify_coverage_gaps(tokens, registry)
    
    # Calculate overall effectiveness score
    overall_coverage = hierarchy_metrics["overall_coverage"]
    
    # Calculate hierarchy balance score
    level_counts = distribution["level_counts"]
    expected_ratio = 2.0  # Ideal ratio between consecutive levels
    
    level_ratio_scores = []
    levels = registry.hierarchy.levels
    for i in range(len(levels) - 1):
        current_count = level_counts.get(levels[i], 0)
        next_count = level_counts.get(levels[i+1], 0)
        
        # Avoid division by zero
        if next_count == 0:
            next_count = 1
        
        actual_ratio = current_count / next_count
        ratio_score = 1.0 - min(1.0, abs(actual_ratio - expected_ratio) / expected_ratio)
        level_ratio_scores.append(ratio_score)
    
    hierarchy_balance = np.mean(level_ratio_scores) if level_ratio_scores else 0.0
    
    # Calculate relationship quality score
    relationship_scores = []
    for key, value in hierarchy_metrics.items():
        if "_parent_coverage" in key:
            relationship_scores.append(value)
    
    relationship_quality = np.mean(relationship_scores) if relationship_scores else 0.0
    
    # Calculate overall quality score
    quality_score = (
        0.5 * overall_coverage / 100.0 +  # Coverage (0-1)
        0.3 * hierarchy_balance +         # Hierarchy balance (0-1)
        0.2 * relationship_quality        # Relationship quality (0-1)
    )
    
    # Generate recommendations
    recommendations = []
    
    if overall_coverage < 80:
        recommendations.append("Increase splat count at the Token level for better coverage")
    
    if hierarchy_balance < 0.7:
        recommendations.append("Adjust splat counts between levels for more balanced hierarchy")
    
    if relationship_quality < 0.8:
        recommendations.append("Rebuild parent-child relationships for better hierarchy")
    
    if len(coverage_gaps) > 0:
        recommendations.append(f"Add {len(coverage_gaps)} new splats to cover gaps in embedding space")
    
    # Compile results
    results = {
        "distribution": distribution,
        "hierarchy_metrics": hierarchy_metrics,
        "coverage_gaps": len(coverage_gaps),
        "quality_scores": {
            "overall": quality_score,
            "coverage": overall_coverage / 100.0,
            "hierarchy_balance": hierarchy_balance,
            "relationship_quality": relationship_quality
        },
        "recommendations": recommendations
    }
    
    return results
