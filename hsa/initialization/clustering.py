"""
Clustering module for Hierarchical Splat Attention (HSA) initialization.

This module provides clustering algorithms and methods for determining
initial splat positions, centers, and distributions in the embedding space.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neighbors import kneighbors_graph

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _initialize_splat_centers(
    data_points: np.ndarray, 
    n_clusters: int,
    n_neighbors: int = 10,
    affinity: str = 'nearest_neighbors',
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Initialize splat centers using spectral clustering.
    
    Args:
        data_points: Sampled data points (token embeddings)
        n_clusters: Number of clusters (splats) to create
        n_neighbors: Number of neighbors for affinity graph
        affinity: Affinity method for spectral clustering
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Cluster centers [n_clusters, embedding_dim]
    """
    # Handle edge case: if n_clusters equals or exceeds data points
    if n_clusters >= len(data_points):
        logger.warning(
            f"Number of clusters ({n_clusters}) >= number of data points ({len(data_points)}). "
            "Using data points directly as centers."
        )
        # If more clusters than points, take all points and then random repeats
        if n_clusters > len(data_points):
            extra_indices = np.random.choice(
                len(data_points), 
                size=n_clusters - len(data_points), 
                replace=True
            )
            return np.vstack([data_points, data_points[extra_indices]])
        return data_points
    
    try:
        # Construct k-nearest neighbors graph
        connectivity = kneighbors_graph(
            data_points, 
            n_neighbors=min(n_neighbors, len(data_points) - 1),
            include_self=True,
            mode='distance'
        )
        
        # Convert to affinity matrix (using Gaussian kernel)
        sigma = np.mean(connectivity.data)  # Estimate scale parameter
        connectivity.data = np.exp(-connectivity.data**2 / (2 * sigma**2))
        affinity_matrix = 0.5 * (connectivity + connectivity.T)  # Make symmetric
        
        # Create and fit spectral clustering model
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=random_seed,
            n_init=5  # Multiple initializations for stability
        )
        
        # Fit the model and get cluster labels
        cluster_labels = spectral.fit_predict(affinity_matrix)
        
        # Calculate cluster centers
        centers = np.zeros((n_clusters, data_points.shape[1]))
        for i in range(n_clusters):
            cluster_points = data_points[cluster_labels == i]
            if len(cluster_points) > 0:
                centers[i] = np.mean(cluster_points, axis=0)
            else:
                # If empty cluster, use a random data point
                centers[i] = data_points[np.random.randint(len(data_points))]
    
    except Exception as e:
        logger.warning(f"Spectral clustering failed: {e}. Falling back to K-means.")
        # Fall back to K-means if spectral clustering fails
        centers = _initialize_kmeans_centers(data_points, n_clusters, random_seed)
    
    return centers

def _initialize_kmeans_centers(
    data_points: np.ndarray, 
    n_clusters: int,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Initialize splat centers using K-means clustering.
    
    Args:
        data_points: Sampled data points (token embeddings)
        n_clusters: Number of clusters (splats) to create
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Cluster centers [n_clusters, embedding_dim]
    """
    # Handle edge case: if n_clusters equals or exceeds data points
    if n_clusters >= len(data_points):
        logger.warning(
            f"Number of clusters ({n_clusters}) >= number of data points ({len(data_points)}). "
            "Using data points directly as centers."
        )
        # If more clusters than points, take all points and then random repeats
        if n_clusters > len(data_points):
            extra_indices = np.random.choice(
                len(data_points), 
                size=n_clusters - len(data_points), 
                replace=True
            )
            return np.vstack([data_points, data_points[extra_indices]])
        return data_points
    
    try:
        # Create and fit K-means model
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            n_init=10  # Multiple initializations for better results
        )
        
        # Fit the model
        kmeans.fit(data_points)
        
        # Return cluster centers
        return kmeans.cluster_centers_
        
    except Exception as e:
        logger.warning(f"K-means clustering failed: {e}. Using random initialization.")
        # Fall back to random initialization
        random_indices = np.random.choice(len(data_points), size=n_clusters, replace=False)
        return data_points[random_indices]

def find_optimal_clusters(
    data_points: np.ndarray,
    max_clusters: int = 10,
    method: str = 'silhouette',
    random_seed: Optional[int] = None
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Find the optimal number of clusters for a dataset.
    
    Args:
        data_points: Data points to cluster
        max_clusters: Maximum number of clusters to consider
        method: Method for determining optimal clusters ('silhouette' or 'elbow')
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Tuple of (optimal_n_clusters, cluster_centers, cluster_labels)
    """
    from sklearn.metrics import silhouette_score
    
    # Limit max_clusters based on data points
    max_clusters = min(max_clusters, len(data_points) // 5)
    max_clusters = max(2, max_clusters)  # At least 2 clusters
    
    best_score = -1
    best_n_clusters = 2
    best_labels = None
    
    # Try different cluster counts
    for n_clusters in range(2, max_clusters + 1):
        try:
            # Fit KMeans
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_seed,
                n_init=5  # Use fewer initializations for speed
            )
            labels = kmeans.fit_predict(data_points)
            
            # Calculate score
            if method == 'silhouette':
                score = silhouette_score(data_points, labels)
            else:  # Default to inertia (elbow method)
                score = -kmeans.inertia_  # Negate so higher is better
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels
                best_centers = kmeans.cluster_centers_
        except Exception as e:
            logger.warning(f"Clustering with {n_clusters} clusters failed: {e}")
            continue
    
    # If all clustering attempts failed, use default
    if best_labels is None:
        best_n_clusters = min(2, len(data_points))
        kmeans = KMeans(
            n_clusters=best_n_clusters,
            random_state=random_seed,
            n_init=10
        )
        best_labels = kmeans.fit_predict(data_points)
        best_centers = kmeans.cluster_centers_
    
    return best_n_clusters, best_centers, best_labels

def find_empty_regions(
    data_points: np.ndarray,
    centers: np.ndarray,
    threshold: float = 2.0,
    max_regions: int = 3
) -> List[np.ndarray]:
    """
    Find regions in the embedding space that are far from existing centers.
    
    Args:
        data_points: Data points in the embedding space
        centers: Existing center positions
        threshold: Distance threshold to consider a region empty
        max_regions: Maximum number of empty regions to find
        
    Returns:
        List of positions for potential new centers
    """
    # Compute distances from each data point to its nearest center
    min_distances = np.ones(len(data_points)) * float('inf')
    
    for center in centers:
        # Calculate Euclidean distances
        diffs = data_points - center
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Update minimum distances
        min_distances = np.minimum(min_distances, distances)
    
    # Find points that are far from any existing center
    far_indices = np.where(min_distances > threshold)[0]
    
    if len(far_indices) < 5:  # Not enough far points
        return []
        
    far_points = data_points[far_indices]
    
    # Cluster the far points to identify distinct empty regions
    if len(far_points) >= 10:
        # Use KMeans to find clusters in the far points
        n_clusters = min(max_regions, len(far_points) // 5)
        if n_clusters < 1:
            n_clusters = 1
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        kmeans.fit(far_points)
        
        # Return cluster centers as potential new center locations
        return [center for center in kmeans.cluster_centers_]
    else:
        # Too few points for clustering, just return the far points
        return [point for point in far_points[:max_regions]]

def calculate_density_distribution(
    data_points: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculate the density distribution of data points in the embedding space.
    
    This is useful for understanding where new splats might be needed.
    
    Args:
        data_points: Data points in the embedding space
        n_bins: Number of bins for density histogram
        
    Returns:
        Dictionary with density distribution information
    """
    # Use PCA to reduce to 2D for density calculation
    from sklearn.decomposition import PCA
    
    # Skip if too few points
    if len(data_points) < 5:
        return {
            "density_map": None,
            "high_density_regions": [],
            "low_density_regions": []
        }
    
    try:
        # Reduce to 2D for density calculation
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(data_points)
        
        # Calculate 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            points_2d[:, 0], points_2d[:, 1], 
            bins=n_bins
        )
        
        # Normalize histogram
        hist = hist / np.sum(hist)
        
        # Find high and low density regions
        high_density_mask = hist > np.percentile(hist, 90)
        low_density_mask = hist < np.percentile(hist, 10)
        
        # Convert back to original space
        high_density_regions = []
        low_density_regions = []
        
        # Calculate bin centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        # For high density regions
        for i in range(n_bins):
            for j in range(n_bins):
                if high_density_mask[i, j]:
                    # Get 2D coordinates
                    point_2d = np.array([x_centers[i], y_centers[j]])
                    
                    # Project back to original space
                    # This is an approximation - find the nearest data point
                    distances = np.sum((points_2d - point_2d)**2, axis=1)
                    nearest_idx = np.argmin(distances)
                    high_density_regions.append(data_points[nearest_idx])
        
        # For low density regions
        for i in range(n_bins):
            for j in range(n_bins):
                if low_density_mask[i, j]:
                    # Get 2D coordinates
                    point_2d = np.array([x_centers[i], y_centers[j]])
                    
                    # Project back to original space - approximate with nearest point
                    distances = np.sum((points_2d - point_2d)**2, axis=1)
                    nearest_idx = np.argmin(distances)
                    low_density_regions.append(data_points[nearest_idx])
        
        return {
            "density_map": hist,
            "high_density_regions": high_density_regions,
            "low_density_regions": low_density_regions
        }
        
    except Exception as e:
        logger.warning(f"Density calculation failed: {e}")
        return {
            "density_map": None,
            "high_density_regions": [],
            "low_density_regions": []
        }
