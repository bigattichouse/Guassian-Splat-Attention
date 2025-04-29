"""
Visualization tools for HSA adaptation module.

This module provides visualization functions to understand adaptation decisions,
particularly focusing on the clustering analysis used in mitosis decisions.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import os
import time

from .data_structures import Splat
from .adaptation import should_perform_mitosis


def plot_splat(
    splat: Splat,
    reduced_dims: List[int],
    color: str = 'blue',
    alpha: float = 0.3,
    ax: Optional[plt.Axes] = None
) -> None:
    """
    Plot a splat as an ellipse in a 2D plane.
    
    Args:
        splat: The splat to visualize
        reduced_dims: The two dimensions to use for visualization [dim1, dim2]
        color: Color for the ellipse
        alpha: Transparency for the ellipse
        ax: Optional matplotlib axes to plot on
    """
    if ax is None:
        ax = plt.gca()
    
    # Extract 2D position
    pos_2d = splat.position[reduced_dims]
    
    # Extract 2D covariance
    cov_2d = np.array([
        [splat.covariance[reduced_dims[0], reduced_dims[0]], 
         splat.covariance[reduced_dims[0], reduced_dims[1]]],
        [splat.covariance[reduced_dims[1], reduced_dims[0]], 
         splat.covariance[reduced_dims[1], reduced_dims[1]]]
    ])
    
    # Calculate eigenvalues and eigenvectors for the ellipse
    eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
    
    # Sort eigenvalues in descending order
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    
    # Convert to positive values (safety check)
    eigenvals = np.abs(eigenvals)
    
    # Calculate angle and width/height for the ellipse
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * np.sqrt(5.991 * eigenvals)  # 95% confidence
    
    # Scale ellipse by amplitude
    amplitude_scale = min(max(splat.amplitude, 0.1), 2.0)
    width *= amplitude_scale
    height *= amplitude_scale
    
    # Create and add the ellipse
    ellipse = Ellipse(
        xy=(pos_2d[0], pos_2d[1]),
        width=width,
        height=height,
        angle=angle,
        alpha=alpha,
        color=color,
        fill=True
    )
    ax.add_patch(ellipse)
    
    # Add splat center
    ax.scatter(pos_2d[0], pos_2d[1], color=color, marker='x', s=50, zorder=10)


def visualize_mitosis_analysis(
    splat: Splat,
    tokens: np.ndarray,
    metrics_tracker: Any = None,
    output_dir: str = "hsa_visualizations",
    show: bool = True,
    save: bool = True,
    title: Optional[str] = None,
    reduced_dims: Optional[List[int]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Visualize the clustering analysis used to make mitosis decisions.
    
    Args:
        splat: The splat being evaluated for mitosis
        tokens: Token embeddings [sequence_length, embedding_dim]
        metrics_tracker: Optional metrics tracker
        output_dir: Directory to save visualizations
        show: Whether to display the visualization
        save: Whether to save the visualization to a file
        title: Optional title for the visualization
        reduced_dims: Optional list of dimensions to visualize (for high-dimensional data)
        
    Returns:
        Tuple of (mitosis_decision, analysis_data)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a mock metrics tracker if not provided
    if metrics_tracker is None:
        class MockMetricsTracker:
            def get_splat_metrics(self, splat_id):
                return {"activation": 1.0, "error_contribution": 1.0}
        
        metrics_tracker = MockMetricsTracker()
    
    # Get the dimensionality
    dim = splat.position.shape[0]
    
    # Select dimensions to plot
    if reduced_dims is None:
        if dim > 2:
            # Default to first two dimensions if none specified
            reduced_dims = [0, 1]
        else:
            reduced_dims = list(range(dim))
    
    # Extract tokens close to the splat
    closest_tokens = []
    distances = []
    
    for token in tokens:
        diff = token - splat.position
        distance = np.sqrt(diff @ splat.covariance_inverse @ diff)
        
        if distance < 2.0:  # Within 2 standard deviations
            closest_tokens.append(token)
            distances.append(distance)
    
    # Create the figure
    plt.figure(figsize=(10, 8))
    
    # Plot the splat
    plot_splat(splat, reduced_dims, color='blue')
    
    # If not enough close tokens, return early
    if len(closest_tokens) < 4:  # Need at least a few tokens for meaningful clustering
        if title is None:
            title = f"Splat {splat.id}: Insufficient tokens for mitosis"
        
        # Plot the few tokens we have
        if closest_tokens:
            closest_tokens_array = np.array(closest_tokens)
            plt.scatter(
                closest_tokens_array[:, reduced_dims[0]],
                closest_tokens_array[:, reduced_dims[1]],
                color='gray', alpha=0.7, label='Tokens'
            )
        
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        
        if save:
            timestamp = int(time.time())
            filename = f"splat_mitosis_analysis_{splat.id}_{timestamp}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return False, {
            "closest_tokens": len(closest_tokens),
            "reason": "insufficient_tokens"
        }
    
    # Convert to numpy array
    closest_tokens_array = np.array(closest_tokens)
    
    # Use KMeans to find 2 clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    
    try:
        kmeans.fit(closest_tokens_array)
        
        # Get cluster assignments and centers
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        # Check if both clusters have enough points
        counts = np.bincount(labels)
        min_cluster_size = 2
        
        if min(counts) < min_cluster_size:
            # Plot with cluster coloring but not enough points
            colors = ['red', 'green']
            
            for i, label in enumerate(np.unique(labels)):
                mask = labels == label
                plt.scatter(
                    closest_tokens_array[mask][:, reduced_dims[0]],
                    closest_tokens_array[mask][:, reduced_dims[1]],
                    color=colors[i], alpha=0.7,
                    label=f'Cluster {i+1} (n={np.sum(mask)})'
                )
            
            if title is None:
                title = f"Splat {splat.id}: Imbalanced clusters - No mitosis"
            
            plt.title(title)
            plt.grid(alpha=0.3)
            plt.legend()
            
            if save:
                timestamp = int(time.time())
                filename = f"splat_mitosis_analysis_{splat.id}_{timestamp}.png"
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return False, {
                "closest_tokens": len(closest_tokens),
                "cluster_counts": counts.tolist(),
                "reason": "imbalanced_clusters"
            }
        
        # Distance between cluster centers
        center_distance = np.linalg.norm(centers[0] - centers[1])
        
        # Average distance from points to their cluster center
        avg_distances = []
        for i in range(2):
            cluster_points = closest_tokens_array[labels == i]
            center = centers[i]
            
            # Compute average distance within this cluster
            avg_dist = np.mean([np.linalg.norm(p - center) for p in cluster_points])
            avg_distances.append(avg_dist)
        
        # Average intra-cluster distance
        avg_intra_distance = (avg_distances[0] + avg_distances[1]) / 2
        
        # Safety check
        if avg_intra_distance < 1e-10:
            avg_intra_distance = 1e-10
        
        # Compute separation ratio
        separation_ratio = center_distance / avg_intra_distance
        
        # Determine if mitosis should occur
        min_separation_ratio = 1.5
        mitosis_decision = separation_ratio >= min_separation_ratio
        
        # Plot with cluster coloring
        colors = ['red', 'green']
        
        for i, label in enumerate(np.unique(labels)):
            mask = labels == label
            plt.scatter(
                closest_tokens_array[mask][:, reduced_dims[0]],
                closest_tokens_array[mask][:, reduced_dims[1]],
                color=colors[i], alpha=0.7,
                label=f'Cluster {i+1} (n={np.sum(mask)})'
            )
        
        # Plot cluster centers
        plt.scatter(
            centers[:, reduced_dims[0]],
            centers[:, reduced_dims[1]],
            color='black', marker='*', s=200,
            label='Cluster centers'
        )
        
        # Add line between centers
        plt.plot(
            [centers[0, reduced_dims[0]], centers[1, reduced_dims[0]]],
            [centers[0, reduced_dims[1]], centers[1, reduced_dims[1]]],
            'k--', alpha=0.5
        )
        
        # Add text with metrics
        plt.text(
            0.05, 0.05,
            f"Center distance: {center_distance:.3f}\n"
            f"Avg intra-distance: {avg_intra_distance:.3f}\n"
            f"Separation ratio: {separation_ratio:.3f}\n"
            f"Decision: {'Mitosis' if mitosis_decision else 'No mitosis'}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        if title is None:
            decision_str = "Mitosis" if mitosis_decision else "No mitosis"
            title = f"Splat {splat.id}: Cluster Analysis - {decision_str}"
        
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        
        if save:
            timestamp = int(time.time())
            filename = f"splat_mitosis_analysis_{splat.id}_{timestamp}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return mitosis_decision, {
            "closest_tokens": len(closest_tokens),
            "cluster_counts": counts.tolist(),
            "center_distance": float(center_distance),
            "avg_intra_distance": float(avg_intra_distance),
            "separation_ratio": float(separation_ratio),
            "min_separation_ratio": min_separation_ratio,
            "reason": "decision_made"
        }
        
    except Exception as e:
        # Plot error case
        plt.scatter(
            closest_tokens_array[:, reduced_dims[0]],
            closest_tokens_array[:, reduced_dims[1]],
            color='gray', alpha=0.7,
            label='Tokens'
        )
        
        if title is None:
            title = f"Splat {splat.id}: Error in cluster analysis"
        
        plt.title(title)
        plt.text(
            0.05, 0.05,
            f"Error: {str(e)}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='red', alpha=0.3)
        )
        plt.grid(alpha=0.3)
        plt.legend()
        
        if save:
            timestamp = int(time.time())
            filename = f"splat_mitosis_analysis_{splat.id}_{timestamp}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return False, {
            "closest_tokens": len(closest_tokens),
            "error": str(e),
            "reason": "error"
        }