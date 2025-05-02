"""
Visualization tools for Hierarchical Splat Attention (HSA) adaptation.

This module provides visualization capabilities to help understand and debug
the adaptation process, showing how splats evolve over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import logging
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Configure logging
logger = logging.getLogger(__name__)


class SplatVisualizer:
    """Visualization tools for HSA splats and adaptation operations."""
    
    # Define color scheme for different levels
    LEVEL_COLORS = {
        "token": "#3498db",      # Blue
        "word": "#2ecc71",       # Green
        "phrase": "#e74c3c",     # Red
        "sentence": "#f39c12",   # Orange
        "paragraph": "#9b59b6",  # Purple
        "document": "#34495e"    # Dark Blue
    }
    
    # Default colors for levels not in the map
    DEFAULT_COLORS = ["#1abc9c", "#d35400", "#2c3e50", "#7f8c8d", "#8e44ad", "#16a085"]
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dim_reduction: Optional[str] = None):
        """Initialize the splat visualizer.
        
        Args:
            figsize: Figure size for visualizations
            dim_reduction: Dimensionality reduction method for high-dimensional splats
                           (None for no reduction, 'pca', 't-sne', etc.)
        """
        self.figsize = figsize
        self.dim_reduction = dim_reduction
        self.dim_reduction_model = None
        
        # Track visualization history for animations
        self.history: List[Dict[str, Any]] = []
    
    def _prepare_dim_reduction(self, data: np.ndarray) -> None:
        """Prepare dimensionality reduction if needed.
        
        Args:
            data: Data to prepare reduction for (splat positions)
        """
        if data.shape[1] <= 2:
            return  # No reduction needed
            
        if self.dim_reduction == 'pca':
            from sklearn.decomposition import PCA
            self.dim_reduction_model = PCA(n_components=2)
            self.dim_reduction_model.fit(data)
        elif self.dim_reduction == 't-sne':
            from sklearn.manifold import TSNE
            self.dim_reduction_model = TSNE(n_components=2, random_state=42)
            # T-SNE is fitted during transform
        
    def _reduce_dimensions(self, data: np.ndarray) -> np.ndarray:
        """Reduce dimensions of data for visualization.
        
        Args:
            data: High-dimensional data
            
        Returns:
            2D data for visualization
        """
        if data.shape[1] <= 2:
            if data.shape[1] == 1:
                # Add a zero second dimension for 1D data
                return np.hstack([data, np.zeros((data.shape[0], 1))])
            return data
            
        if self.dim_reduction_model is None:
            self._prepare_dim_reduction(data)
            
        if self.dim_reduction == 'pca':
            return self.dim_reduction_model.transform(data)
        elif self.dim_reduction == 't-sne':
            return self.dim_reduction_model.fit_transform(data)
        else:
            # Default: just use first two dimensions
            return data[:, :2]
    
    def _get_color_for_level(self, level: str) -> str:
        """Get color for a specific hierarchy level.
        
        Args:
            level: Level name
            
        Returns:
            Color string
        """
        if level in self.LEVEL_COLORS:
            return self.LEVEL_COLORS[level]
        
        # Use hash of level name to select a default color
        hash_val = hash(level) % len(self.DEFAULT_COLORS)
        return self.DEFAULT_COLORS[hash_val]
    
    def _draw_splat_ellipse(
        self,
        ax: plt.Axes,
        position: np.ndarray,
        covariance: np.ndarray,
        color: str,
        alpha: float = 0.3,
        label: Optional[str] = None
    ) -> Ellipse:
        """Draw an ellipse representing a splat.
        
        Args:
            ax: Matplotlib axes to draw on
            position: Splat position (2D)
            covariance: Covariance matrix (2x2)
            color: Color for the ellipse
            alpha: Transparency value
            label: Optional label for the ellipse
            
        Returns:
            The created Ellipse object
        """
        # Compute eigenvalues and eigenvectors for the covariance
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # Make sure eigenvalues are positive (they should be, but just in case)
        eigenvalues = np.maximum(eigenvalues, 0.001)
        
        # Calculate width and height of ellipse (2 std deviations)
        width, height = 2 * np.sqrt(eigenvalues)
        
        # Angle in degrees
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Create ellipse
        ellipse = Ellipse(
            xy=position,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=1.5
        )
        
        ax.add_patch(ellipse)
        
        # Draw center point
        ax.scatter(position[0], position[1], c=color, s=30, label=label)
        
        return ellipse
    
    def visualize_registry(
        self,
        registry,
        tokens=None,
        highlight_splats=None,
        title="Hierarchical Splat Attention Visualization",
        show_legend=True,
        save_path=None
    ):
        """Visualize all splats in a registry.
        
        Args:
            registry: SplatRegistry to visualize
            tokens: Optional token embeddings to show as background points
            highlight_splats: Optional set of splat IDs to highlight
            title: Title for the visualization
            show_legend: Whether to show the legend
            save_path: Path to save the visualization to (if None, just displays)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Set equal aspect ratio for better ellipse visualization
        ax.set_aspect('equal')
        
        # Get all splats
        all_splats = registry.get_all_splats()
        
        if not all_splats:
            ax.text(0.5, 0.5, "No splats to visualize", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                
            return fig
        
        # Extract positions for dimensionality reduction
        all_positions = np.stack([splat.position for splat in all_splats])
        
        # Add token positions if provided
        if tokens is not None and tokens.shape[0] > 0:
            all_positions = np.vstack([all_positions, tokens])
        
        # Visualize tokens if provided
        if tokens is not None and tokens.shape[0] > 0:
            # For 2D visualization, use first 2 dimensions
            token_positions_2d = tokens[:, :2] if tokens.shape[1] > 1 else np.hstack([tokens, np.zeros((tokens.shape[0], 1))])
            ax.scatter(
                token_positions_2d[:, 0],
                token_positions_2d[:, 1],
                c='gray',
                s=5,
                alpha=0.3,
                label='Tokens'
            )
        
        # Track legend entries
        legend_entries = {}
        
        # Draw splats
        for splat in all_splats:
            # For 2D visualization, use first 2 dimensions
            position_2d = splat.position[:2] if splat.dim > 1 else np.array([splat.position[0], 0])
            
            # Extract or create 2x2 covariance matrix
            if splat.dim == 1:
                # For 1D, create a diagonal 2x2 matrix
                cov_2d = np.diag([splat.covariance[0, 0], 0.1])
            elif splat.dim == 2:
                # Already 2D
                cov_2d = splat.covariance
            else:
                # Use upper-left 2x2 submatrix
                cov_2d = splat.covariance[:2, :2]
            
            # Get color for this level
            color = self._get_color_for_level(splat.level)
            
            # Check if this splat should be highlighted
            is_highlighted = highlight_splats and splat.id in highlight_splats
            
            # Draw with higher opacity and thicker border if highlighted
            if is_highlighted:
                edge_color = color
                face_alpha = 0.5
                edge_width = 2.0
            else:
                edge_color = color
                face_alpha = 0.3
                edge_width = 1.0
            
            # Add legend entry only once per level
            if splat.level not in legend_entries:
                label = f"Level: {splat.level}"
                legend_entries[splat.level] = True
            else:
                label = None
            
            # Calculate angle (in degrees)
            if cov_2d[0, 0] != cov_2d[1, 1]:
                angle = np.degrees(0.5 * np.arctan2(2 * cov_2d[0, 1], cov_2d[0, 0] - cov_2d[1, 1]))
            else:
                angle = 0
            
            # Draw ellipse
            ellipse = Ellipse(
                xy=position_2d,
                width=2 * np.sqrt(cov_2d[0, 0]),
                height=2 * np.sqrt(cov_2d[1, 1]),
                angle=angle,
                facecolor=color,
                alpha=face_alpha,
                edgecolor=edge_color,
                linewidth=edge_width,
                label=label
            )
            
            ax.add_patch(ellipse)
            
            # Add center point
            ax.scatter(
                position_2d[0], 
                position_2d[1], 
                c=color, 
                s=30, 
                zorder=5
            )
            
            # Add ID text if highlighted
            if is_highlighted:
                ax.text(
                    position_2d[0], 
                    position_2d[1] + 0.05, 
                    splat.id[-6:],  # Show last 6 chars of ID
                    ha='center', 
                    va='bottom', 
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                )
        
        # Add legend if requested
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper right', fontsize=10)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        # Set limits with some padding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        padding = 0.1
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]
        ax.set_xlim(xlim[0] - width * padding, xlim[1] + width * padding)
        ax.set_ylim(ylim[0] - height * padding, ylim[1] + height * padding)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_splat_attention_map(
        self,
        splat,
        tokens: np.ndarray,
        attention_map: np.ndarray,
        token_indices: Optional[List[int]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize the attention map for a specific splat.
        
        Args:
            splat: Splat to visualize attention for
            tokens: Token embeddings
            attention_map: Attention map for this splat
            token_indices: Optional list of token indices to visualize 
                          (if None, selects a few automatically)
            title: Title for the visualization
            save_path: Path to save the visualization to (if None, just displays)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        
        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # Set default title if not provided
        if title is None:
            title = f"Attention Map for Splat {splat.id} (Level: {splat.level})"
        
        # Select token indices if not provided
        if token_indices is None:
            # Find tokens with high attention
            token_attention = np.max(attention_map, axis=1)
            sorted_indices = np.argsort(token_attention)[::-1]
            
            # Take top 20 tokens
            token_indices = sorted_indices[:min(20, len(sorted_indices))]
        
        # Reduce dimensions of token positions
        token_positions_2d = self._reduce_dimensions(tokens[token_indices])
        
        # Create meshgrid for attention surface
        x = token_positions_2d[:, 0]
        y = token_positions_2d[:, 1]
        
        # Use attention values
        attention_values = np.zeros((len(token_indices), len(token_indices)))
        for i, src_idx in enumerate(token_indices):
            for j, tgt_idx in enumerate(token_indices):
                attention_values[i, j] = attention_map[src_idx, tgt_idx]
        
        # Create X, Y meshgrid
        X, Y = np.meshgrid(x, y)
        
        # Plot surface
        surf = ax.plot_surface(
            X, Y, attention_values,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
            alpha=0.7
        )
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Attention Value')
        
        # Plot token positions
        ax.scatter(
            token_positions_2d[:, 0],
            token_positions_2d[:, 1],
            np.max(attention_values, axis=1),
            c='black',
            s=50,
            zorder=10
        )
        
        # Add token indices
        for i, idx in enumerate(token_indices):
            ax.text(
                token_positions_2d[i, 0],
                token_positions_2d[i, 1],
                np.max(attention_values, axis=1)[i] + 0.05,
                str(idx),
                size=8,
                zorder=10
            )
        
        # Set labels and title
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Attention Value')
        ax.set_title(title)
        
        # Adjust view
        ax.view_init(elev=30, azim=120)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_3d_splats(
        self,
        registry,
        tokens: Optional[np.ndarray] = None,
        highlight_splats: Optional[Set[str]] = None,
        title: str = "3D Visualization of Hierarchical Splat Attention",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize splats in 3D space (for 3D embeddings).
        
        Args:
            registry: SplatRegistry to visualize
            tokens: Optional token embeddings to show as background points
            highlight_splats: Optional set of splat IDs to highlight
            title: Title for the visualization
            save_path: Path to save the visualization to (if None, just displays)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        
        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # Get all splats
        all_splats = registry.get_all_splats()
        
        if not all_splats:
            ax.text(0.5, 0.5, 0.5, "No splats to visualize", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                
            return fig
        
        # Check dimension and reduce if needed
        dim = registry.embedding_dim
        
        if dim < 3:
            logger.warning(f"Embedding dimension {dim} is less than 3. Adding zero dimensions.")
            reduce_dim = False
            # Function to add zero dimensions
            def process_position(pos):
                if dim == 1:
                    return np.array([pos[0], 0, 0])
                else:  # dim == 2
                    return np.array([pos[0], pos[1], 0])
        elif dim == 3:
            reduce_dim = False
            # No reduction needed
            def process_position(pos):
                return pos
        else:  # dim > 3
            reduce_dim = True
            # Use first 3 principal components
            from sklearn.decomposition import PCA
            all_positions = np.stack([splat.position for splat in all_splats])
            if tokens is not None and tokens.shape[0] > 0:
                all_positions = np.vstack([all_positions, tokens])
            
            pca = PCA(n_components=3)
            pca.fit(all_positions)
            
            def process_position(pos):
                return pca.transform(pos.reshape(1, -1))[0]
        
        # Track legend entries
        legend_entries = {}
        
        # Draw splats
        for splat in all_splats:
            # Process position to 3D
            position_3d = process_position(splat.position)
            
            # Get color for this level
            color = self._get_color_for_level(splat.level)
            
            # Check if this splat should be highlighted
            is_highlighted = highlight_splats and splat.id in highlight_splats
            
            # Draw with higher opacity and larger size if highlighted
            size = 100 if is_highlighted else 50
            alpha = 0.8 if is_highlighted else 0.5
            
            # Add legend entry only once per level
            if splat.level not in legend_entries:
                label = f"Level: {splat.level}"
                legend_entries[splat.level] = True
            else:
                label = None
            
            # Draw splat as a sphere
            ax.scatter(
                position_3d[0],
                position_3d[1],
                position_3d[2],
                c=[color],
                s=size,
                alpha=alpha,
                label=label
            )
            
            # Add ID text if highlighted
            if is_highlighted:
                ax.text(
                    position_3d[0],
                    position_3d[1],
                    position_3d[2] + 0.1,
                    splat.id[-6:],  # Show last 6 chars of ID
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )
            
            # Draw ellipsoid for covariance
            if not reduce_dim and splat.dim >= 3:
                # TODO: Add code to draw 3D ellipsoid based on covariance
                # This is complex and would require additional functions
                pass
        
        # Visualize tokens if provided
        if tokens is not None and tokens.shape[0] > 0:
            if reduce_dim:
                token_positions_3d = pca.transform(tokens)
            else:
                token_positions_3d = np.array([process_position(token) for token in tokens])
            
            ax.scatter(
                token_positions_3d[:, 0],
                token_positions_3d[:, 1],
                token_positions_3d[:, 2],
                c='gray',
                s=5,
                alpha=0.3,
                label='Tokens'
            )
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        
        # Adjust view
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_hierarchy_levels(
        self,
        registry,
        tokens: Optional[np.ndarray] = None,
        title: str = "Hierarchical Levels of Splat Attention",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize splats grouped by hierarchical level.
        
        Args:
            registry: SplatRegistry to visualize
            tokens: Optional token embeddings to show as background points
            title: Title for the visualization
            save_path: Path to save the visualization to (if None, just displays)
            
        Returns:
            Matplotlib figure object
        """
        # Get hierarchy levels
        levels = registry.hierarchy.levels
        n_levels = len(levels)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 6), sharey=True)
        
        # Handle case of single level
        if n_levels == 1:
            axes = [axes]
        
        # Get all splats
        all_splats = registry.get_all_splats()
        
        if not all_splats:
            for ax in axes:
                ax.text(0.5, 0.5, "No splats to visualize", 
                        ha='center', va='center', transform=ax.transAxes)
            fig.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                
            return fig
        
        # Extract positions for dimensionality reduction
        all_positions = np.stack([splat.position for splat in all_splats])
        
        # Add token positions if provided
        if tokens is not None and tokens.shape[0] > 0:
            all_positions = np.vstack([all_positions, tokens])
        
        # Prepare dimensionality reduction
        self._prepare_dim_reduction(all_positions)
        
        # Visualize each level in a separate subplot
        for i, level in enumerate(levels):
            ax = axes[i]
            
            # Get splats at this level
            level_splats = list(registry.get_splats_at_level(level))
            
            # Skip empty levels
            if not level_splats:
                ax.text(0.5, 0.5, f"No splats at level: {level}", 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Level: {level}")
                continue
            
            # Set equal aspect ratio for better ellipse visualization
            ax.set_aspect('equal')
            
            # Visualize tokens if provided
            if tokens is not None and tokens.shape[0] > 0:
                token_positions_2d = self._reduce_dimensions(tokens)
                ax.scatter(
                    token_positions_2d[:, 0],
                    token_positions_2d[:, 1],
                    c='gray',
                    s=5,
                    alpha=0.3,
                    label='Tokens'
                )
            
            # Get color for this level
            color = self._get_color_for_level(level)
            
            # Draw splats
            for splat in level_splats:
                # Reduce position dimensionality if needed
                position_2d = self._reduce_dimensions(splat.position.reshape(1, -1))[0]
                
                # For 2D covariance, we need to project the covariance matrix
                if splat.dim > 2 and self.dim_reduction_model is not None:
                    if self.dim_reduction == 'pca':
                        # For PCA, transform covariance using the components
                        components = self.dim_reduction_model.components_
                        cov_2d = components @ splat.covariance @ components.T
                    else:
                        # For other methods, just use identity scaled by trace
                        cov_2d = np.eye(2) * np.trace(splat.covariance) / splat.dim
                elif splat.dim == 1:
                    # For 1D, create a diagonal 2x2 matrix
                    cov_2d = np.diag([splat.covariance[0, 0], 0.1])
                else:
                    # Already 2D
                    cov_2d = splat.covariance
                
                # Draw ellipse
                ellipse = Ellipse(
                    xy=position_2d,
                    width=2 * np.sqrt(cov_2d[0, 0]),
                    height=2 * np.sqrt(cov_2d[1, 1]),
                    angle=np.degrees(0.5 * np.arctan2(2 * cov_2d[0, 1], cov_2d[0, 0] - cov_2d[1, 1])) if cov_2d[0, 0] != cov_2d[1, 1] else 0,
                    facecolor=color,
                    alpha=0.3,
                    edgecolor=color,
                    linewidth=1.5
                )
                
                ax.add_patch(ellipse)
                
                # Add center point
                ax.scatter(
                    position_2d[0], 
                    position_2d[1], 
                    c=color, 
                    s=30, 
                    zorder=5
                )
            
            # Set title and labels
            ax.set_title(f"Level: {level}")
            if i == 0:
                ax.set_ylabel('Dimension 2')
            ax.set_xlabel('Dimension 1')
            
            # Add count in the corner
            ax.text(
                0.05, 0.95, 
                f"Count: {len(level_splats)}", 
                transform=ax.transAxes,
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
            )
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_adaptation_metrics(
        self,
        registry,
        metrics,
        highlight_top=5,
        title="Adaptation Metrics Visualization",
        save_path=None
    ):
        """Visualize adaptation metrics for splats.
        
        Args:
            registry: SplatRegistry to visualize
            metrics: Dictionary mapping splat IDs to metrics
            highlight_top: Number of top splats to highlight for each metric
            title: Title for the visualization
            save_path: Path to save the visualization to (if None, just displays)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure with grid of subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Define metrics to visualize
        metric_names = [
            "activation_mean",
            "activation_trend",
            "information_contribution",
            "variance"
        ]
        
        n_metrics = len(metric_names)
        
        # Create grid of subplots
        grid = plt.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
        
        # Get all splats
        all_splats = registry.get_all_splats()
        splat_ids = [splat.id for splat in all_splats]
        
        # Filter metrics to only include existing splats
        filtered_metrics = {
            splat_id: metric_values 
            for splat_id, metric_values in metrics.items() 
            if splat_id in splat_ids
        }
        
        # Create subplots for each metric
        for i, metric_name in enumerate(metric_names):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(grid[row, col])
            
            # Extract metric values for all splats
            values = []
            ids = []
            
            for splat_id, metric_values in filtered_metrics.items():
                if hasattr(metric_values, metric_name):
                    values.append(getattr(metric_values, metric_name))
                    ids.append(splat_id)
            
            # Skip if no values
            if not values:
                ax.text(0.5, 0.5, f"No data for metric: {metric_name}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric_name.replace('_', ' ').title())
                continue
            
            # Sort splats by metric value
            sorted_indices = np.argsort(values)
            if metric_name != "variance":
                # For most metrics, higher is more interesting
                sorted_indices = sorted_indices[::-1]
            
            # Highlight top splats
            top_indices = sorted_indices[:highlight_top]
            
            # Create colors based on levels
            colors = []
            
            for splat_id in ids:
                splat = registry.safe_get_splat(splat_id)
                if splat:
                    colors.append(self._get_color_for_level(splat.level))
                else:
                    colors.append('#cccccc')  # Gray for missing splats
            
            # Create bar chart
            bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7)
            
            # Highlight top bars
            for j in top_indices:
                if j < len(bars):
                    bars[j].set_edgecolor('black')
                    bars[j].set_linewidth(2)
                    bars[j].set_alpha(1.0)
            
            # Add splat IDs for top values
            for j in top_indices:
                if j < len(bars):
                    bar = bars[j]
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        ids[j][-6:],  # Show last 6 chars of ID
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation=90
                    )
            
            # Set title and labels
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_xlabel('Splat Index')
            ax.set_ylabel('Value')
            
            # Set X-axis ticks
            if len(values) > 10:
                ax.set_xticks([])
            else:
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels([f"{i}" for i in range(len(values))], rotation=90)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_adaptation_operation(
        self,
        result,
        registry,
        tokens: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize a specific adaptation operation.
        
        Args:
            result: AdaptationResult from the operation
            registry: SplatRegistry after the operation
            tokens: Optional token embeddings to show as background points
            save_path: Path to save the visualization to (if None, just displays)
            
        Returns:
            Matplotlib figure object
        """
        adaptation_type = result.adaptation_type
        
        # Set title based on adaptation type
        if adaptation_type.name == "MITOSIS":
            title = f"Mitosis Operation: {result.original_splat_id} → {', '.join(result.new_splat_ids)}"
            highlight_splats = set(result.new_splat_ids)
        elif adaptation_type.name == "BIRTH":
            title = f"Birth Operation: Created {', '.join(result.new_splat_ids)}"
            highlight_splats = set(result.new_splat_ids)
        elif adaptation_type.name == "DEATH":
            title = f"Death Operation: Removed {result.original_splat_id}"
            highlight_splats = set()  # Nothing to highlight
        elif adaptation_type.name == "MERGE":
            title = f"Merge Operation: {result.original_splat_id} + {result.removed_splat_ids[1] if len(result.removed_splat_ids) > 1 else '?'} → {', '.join(result.new_splat_ids)}"
            highlight_splats = set(result.new_splat_ids)
        elif adaptation_type.name == "ADJUST":
            title = f"Adjust Operation: Updated {result.original_splat_id}"
            highlight_splats = {result.original_splat_id}
        else:
            title = f"Unknown Operation: {adaptation_type}"
            highlight_splats = set()
        
        # Create visualization
        fig = self.visualize_registry(
            registry=registry,
            tokens=tokens,
            highlight_splats=highlight_splats,
            title=title,
            show_legend=True,
            save_path=save_path
        )
        
        return fig
    
    def visualize_adaptation_history(
        self,
        history,
        registry,
        tokens: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize a history of adaptation operations.
        
        Args:
            history: List of AdaptationResult objects
            registry: SplatRegistry after all operations
            tokens: Optional token embeddings to show as background points
            save_path: Path to save the visualization to (if None, just displays)
            
        Returns:
            Matplotlib figure object
        """
        # Count operation types
        operation_counts = {}
        for result in history:
            op_type = result.adaptation_type.name
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        
        # Format counts for title
        counts_str = ", ".join([f"{count} {op_type}" for op_type, count in operation_counts.items()])
        
        title = f"Adaptation History: {counts_str}"
        
        # Highlight the most recent operation's splats
        highlight_splats = set()
        if history:
            latest = history[-1]
            highlight_splats.update(latest.new_splat_ids)
        
        # Create visualization
        fig = self.visualize_registry(
            registry=registry,
            tokens=tokens,
            highlight_splats=highlight_splats,
            title=title,
            show_legend=True,
            save_path=save_path
        )
        
        return fig
    
    def create_adaptation_animation(
        self,
        registry_states,
        adaptation_results,
        tokens: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        fps: int = 2
    ):
        """Create an animation of adaptation operations.
        
        Args:
            registry_states: List of SplatRegistry states at each step
            adaptation_results: List of AdaptationResult objects for each step
            tokens: Optional token embeddings to show as background points
            save_path: Path to save the animation to (if None, just displays)
            fps: Frames per second for the animation
            
        Returns:
            Matplotlib animation object
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect('equal')
        
        # Collect all splat positions across all states
        all_positions = []
        for registry in registry_states:
            splats = registry.get_all_splats()
            if splats:
                all_positions.extend([splat.position for splat in splats])
        
        # Add tokens if provided
        if tokens is not None and tokens.shape[0] > 0:
            all_positions.extend([token for token in tokens])
        
        # Convert to array if we have positions
        if all_positions:
            all_positions = np.stack(all_positions)
            # Prepare dimensionality reduction
            self._prepare_dim_reduction(all_positions)
        
        # Function to update the plot for each frame
        def update(frame):
            ax.clear()
            
            # Get registry and result for this frame
            registry = registry_states[frame]
            result = adaptation_results[frame] if frame < len(adaptation_results) else None
            
            # Set title based on adaptation type
            if result:
                if result.adaptation_type.name == "MITOSIS":
                    title = f"Frame {frame}: Mitosis - {result.original_splat_id} → {', '.join(result.new_splat_ids)}"
                    highlight_splats = set(result.new_splat_ids)
                elif result.adaptation_type.name == "BIRTH":
                    title = f"Frame {frame}: Birth - Created {', '.join(result.new_splat_ids)}"
                    highlight_splats = set(result.new_splat_ids)
                elif result.adaptation_type.name == "DEATH":
                    title = f"Frame {frame}: Death - Removed {result.original_splat_id}"
                    highlight_splats = set()
                elif result.adaptation_type.name == "MERGE":
                    title = f"Frame {frame}: Merge - Combined splats into {', '.join(result.new_splat_ids)}"
                    highlight_splats = set(result.new_splat_ids)
                elif result.adaptation_type.name == "ADJUST":
                    title = f"Frame {frame}: Adjust - Updated {result.original_splat_id}"
                    highlight_splats = {result.original_splat_id}
                else:
                    title = f"Frame {frame}: Unknown Operation"
                    highlight_splats = set()
            else:
                title = f"Frame {frame}: Initial State"
                highlight_splats = set()
            
            # Visualize tokens if provided
            if tokens is not None and tokens.shape[0] > 0:
                token_positions_2d = self._reduce_dimensions(tokens)
                ax.scatter(
                    token_positions_2d[:, 0],
                    token_positions_2d[:, 1],
                    c='gray',
                    s=5,
                    alpha=0.3,
                    label='Tokens'
                )
            
            # Track legend entries
            legend_entries = {}
            
            # Draw splats
            for splat in registry.get_all_splats():
                # Reduce position dimensionality if needed
                position_2d = self._reduce_dimensions(splat.position.reshape(1, -1))[0]
                
                # For 2D covariance, we need to project the covariance matrix
                if splat.dim > 2 and self.dim_reduction_model is not None:
                    if self.dim_reduction == 'pca':
                        # For PCA, transform covariance using the components
                        components = self.dim_reduction_model.components_
                        cov_2d = components @ splat.covariance @ components.T
                    else:
                        # For other methods, just use identity scaled by trace
                        cov_2d = np.eye(2) * np.trace(splat.covariance) / splat.dim
                elif splat.dim == 1:
                    # For 1D, create a diagonal 2x2 matrix
                    cov_2d = np.diag([splat.covariance[0, 0], 0.1])
                else:
                    # Already 2D
                    cov_2d = splat.covariance
                
                # Get color for this level
                color = self._get_color_for_level(splat.level)
                
                # Check if this splat should be highlighted
                is_highlighted = splat.id in highlight_splats
                
                # Draw with higher opacity and thicker border if highlighted
                if is_highlighted:
                    edge_color = color
                    face_alpha = 0.5
                    edge_width = 2.0
                else:
                    edge_color = color
                    face_alpha = 0.3
                    edge_width = 1.0
                
                # Add legend entry only once per level
                if splat.level not in legend_entries:
                    label = f"Level: {splat.level}"
                    legend_entries[splat.level] = True
                else:
                    label = None
                
                # Draw ellipse
                ellipse = Ellipse(
                    xy=position_2d,
                    width=2 * np.sqrt(cov_2d[0, 0]),
                    height=2 * np.sqrt(cov_2d[1, 1]),
                    angle=np.degrees(0.5 * np.arctan2(2 * cov_2d[0, 1], cov_2d[0, 0] - cov_2d[1, 1])) if cov_2d[0, 0] != cov_2d[1, 1] else 0,
                    facecolor=color,
                    alpha=face_alpha,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    label=label
                )
                
                ax.add_patch(ellipse)
                
                # Add center point
                ax.scatter(
                    position_2d[0], 
                    position_2d[1], 
                    c=color, 
                    s=30, 
                    zorder=5
                )
                
                # Add ID text if highlighted
                if is_highlighted:
                    ax.text(
                        position_2d[0], 
                        position_2d[1] + 0.05, 
                        splat.id[-6:],  # Show last 6 chars of ID
                        ha='center', 
                        va='bottom', 
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                    )
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper right', fontsize=10)
            
            # Set title and labels
            ax.set_title(title)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            
            # Set fixed limits if possible
            if hasattr(self, 'animation_limits'):
                ax.set_xlim(self.animation_limits[0])
                ax.set_ylim(self.animation_limits[1])
            else:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                if frame == 0:
                    # Store initial limits
                    self.animation_limits = (xlim, ylim)
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(registry_states), interval=1000/fps, blit=False
        )
        
        # Save if requested
        if save_path:
            ani.save(save_path, writer='pillow', fps=fps, dpi=100)
        
        return ani
    
    def visualize_attention_flow(
        self,
        registry,
        attention_matrix,
        tokens,
        token_indices=None,
        title="Attention Flow through Splats",
        save_path=None
    ):
        """Visualize attention flow through splats for specific token pairs.
        
        Args:
            registry: SplatRegistry to visualize
            attention_matrix: Full attention matrix
            tokens: Token embeddings
            token_indices: Optional list of token indices to visualize 
                          (if None, selects a few automatically)
            title: Title for the visualization
            save_path: Path to save the visualization to (if None, just displays)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Set equal aspect ratio for better ellipse visualization
        ax.set_aspect('equal')
        
        # Get all splats
        all_splats = registry.get_all_splats()
        
        if not all_splats:
            ax.text(0.5, 0.5, "No splats to visualize", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                
            return fig
        
        # For 2D visualization, use first 2 dimensions for all positions
        if tokens.shape[1] > 2:
            token_positions_2d = tokens[:, :2]
        else:
            token_positions_2d = tokens
        
        # Select token indices if not provided
        if token_indices is None:
            # Find token pairs with high attention
            if attention_matrix.shape[0] > 10:
                # Take diagonal for reference
                diag_mean = np.mean(np.diag(attention_matrix))
                
                # Find off-diagonal elements with high attention
                off_diag = attention_matrix.copy()
                np.fill_diagonal(off_diag, 0)
                high_attention = off_diag > diag_mean
                
                # Get indices of high attention pairs
                src_idx, tgt_idx = np.where(high_attention)
                
                # Take a few pairs
                if len(src_idx) > 0:
                    # Sort by attention value
                    pairs = [(src_idx[i], tgt_idx[i], off_diag[src_idx[i], tgt_idx[i]]) 
                            for i in range(len(src_idx))]
                    pairs.sort(key=lambda x: x[2], reverse=True)
                    
                    # Take top pairs
                    token_indices = []
                    for i in range(min(5, len(pairs))):
                        src, tgt, _ = pairs[i]
                        if src not in token_indices:
                            token_indices.append(src)
                        if tgt not in token_indices:
                            token_indices.append(tgt)
                    
                    # Limit to 10 tokens
                    token_indices = token_indices[:10]
                else:
                    # Fallback: just take first few tokens
                    token_indices = list(range(min(10, tokens.shape[0])))
            else:
                # If few tokens, use all
                token_indices = list(range(tokens.shape[0]))
        
        # Draw splats
        for splat in all_splats:
            # Use first 2 dimensions
            position_2d = splat.position[:2] if splat.dim > 1 else np.array([splat.position[0], 0])
            
            # Extract or create 2x2 covariance matrix
            if splat.dim == 1:
                cov_2d = np.diag([splat.covariance[0, 0], 0.1])
            elif splat.dim == 2:
                cov_2d = splat.covariance
            else:
                cov_2d = splat.covariance[:2, :2]
            
            # Get color for this level
            color = self._get_color_for_level(splat.level)
            
            # Draw ellipse
            width = 2 * np.sqrt(cov_2d[0, 0])
            height = 2 * np.sqrt(cov_2d[1, 1])
            
            # Calculate angle - carefully handling potential division by zero
            if cov_2d[0, 0] != cov_2d[1, 1]:
                angle = np.degrees(0.5 * np.arctan2(2 * cov_2d[0, 1], cov_2d[0, 0] - cov_2d[1, 1]))
            else:
                angle = 0
            
            ellipse = Ellipse(
                xy=position_2d,
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                alpha=0.2,
                edgecolor=color,
                linewidth=1.5
            )
            
            ax.add_patch(ellipse)
            
            # Add center point
            ax.scatter(
                position_2d[0], 
                position_2d[1], 
                c=color, 
                s=30, 
                zorder=5
            )
        
        # Draw tokens with different colors based on their indices
        colors = plt.cm.tab10(np.linspace(0, 1, len(token_indices)))
        
        for i, idx in enumerate(token_indices):
            ax.scatter(
                token_positions_2d[idx, 0],
                token_positions_2d[idx, 1],
                c=[colors[i]],
                s=100,
                marker='o',
                edgecolors='black',
                linewidths=1.5,
                zorder=10,
                label=f"Token {idx}"
            )
            
            # Add token index
            ax.text(
                token_positions_2d[idx, 0],
                token_positions_2d[idx, 1] + 0.1,
                str(idx),
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
            )
        
        # Draw attention arrows between tokens
        max_attention = np.max(attention_matrix)
        min_line_width = 0.5
        max_line_width = 3.0
        
        for i, src_idx in enumerate(token_indices):
            for j, tgt_idx in enumerate(token_indices):
                if src_idx != tgt_idx:
                    # Get attention value
                    att_value = attention_matrix[src_idx, tgt_idx]
                    
                    # Skip if attention is too low
                    if att_value < 0.1 * max_attention:
                        continue
                    
                    # Calculate line width based on attention
                    width = min_line_width + (max_line_width - min_line_width) * (att_value / max_attention)
                    
                    # Draw arrow
                    ax.annotate(
                        "",
                        xy=(token_positions_2d[tgt_idx, 0], token_positions_2d[tgt_idx, 1]),
                        xytext=(token_positions_2d[src_idx, 0], token_positions_2d[src_idx, 1]),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=colors[i],
                            linewidth=width,
                            alpha=0.6,
                            connectionstyle="arc3,rad=0.2"
                        )
                    )
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
