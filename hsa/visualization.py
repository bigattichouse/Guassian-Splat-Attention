"""
HSA Visualization Module

This module provides visualization tools for Hierarchical Splat Attention (HSA).
It creates visual representations of:
- Attention patterns
- Splat distributions
- Hierarchical relationships
- Adaptation events
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import time
import warnings
from sklearn.decomposition import PCA

class HSAVisualizer:
    """
    Visualizer for Hierarchical Splat Attention.
    
    This class provides methods to visualize various aspects of HSA:
    - Attention heatmaps
    - Splat distributions
    - Hierarchical relationships
    - Adaptation events
    """
    
    def __init__(self, output_dir: str = "hsa_visualizations"):
        """
        Initialize the HSA visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up color scheme for different hierarchy levels
        self.level_colors = {
            "Token": "#3498db",     # Blue
            "Phrase": "#2ecc71",    # Green
            "Section": "#e67e22",   # Orange
            "Document": "#e74c3c",  # Red
        }
        
        # Default color for levels not in the predefined mapping
        self.default_color = "#9b59b6"  # Purple
        
        # Track adaptation events for visualization
        self.adaptation_history = []
    
    def safe_legend(self, ax=None, min_artists=1, **kwargs):
        """
        Safely add a legend only if there are labeled artists.
        
        Args:
            ax: Matplotlib axes to add legend to (uses current axes if None)
            min_artists: Minimum number of labeled artists required to add legend
            **kwargs: Additional arguments to pass to legend()
        """
        if ax is None:
            ax = plt.gca()
        
        # Check if there are any labeled artists
        handles, labels = ax.get_legend_handles_labels()
        
        if len(handles) >= min_artists:
            ax.legend(**kwargs)
    
    def safe_tight_layout(self, fig=None, warn=False, **kwargs):
        """
        Safely apply tight_layout, suppressing warnings if desired.
        
        Args:
            fig: Figure to apply tight_layout to (uses current figure if None)
            warn: Whether to show warnings (False to suppress)
            **kwargs: Additional arguments to pass to tight_layout()
        """
        if fig is None:
            fig = plt.gcf()
        
        if not warn:
            # Temporarily suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.tight_layout(**kwargs)
        else:
            fig.tight_layout(**kwargs)
    
    def safe_ylim(self, ymin=None, ymax=None, min_range=0.1, ax=None):
        """
        Safely set y-axis limits, ensuring they're different by at least min_range.
        
        Args:
            ymin: Lower y limit
            ymax: Upper y limit
            min_range: Minimum range between limits
            ax: Matplotlib axes to apply to (uses current axes if None)
        """
        if ax is None:
            ax = plt.gca()
            
        if ymin is not None and ymax is not None:
            # Ensure there's a minimum range between limits
            if abs(ymax - ymin) < min_range:
                # Calculate mean and expand in both directions
                mean = (ymax + ymin) / 2
                ymin = mean - min_range / 2
                ymax = mean + min_range / 2
        
        ax.set_ylim(bottom=ymin, top=ymax)
    
    def visualize_attention_matrix(
        self, 
        attention_matrix: np.ndarray,
        tokens: Optional[List[str]] = None,
        title: str = "HSA Attention Matrix",
        show: bool = True,
        save: bool = True
    ) -> str:
        """
        Visualize an attention matrix as a heatmap.
        
        Args:
            attention_matrix: The attention matrix to visualize
            tokens: Optional list of token strings for axis labels
            title: Title for the visualization
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            
        Returns:
            Path to saved visualization if save=True, empty string otherwise
        """
        plt.figure(figsize=(10, 8))
        
        # Plot the heatmap
        plt.imshow(attention_matrix, cmap='viridis')
        plt.colorbar(label='Attention Score')
        
        # Add title and labels
        plt.title(title)
        plt.xlabel('Target Tokens')
        plt.ylabel('Source Tokens')
        
        # Add token labels if provided (limited to 30 tokens for readability)
        if tokens is not None:
            max_tokens = min(30, len(tokens))
            step = max(1, len(tokens) // max_tokens)
            
            token_indices = list(range(0, len(tokens), step))
            token_labels = [tokens[i] for i in token_indices]
            
            plt.xticks(token_indices, token_labels, rotation=90)
            plt.yticks(token_indices, token_labels)
        
        # Use safe tight_layout to avoid warnings
        self.safe_tight_layout()
        
        # Save the visualization
        file_path = ""
        if save:
            timestamp = int(time.time())
            file_path = os.path.join(self.output_dir, f"attention_matrix_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        # Show the visualization
        if show:
            plt.show()
        else:
            plt.close()
            
        return file_path
    
    def visualize_splat_distribution(
        self, 
        splat_registry: Any,
        tokens: Optional[np.ndarray] = None,
        method: str = "pca",
        title: str = "HSA Splat Distribution",
        show: bool = True,
        save: bool = True
    ) -> str:
        """
        Visualize the distribution of splats in the embedding space.
        
        Args:
            splat_registry: Registry containing all splats
            tokens: Optional token embeddings to show alongside splats
            method: Dimensionality reduction method ('pca' or 'tsne')
            title: Title for the visualization
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            
        Returns:
            Path to saved visualization if save=True, empty string otherwise
        """
        # Collect all splats
        splats = list(splat_registry.splats.values())
        
        # Handle empty registry
        if not splats:
            plt.figure(figsize=(10, 8))
            plt.title(f"{title} (No splats available)")
            plt.text(0.5, 0.5, "No splats available to visualize", 
                     ha='center', va='center', fontsize=14)
            
            # Save the visualization
            file_path = ""
            if save:
                timestamp = int(time.time())
                file_path = os.path.join(self.output_dir, f"splat_distribution_{timestamp}.png")
                plt.savefig(file_path, dpi=300)
                
            # Show the visualization
            if show:
                plt.show()
            else:
                plt.close()
                
            return file_path
        
        # Extract positions and create level mapping
        positions = np.array([splat.position for splat in splats])
        levels = [splat.level for splat in splats]
        
        # Get dimensions of positions
        dim = positions.shape[1]
        
        # Perform dimensionality reduction if needed
        if dim > 2:
            if method == "pca":
                # Use PCA for dimensionality reduction
                pca = PCA(n_components=2)
                positions_2d = pca.fit_transform(positions)
                
                # Transform tokens if provided
                tokens_2d = None
                if tokens is not None:
                    tokens_2d = pca.transform(tokens)
            else:
                # Fallback to PCA if method not recognized
                pca = PCA(n_components=2)
                positions_2d = pca.fit_transform(positions)
                
                # Transform tokens if provided
                tokens_2d = None
                if tokens is not None:
                    tokens_2d = pca.transform(tokens)
        else:
            # Already 2D or less
            positions_2d = positions
            tokens_2d = tokens
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot tokens if provided
        if tokens_2d is not None:
            plt.scatter(
                tokens_2d[:, 0], 
                tokens_2d[:, 1], 
                alpha=0.2, 
                color='gray', 
                marker='.', 
                label='Tokens'
            )
        
        # Track if we've added any labeled elements for legend
        has_labels = False
        
        # Plot splats by level
        for level in set(levels):
            level_indices = [i for i, l in enumerate(levels) if l == level]
            level_positions = positions_2d[level_indices]
            
            # Get color for this level
            color = self.level_colors.get(level, self.default_color)
            
            # Get corresponding splats for this level
            level_splats = [splats[i] for i in level_indices]
            
            # Plot points
            plt.scatter(
                level_positions[:, 0],
                level_positions[:, 1],
                alpha=0.7,
                color=color,
                label=f"{level} Level"
            )
            has_labels = True
            
            # Add ellipses for covariance (reduced to 2D)
            for i, splat in enumerate(level_splats):
                # Project covariance matrix to 2D
                if dim > 2 and method == "pca":
                    cov_2d = pca.components_ @ splat.covariance @ pca.components_.T
                else:
                    # Use a subset of the covariance matrix if already low-dimensional
                    cov_2d = splat.covariance[:2, :2] if splat.covariance.shape[0] > 1 else np.eye(2)
                
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
                    xy=(level_positions[i, 0], level_positions[i, 1]),
                    width=width,
                    height=height,
                    angle=angle,
                    alpha=0.2,
                    color=color
                )
                plt.gca().add_patch(ellipse)
        
        # Add title and legend (only if we have labeled elements)
        plt.title(title)
        if has_labels:
            self.safe_legend()
        plt.grid(alpha=0.3)
        
        # Make axes equal to avoid distortion
        plt.axis('equal')
        self.safe_tight_layout()
        
        # Save the visualization
        file_path = ""
        if save:
            timestamp = int(time.time())
            file_path = os.path.join(self.output_dir, f"splat_distribution_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        # Show the visualization
        if show:
            plt.show()
        else:
            plt.close()
            
        return file_path
    
    def visualize_hierarchy(
        self, 
        splat_registry: Any,
        title: str = "HSA Hierarchy Visualization",
        show: bool = True,
        save: bool = True
    ) -> str:
        """
        Visualize the hierarchical relationships between splats.
        
        Args:
            splat_registry: Registry containing all splats
            title: Title for the visualization
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            
        Returns:
            Path to saved visualization if save=True, empty string otherwise
        """
        # Get hierarchy levels
        hierarchy = splat_registry.hierarchy
        levels = hierarchy.levels
        
        # Create a nested structure for plotting
        plt.figure(figsize=(12, 8))
        
        # Handle case with no splats
        if len(splat_registry.splats) == 0:
            plt.title(f"{title} (No splats available)")
            plt.text(0.5, 0.5, "No splats available to visualize", 
                    ha='center', va='center', fontsize=14)
            
            # Save the visualization
            file_path = ""
            if save:
                timestamp = int(time.time())
                file_path = os.path.join(self.output_dir, f"hierarchy_{timestamp}.png")
                plt.savefig(file_path, dpi=300)
                
            # Show the visualization
            if show:
                plt.show()
            else:
                plt.close()
                
            return file_path
        
        # Plot as a tree-like structure
        level_heights = np.linspace(0, 1, len(levels))
        level_height_map = {level: height for level, height in zip(levels, level_heights)}
        
        # Count splats per level for horizontal spacing
        splats_per_level = {}
        for level in levels:
            splats_per_level[level] = len(splat_registry.get_splats_at_level(level))
        
        # Plot each level
        for level_idx, level in enumerate(levels):
            splats = list(splat_registry.get_splats_at_level(level))
            
            # Skip if no splats at this level
            if not splats:
                continue
            
            # Get color for this level
            color = self.level_colors.get(level, self.default_color)
            
            # Calculate horizontal positions
            count = len(splats)
            if count > 0:
                x_positions = np.linspace(0.1, 0.9, count)
                
                # Plot splats at this level
                y_position = level_height_map[level]
                
                for i, splat in enumerate(splats):
                    x_position = x_positions[i]
                    
                    # Plot splat
                    plt.scatter(x_position, y_position, color=color, s=100, zorder=10)
                    
                    # Add splat ID label
                    plt.text(
                        x_position, y_position + 0.02,
                        f"{splat.id[-4:]}",
                        ha='center', va='bottom',
                        fontsize=8
                    )
                    
                    # Draw connection to parent
                    if splat.parent is not None:
                        # Find parent's position
                        parent_level = splat.parent.level
                        parent_y = level_height_map[parent_level]
                        
                        # Find parent's index in its level
                        parent_splats = list(splat_registry.get_splats_at_level(parent_level))
                        parent_idx = parent_splats.index(splat.parent)
                        parent_count = len(parent_splats)
                        parent_x = np.linspace(0.1, 0.9, parent_count)[parent_idx]
                        
                        # Draw connection line
                        plt.plot(
                            [x_position, parent_x],
                            [y_position, parent_y],
                            'k-', alpha=0.3, zorder=1
                        )
        
        # Add level labels
        for level, height in level_height_map.items():
            plt.text(
                0.02, height,
                level,
                ha='left', va='center',
                fontsize=10, fontweight='bold'
            )
        
        # Remove axis ticks
        plt.xticks([])
        plt.yticks([])
        
        # Add title
        plt.title(title)
        self.safe_tight_layout()
        
        # Save the visualization
        file_path = ""
        if save:
            timestamp = int(time.time())
            file_path = os.path.join(self.output_dir, f"hierarchy_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        # Show the visualization
        if show:
            plt.show()
        else:
            plt.close()
            
        return file_path
    
    def record_adaptation_event(
        self,
        event_type: str,
        splat_id: str,
        tokens: Optional[np.ndarray] = None
    ) -> None:
        """
        Record an adaptation event for later visualization.
        
        Args:
            event_type: Type of adaptation event ('mitosis' or 'death')
            splat_id: ID of the affected splat
            tokens: Optional token embeddings associated with the event
        """
        # Record the event with timestamp
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'splat_id': splat_id,
            'tokens': tokens.shape if tokens is not None else None
        }
        
        self.adaptation_history.append(event)
    
    def visualize_adaptation_history(
        self,
        title: str = "HSA Adaptation History",
        show: bool = True,
        save: bool = True
    ) -> str:
        """
        Visualize the history of adaptation events.
        
        Args:
            title: Title for the visualization
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            
        Returns:
            Path to saved visualization if save=True, empty string otherwise
        """
        if not self.adaptation_history:
            print("No adaptation events to visualize.")
            return ""
        
        # Count events by type
        event_types = [event['type'] for event in self.adaptation_history]
        event_counts = {}
        for event_type in set(event_types):
            event_counts[event_type] = event_types.count(event_type)
        
        # Get timestamps for timeline
        timestamps = [event['timestamp'] for event in self.adaptation_history]
        min_time = min(timestamps)
        times = [(t - min_time) for t in timestamps]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Set for tracking added labels
        added_labels = set()
        
        # Plot adaptation events on a timeline
        for i, event in enumerate(self.adaptation_history):
            event_type = event['type']
            
            # Determine color and marker by event type
            if event_type == 'mitosis':
                color = 'green'
                marker = 'o'
            elif event_type == 'death':
                color = 'red'
                marker = 'x'
            else:
                color = 'blue'
                marker = 's'
            
            # Only add label if not already added
            label = event_type if event_type not in added_labels else ""
            if label:
                added_labels.add(event_type)
            
            # Plot event
            plt.scatter(
                times[i], i % 5,  # Use modulo to spread vertically
                color=color,
                marker=marker,
                s=80,
                label=label
            )
        
        # Add summary bar chart
        subplot_ax = plt.axes([0.65, 0.6, 0.3, 0.3])  # [left, bottom, width, height]
        colors = ['green' if t == 'mitosis' else 'red' if t == 'death' else 'blue' 
                 for t in event_counts.keys()]
        
        # Check if event_counts is empty
        if event_counts:
            subplot_ax.bar(
                list(event_counts.keys()), 
                list(event_counts.values()),
                color=colors
            )
            subplot_ax.set_title("Event Counts")
            subplot_ax.tick_params(labelsize=8)
        else:
            subplot_ax.text(0.5, 0.5, "No events", ha='center', va='center')
        
        # Main plot finishing touches
        plt.title(title)
        plt.xlabel("Time (seconds)")
        plt.yticks([])
        
        # Only add legend if we have labeled elements
        if added_labels:
            self.safe_legend()
            
        plt.grid(axis='x', alpha=0.3)
        
        # Use safe_tight_layout to avoid warnings
        self.safe_tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the subplot
        
        # Save the visualization
        file_path = ""
        if save:
            timestamp = int(time.time())
            file_path = os.path.join(self.output_dir, f"adaptation_history_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        # Show the visualization
        if show:
            plt.show()
        else:
            plt.close()
            
        return file_path
    
    def visualize_level_contributions(
        self,
        level_contributions: Dict[str, float],
        title: str = "HSA Level Contributions",
        show: bool = True,
        save: bool = True
    ) -> str:
        """
        Visualize the contribution of each hierarchy level to attention.
        
        Args:
            level_contributions: Dictionary mapping level names to contribution values
            title: Title for the visualization
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            
        Returns:
            Path to saved visualization if save=True, empty string otherwise
        """
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Handle empty contributions
        if not level_contributions:
            plt.title(f"{title} (No data)")
            plt.text(0.5, 0.5, "No level contribution data available", 
                    ha='center', va='center', fontsize=14)
                    
            # Save the visualization
            file_path = ""
            if save:
                timestamp = int(time.time())
                file_path = os.path.join(self.output_dir, f"level_contributions_{timestamp}.png")
                plt.savefig(file_path, dpi=300)
                
            # Show the visualization
            if show:
                plt.show()
            else:
                plt.close()
                
            return file_path
        
        # Get levels and contributions
        levels = list(level_contributions.keys())
        contributions = list(level_contributions.values())
        
        # Get colors for each level
        colors = [self.level_colors.get(level, self.default_color) for level in levels]
        
        # Create bar chart
        bars = plt.bar(levels, contributions, color=colors)
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            # Avoid adding labels to zero-height bars
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height * 1.01,
                    f'{height:.1%}',
                    ha='center', va='bottom'
                )
        
        # Add title and labels
        plt.title(title)
        plt.ylabel('Contribution')
        
        # Use safe_ylim to avoid identical limits warning
        max_contribution = max(contributions) if contributions else 0.1
        self.safe_ylim(ymin=0, ymax=max_contribution * 1.2, min_range=0.01)
        
        # Add grid
        plt.grid(axis='y', alpha=0.3)
        self.safe_tight_layout()
        
        # Save the visualization
        file_path = ""
        if save:
            timestamp = int(time.time())
            file_path = os.path.join(self.output_dir, f"level_contributions_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        # Show the visualization
        if show:
            plt.show()
        else:
            plt.close()
            
        return file_path

    def visualize_attention_sparsity(
        self,
        attention_matrix: np.ndarray,
        title: str = "HSA Attention Sparsity",
        show: bool = True,
        save: bool = True
    ) -> str:
        """
        Visualize the sparsity pattern of an attention matrix.
        
        Args:
            attention_matrix: The attention matrix to visualize
            title: Title for the visualization
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            
        Returns:
            Path to saved visualization if save=True, empty string otherwise
        """
        plt.figure(figsize=(10, 8))
        
        # Create a binary matrix showing where attention is non-zero
        binary_matrix = (attention_matrix > 0).astype(int)
        
        # Plot the sparsity pattern
        plt.imshow(binary_matrix, cmap='Blues')
        
        # Calculate and display sparsity percentage
        sparsity = 1.0 - (np.count_nonzero(attention_matrix) / attention_matrix.size)
        plt.title(f"{title}\nSparsity: {sparsity:.2%}")
        
        # Add labels
        plt.xlabel('Target Tokens')
        plt.ylabel('Source Tokens')
        
        # Add colorbar
        plt.colorbar(label='Attention Present')
        
        self.safe_tight_layout()
        
        # Save the visualization
        file_path = ""
        if save:
            timestamp = int(time.time())
            file_path = os.path.join(self.output_dir, f"attention_sparsity_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        # Show the visualization
        if show:
            plt.show()
        else:
            plt.close()
            
        return file_path
    
    def create_dashboard(
        self,
        splat_registry: Any,
        attention_matrix: np.ndarray,
        tokens: Optional[np.ndarray] = None,
        title: str = "HSA Dashboard",
        show: bool = True,
        save: bool = True
    ) -> str:
        """
        Create a comprehensive dashboard of HSA visualizations.
        
        Args:
            splat_registry: Registry containing all splats
            attention_matrix: The attention matrix to visualize
            tokens: Optional token embeddings
            title: Title for the dashboard
            show: Whether to display the dashboard
            save: Whether to save the dashboard to a file
            
        Returns:
            Path to saved dashboard if save=True, empty string otherwise
        """
        # Create a multi-panel figure
        fig = plt.figure(figsize=(16, 12))
        
        # Add title
        plt.suptitle(title, fontsize=16, y=0.98)
        
        # Panel 1: Attention Matrix
        plt.subplot(2, 2, 1)
        binary_matrix = (attention_matrix > 0).astype(int)
        plt.imshow(attention_matrix, cmap='viridis')
        plt.title("Attention Matrix")
        plt.colorbar(label='Attention Score')
        
        # Panel 2: Sparsity Pattern
        plt.subplot(2, 2, 2)
        sparsity = 1.0 - (np.count_nonzero(attention_matrix) / attention_matrix.size)
        plt.imshow(binary_matrix, cmap='Blues')
        plt.title(f"Attention Sparsity: {sparsity:.2%}")
        plt.colorbar(label='Attention Present')
        
        # Panel 3: Level Contributions
        plt.subplot(2, 2, 3)
        
        # Get level contributions from registry stats
        level_contributions = {}
        stats = splat_registry.hierarchy.level_weights
        for i, level in enumerate(splat_registry.hierarchy.levels):
            level_contributions[level] = stats[i]
        
        # Create bar chart for level contributions
        levels = list(level_contributions.keys())
        contributions = list(level_contributions.values())
        colors = [self.level_colors.get(level, self.default_color) for level in levels]
        plt.bar(levels, contributions, color=colors)
        plt.title("Level Contributions")
        plt.ylabel('Weight')
        
        # Panel 4: Splat Counts
        plt.subplot(2, 2, 4)
        
        # Count splats per level
        splat_counts = {}
        for level in splat_registry.hierarchy.levels:
            splat_counts[level] = len(splat_registry.get_splats_at_level(level))
        
        # Create bar chart for splat counts
        levels = list(splat_counts.keys())
        counts = list(splat_counts.values())
        colors = [self.level_colors.get(level, self.default_color) for level in levels]
        plt.bar(levels, counts, color=colors)
        plt.title("Splat Counts")
        plt.ylabel('Count')
        
        # Use safe_tight_layout to avoid warnings
        self.safe_tight_layout(fig=fig, rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save the dashboard
        file_path = ""
        if save:
            timestamp = int(time.time())
            file_path = os.path.join(self.output_dir, f"hsa_dashboard_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        # Show the dashboard
        if show:
            plt.show()
        else:
            plt.close()
            
        return file_path


# Example usage
if __name__ == "__main__":
    # This is a simple example of how to use the visualizer
    # In practice, you would use it with real HSA data
    
    # Create sample attention matrix
    seq_len = 20
    attention_matrix = np.random.rand(seq_len, seq_len)
    attention_matrix = attention_matrix * (attention_matrix > 0.8)  # Make it sparse
    
    # Create sample tokens
    tokens = np.random.randn(seq_len, 64)
    
    # Create visualizer
    visualizer = HSAVisualizer()
    
    # Create and save a visualization
    visualizer.visualize_attention_matrix(
        attention_matrix=attention_matrix,
        title="Sample HSA Attention Matrix",
        show=True
    )
