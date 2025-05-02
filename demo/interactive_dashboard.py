"""
Interactive dashboard for visualizing Hierarchical Splat Attention (HSA) adaptation.

This script creates an interactive dashboard for visualizing the adaptation process
in HSA, showing how splats evolve over time as tokens are processed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import logging
from typing import List, Dict, Optional, Tuple, Set, Any

# Import HSA components
from hsa.splat import Splat
from hsa.hierarchy import Hierarchy
from hsa.registry import SplatRegistry
from hsa.dense_attention import DenseAttentionComputer
from hsa.attention_interface import AttentionConfig
from hsa.adaptation_types import AdaptationType, AdaptationResult, AdaptationConfig
from hsa.adaptation_metrics_base import AdaptationMetricsComputer, AdaptationMetrics
from hsa.adaptation_controller import AdaptationController

# Import visualization
from hsa.splat_visualization import SplatVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMetricsComputer(AdaptationMetricsComputer):
    """Simple implementation of metrics computer for demo purposes."""
    
    def compute_metrics(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> AdaptationMetrics:
        activation = self.compute_splat_activation(splat, tokens)
        trend = self.compute_activation_trend(splat)
        variance = self.compute_splat_variance(splat, tokens)
        info_contribution = self.compute_information_contribution(splat, registry, tokens)
        
        return AdaptationMetrics(
            activation_mean=activation,
            activation_trend=trend,
            information_contribution=info_contribution,
            coverage_uniformity=0.5,  # Placeholder
            variance=variance,
            similarity_to_others={}  # Placeholder
        )
    
    def compute_splat_activation(
        self,
        splat: Splat,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        return splat.get_average_activation()
    
    def compute_activation_trend(
        self,
        splat: Splat
    ) -> float:
        values = splat.activation_history.get_values()
        if len(values) < 2:
            return 0.0
        
        # Simple trend calculation
        return values[-1] - values[0]
    
    def compute_splat_variance(
        self,
        splat: Splat,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        # Use trace of covariance as variance
        return np.trace(splat.covariance) / splat.dim
    
    def compute_similarity(
        self,
        splat_a: Splat,
        splat_b: Splat
    ) -> float:
        # Simple distance-based similarity
        distance = np.linalg.norm(splat_a.position - splat_b.position)
        return np.exp(-distance)
    
    def compute_coverage_uniformity(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        return 0.5  # Placeholder
    
    def compute_information_contribution(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        # For demo, use activation as proxy for contribution
        return splat.get_average_activation()


class SimpleCandidateEvaluator:
    """Simple implementation of candidate evaluator for demo purposes."""
    
    def evaluate_mitosis_candidates(self, *args, **kwargs) -> Tuple[Splat, Splat]:
        """Return the first candidate pair."""
        return args[1][0]
    
    def evaluate_merge_candidates(self, *args, **kwargs) -> Splat:
        """Return the first candidate."""
        return args[2][0]
    
    def evaluate_birth_candidates(self, *args, **kwargs) -> Splat:
        """Return the first candidate."""
        return args[0][0]
    
    def evaluate_adjust_candidates(self, *args, **kwargs) -> Splat:
        """Return the first candidate."""
        return args[1][0]


class AdaptationVisualizer:
    """Interactive visualization of HSA adaptation process."""
    
    def __init__(self, dim: int = 2, n_tokens: int = 100, adaptation_frequency: int = 10):
        """Initialize adaptation visualizer.
        
        Args:
            dim: Dimensionality of the embedding space
            n_tokens: Number of token embeddings to generate
            adaptation_frequency: How often to run adaptation
        """
        self.dim = dim
        self.n_tokens = n_tokens
        self.adaptation_frequency = adaptation_frequency
        
        # Initialize components
        self._init_components()
        
        # History of registry states and adaptation results
        self.registry_states = [self.registry.get_all_splats()]
        self.adaptation_results = []
        
        # Current step
        self.step = 0
        
        # Initialize Tkinter UI
        self._init_ui()
    
    def _init_components(self):
        """Initialize HSA components."""
        # Create hierarchy
        self.hierarchy = Hierarchy(
            levels=["token", "phrase", "document"],
            init_splats_per_level=[20, 10, 5],
            level_weights=[0.5, 0.3, 0.2]
        )
        
        # Create registry
        self.registry = SplatRegistry(hierarchy=self.hierarchy, embedding_dim=self.dim)
        
        # Generate random token embeddings
        self.tokens = np.random.normal(0, 1.0, (self.n_tokens, self.dim))
        
        # Initialize splats based on token distribution
        self.registry.initialize_splats(self.tokens)
        
        # Create attention computer
        self.attention_computer = DenseAttentionComputer(
            config=AttentionConfig(
                normalize_levels=True,
                normalize_rows=True,
                causal=False
            )
        )
        
        # Create metrics computer
        self.metrics_computer = SimpleMetricsComputer()
        
        # Create candidate evaluator
        self.candidate_evaluator = SimpleCandidateEvaluator()
        
        # Create adaptation controller
        self.adaptation_controller = AdaptationController(
            registry=self.registry,
            metrics_computer=self.metrics_computer,
            candidate_evaluator=self.candidate_evaluator,
            config=AdaptationConfig(  # Changed from AttentionConfig to AdaptationConfig
                adaptation_frequency=self.adaptation_frequency,
                max_adaptations_per_cycle=3,
                min_lifetime_before_adaptation=5
            )
        )
        
        # Create visualizer
        self.visualizer = SplatVisualizer(figsize=(8, 6))
    
    def _init_ui(self):
        """Initialize the Tkinter UI."""
        self.root = tk.Tk()
        self.root.title("HSA Adaptation Visualizer")
        self.root.geometry("1200x800")
        
        # Create frame for visualization
        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create control panel
        control_frame = ttk.Frame(self.root, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Add controls
        ttk.Label(control_frame, text="HSA Adaptation Controls", font=("Arial", 14)).pack(pady=10)
        
        # Step control
        step_frame = ttk.Frame(control_frame)
        step_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(step_frame, text="Current Step:").pack(side=tk.LEFT)
        self.step_label = ttk.Label(step_frame, text="0")
        self.step_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Step Forward", command=self.step_forward).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Run 10 Steps", command=lambda: self.run_steps(10)).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Run 50 Steps", command=lambda: self.run_steps(50)).pack(fill=tk.X, pady=5)
        
        # Animation control
        ttk.Button(control_frame, text="Start Animation", command=self.start_animation).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Stop Animation", command=self.stop_animation).pack(fill=tk.X, pady=5)
        
        # Visualization options
        ttk.Label(control_frame, text="Visualization Options", font=("Arial", 12)).pack(pady=10)
        
        # Show tokens checkbox
        self.show_tokens_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Tokens", variable=self.show_tokens_var, 
                        command=self.update_visualization).pack(fill=tk.X, pady=5)
        
        # Show level labels
        self.show_level_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Level Labels", variable=self.show_level_labels_var, 
                        command=self.update_visualization).pack(fill=tk.X, pady=5)
        
        # Visualization type
        ttk.Label(control_frame, text="Visualization Type:").pack(anchor=tk.W, pady=5)
        
        self.viz_type_var = tk.StringVar(value="Registry")
        viz_types = ["Registry", "Hierarchy Levels", "Attention Flow", "Adaptation History"]
        
        for viz_type in viz_types:
            ttk.Radiobutton(control_frame, text=viz_type, variable=self.viz_type_var, 
                            value=viz_type, command=self.update_visualization).pack(anchor=tk.W)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initial visualization
        self.update_visualization()
        
        # Animation variables
        self.animation_running = False
        self.animation_id = None
    
    def step_forward(self):
        """Advance the simulation by one step."""
        self.step += 1
        self.step_label.config(text=str(self.step))
        
        # Process tokens
        token_batch = self._get_token_batch()
        
        # Compute attention
        attention_matrix = self.attention_computer.compute_attention(
            tokens=token_batch,
            splat_registry=self.registry
        )
        
        # Apply to splats (update activation history)
        self._apply_attention_to_splats(token_batch, attention_matrix)
        
        # Run adaptation if it's time
        if self.step % self.adaptation_frequency == 0:
            self.status_var.set(f"Step {self.step}: Running adaptation...")
            
            # Run adaptation
            results = self.adaptation_controller.step(token_batch)
            
            if results:
                self.adaptation_results.extend(results)
                self.status_var.set(f"Step {self.step}: Adaptation complete - {len(results)} operations")
            else:
                self.status_var.set(f"Step {self.step}: No adaptation needed")
        else:
            self.status_var.set(f"Step {self.step}: Processing tokens")
        
        # Store current state
        self.registry_states.append(self.registry.get_all_splats())
        
        # Update visualization
        self.update_visualization()
    
    def run_steps(self, n_steps: int):
        """Run multiple steps at once.
        
        Args:
            n_steps: Number of steps to run
        """
        for _ in range(n_steps):
            self.step_forward()
    
    def start_animation(self):
        """Start the animation."""
        if self.animation_running:
            return
            
        self.animation_running = True
        self.animation_id = self.root.after(500, self._animation_step)
        self.status_var.set("Animation running")
    
    def stop_animation(self):
        """Stop the animation."""
        if not self.animation_running:
            return
            
        self.animation_running = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
        self.status_var.set("Animation stopped")
    
    def _animation_step(self):
        """Perform one animation step."""
        if not self.animation_running:
            return
            
        self.step_forward()
        self.animation_id = self.root.after(500, self._animation_step)
    
    def _get_token_batch(self) -> np.ndarray:
        """Get a batch of tokens for the current step.
        
        Returns:
            Token batch
        """
        # For demo, just return random subset of tokens
        batch_size = 20
        indices = np.random.choice(self.n_tokens, batch_size, replace=False)
        return self.tokens[indices]
    
    def _apply_attention_to_splats(self, tokens: np.ndarray, attention_matrix: np.ndarray):
        """Apply attention values to update splat activation history.
        
        Args:
            tokens: Token embeddings
            attention_matrix: Attention matrix
        """
        # Get all splats
        all_splats = self.registry.get_all_splats()
        
        # For each splat, compute attention contribution and update history
        for splat in all_splats:
            # Simple approximation: compute average attention through this splat
            attention_sum = 0.0
            count = 0
            
            for i in range(len(tokens)):
                for j in range(len(tokens)):
                    # Skip diagonal (self-attention)
                    if i == j:
                        continue
                        
                    # Compute attention contribution through this splat
                    contribution = splat.compute_attention(tokens[i], tokens[j])
                    
                    if contribution > 0.01:  # Threshold for counting
                        attention_sum += contribution
                        count += 1
            
            # Compute average and update history
            avg_attention = attention_sum / max(1, count)
            splat.activation_history.add(avg_attention)
    
    def update_visualization(self):
        """Update the visualization based on current settings."""
        # Get visualization type
        viz_type = self.viz_type_var.get()
        
        # Get tokens if needed
        tokens = self.tokens if self.show_tokens_var.get() else None
        
        # Create new figure for the visualization
        plt.close('all')  # Close previous figures
        
        # Create visualization based on type
        if viz_type == "Registry":
            fig = self.visualizer.visualize_registry(
                registry=self.registry,
                tokens=tokens,
                title=f"HSA Registry (Step {self.step})",
                show_legend=self.show_level_labels_var.get(),
                save_path=None
            )
            # Update canvas with new figure
            self.canvas.figure = fig
            
        elif viz_type == "Hierarchy Levels":
            fig = self.visualizer.visualize_hierarchy_levels(
                registry=self.registry,
                tokens=tokens,
                title=f"Hierarchical Levels (Step {self.step})",
                save_path=None
            )
            # Update canvas with new figure
            self.canvas.figure = fig
            
        elif viz_type == "Attention Flow":
            # Create a simple attention matrix for visualization
            token_batch = self._get_token_batch()
            attention_matrix = self.attention_computer.compute_attention(
                tokens=token_batch,
                splat_registry=self.registry
            )
            
            fig = self.visualizer.visualize_attention_flow(
                registry=self.registry,
                attention_matrix=attention_matrix,
                tokens=token_batch,
                token_indices=list(range(min(10, len(token_batch)))),
                title=f"Attention Flow (Step {self.step})",
                save_path=None
            )
            # Update canvas with new figure
            self.canvas.figure = fig
            
        elif viz_type == "Adaptation History":
            if self.adaptation_results:
                fig = self.visualizer.visualize_adaptation_history(
                    history=self.adaptation_results[-10:],  # Show last 10 results
                    registry=self.registry,
                    tokens=tokens,
                    save_path=None
                )
                # Update canvas with new figure
                self.canvas.figure = fig
            else:
                # Create a new figure for "No adaptation results yet" message
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "No adaptation results yet", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Adaptation History (Step {self.step})")
                self.canvas.figure = fig
        
        # Update canvas
        self.canvas.draw()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


if __name__ == "__main__":
    # Create and run visualizer
    visualizer = AdaptationVisualizer(dim=2, n_tokens=100, adaptation_frequency=10)
    visualizer.run()
