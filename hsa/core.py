"""
Main entry point for Hierarchical Splat Attention (HSA).

This module provides the core HSA class and orchestrates all components:
- Data Structures
- Initialization
- Attention Computation
- Adaptation
- Training
- Model Integration

This is the main interface for using HSA in transformer models.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import all HSA components
from .data_structures import Splat, Hierarchy, SplatRegistry
from .initialization import initialize_splats, reinitialize_splat
from .attention import create_attention_computer, AttentionComputer, SplatAttentionMetrics
from .adaptation import check_adaptation_triggers, perform_adaptations, AdaptationType
from .training import train_hsa, evaluate_hsa, TrainingConfig
from .model_integration import HSAAttention, HSATransformerLayer, replace_attention_with_hsa, HSAModelAdapter

class HSA:
    """
    Main class for Hierarchical Splat Attention.
    
    This class provides a unified interface for working with HSA,
    orchestrating the interaction between all components.
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        device: str = "cpu"
    ):
        """
        Initialize HSA with the provided configuration.
        
        Args:
            config: Configuration dictionary for HSA
            device: Device to use ('cpu' or 'cuda')
        """
        # Use default config if none provided
        self.config = config or self._get_default_config()
        self.device = device
        
        # Extract hierarchy configuration
        self.hierarchy = self._create_hierarchy_from_config()
        
        # Initialize components
        self.splat_registry = None  # Will be created during initialization
        self.attention_computer = create_attention_computer(
            hierarchy=self.hierarchy,
            sparse_topk=self.config.get("attention", {}).get("sparse_topk", 64),
            efficient=True
        )
        
        # Metrics tracker
        self.metrics_tracker = SplatAttentionMetrics()
        
        # Model adapter for integration
        self.model_adapter = HSAModelAdapter(
            model_type=self.config.get("model", {}).get("type", "default")
        )
        
        # Initialization state
        self.is_initialized = False
        
        # Statistics tracking
        self.stats = {
            "adaptations": {
                "mitosis": 0,
                "death": 0
            },
            "attention_sparsity": 0.0,
            "level_contributions": {}
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for HSA.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "hierarchy": {
                "levels": ["Token", "Phrase", "Section", "Document"],
                "init_splats_per_level": [100, 50, 20, 5],
                "level_weights": [0.4, 0.3, 0.2, 0.1]
            },
            "initialization": {
                "clustering": {
                    "method": "spectral",
                    "n_neighbors": 10,
                    "affinity": "nearest_neighbors"
                }
            },
            "attention": {
                "sparse_topk": 64
            },
            "adaptation": {
                "mitosis_threshold": 0.1,
                "death_threshold": 0.01,
                "consecutive_batches": 3,
                "adaptation_frequency": 5,
                "enable_adaptation": True
            },
            "training": {
                "epochs": 20,
                "batch_size": 32,
                "learning_rate": 0.001,
                "early_stopping_patience": 5
            },
            "model": {
                "type": "default",
                "replace_in_place": True
            }
        }
    
    def _create_hierarchy_from_config(self) -> Hierarchy:
        """
        Create a Hierarchy object from the configuration.
        
        Returns:
            Hierarchy object
        """
        hierarchy_config = self.config.get("hierarchy", {})
        
        # Get values with defaults
        levels = hierarchy_config.get("levels", ["Token", "Phrase", "Section", "Document"])
        init_splats_per_level = hierarchy_config.get("init_splats_per_level", [100, 50, 20, 5])
        level_weights = hierarchy_config.get("level_weights", [0.4, 0.3, 0.2, 0.1])
        
        # Create hierarchy
        return Hierarchy(
            levels=levels,
            init_splats_per_level=init_splats_per_level,
            level_weights=level_weights
        )
    
    def initialize(self, tokens: np.ndarray) -> None:
        """
        Initialize splats based on token embeddings.
        
        Args:
            tokens: Token embeddings [sequence_length, embedding_dim]
        """
        # Extract initialization parameters
        init_config = self.config.get("initialization", {})
        clustering_config = init_config.get("clustering", {})
        
        # Get clustering parameters
        method = clustering_config.get("method", "spectral")
        n_neighbors = clustering_config.get("n_neighbors", 10)
        affinity = clustering_config.get("affinity", "nearest_neighbors")
        
        # Initialize splats
        self.splat_registry = initialize_splats(
            tokens=tokens,
            hierarchy_config=self.config["hierarchy"],
            n_neighbors=n_neighbors,
            affinity=affinity
        )
        
        self.is_initialized = True
        
        # Log stats
        for level in self.hierarchy.levels:
            splats = self.splat_registry.get_splats_at_level(level)
            print(f"Initialized {len(splats)} splats at level '{level}'")
    
    def compute_attention(
        self, 
        tokens: np.ndarray
    ) -> np.ndarray:
        """
        Compute attention matrix using HSA.
        
        Args:
            tokens: Token embeddings [sequence_length, embedding_dim]
            
        Returns:
            Attention matrix [sequence_length, sequence_length]
            
        Raises:
            RuntimeError: If splats are not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("HSA has not been initialized. Call initialize() first.")
        
        # Compute attention
        attention_matrix = self.attention_computer.compute_attention(
            tokens=tokens,
            splat_registry=self.splat_registry
        )
        
        # Compute sparsity (percentage of zeros)
        sparsity = np.count_nonzero(attention_matrix == 0) / attention_matrix.size
        self.stats["attention_sparsity"] = sparsity
        
        # Compute per-level contribution
        self._compute_level_contributions(tokens)
        
        return attention_matrix
    
    def _compute_level_contributions(self, tokens: np.ndarray) -> None:
        """
        Compute the relative contribution of each level to attention.
        
        Args:
            tokens: Token embeddings
        """
        # Compute full attention matrix
        full_attention = self.attention_computer.compute_attention(
            tokens=tokens,
            splat_registry=self.splat_registry
        )
        total_attention = np.sum(full_attention)
        if abs(total_attention) < 1e-10:  # If total attention is almost zero
            #logger.warning("Total attention is near zero, skipping level contribution calculation")
            return
    
        # Skip if no attention
        if total_attention == 0:
            return
        
        # Compute per-level contribution
        level_contributions = {}
        
        for level in self.hierarchy.levels:
            # Create a registry with only this level's splats
            level_registry = SplatRegistry(self.hierarchy)
            for splat in self.splat_registry.get_splats_at_level(level):
                level_registry.register(splat)
            
            # Compute this level's attention
            level_attention = self.attention_computer.compute_attention(
                tokens=tokens,
                splat_registry=level_registry
            )
            level_attention_sum = np.sum(level_attention)
            
            # Calculate relative contribution
            level_contributions[level] = level_attention_sum / (total_attention + 1e-10)
        
        self.stats["level_contributions"] = level_contributions
    
    def adapt(
        self, 
        tokens: np.ndarray, 
        target_attention: Optional[np.ndarray] = None,
        cpu_efficient: bool = True  # New parameter for CPU optimization
    ) -> Dict[str, int]:
        """
        Perform adaptation (mitosis & death) on splats.
        Optimized for CPU execution with reduced computational complexity.
        
        Args:
            tokens: Token embeddings
            target_attention: Optional target attention matrix
            cpu_efficient: Whether to use optimizations for CPU (reduced computation)
            
        Returns:
            Adaptation results (dictionary mapping adaptation types to counts)
            
        Raises:
            RuntimeError: If splats are not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("HSA has not been initialized. Call initialize() first.")
        
        # Initialize stats dictionary if needed
        if "adaptations" not in self.stats:
            self.stats["adaptations"] = {}
    
        # Ensure all adaptation types have keys
        for key in ["mitosis", "death", "merge", "birth"]:
            if key not in self.stats["adaptations"]:
                self.stats["adaptations"][key] = 0
        
        # For CPU optimization, subsample tokens to reduce computation
        if cpu_efficient and tokens.shape[0] > 100:
            # Take a subsample of tokens
            subsample_size = min(100, tokens.shape[0])
            subsample_indices = np.random.choice(tokens.shape[0], size=subsample_size, replace=False)
            tokens = tokens[subsample_indices]
            print(f"Using {subsample_size} tokens for adaptation (CPU optimization)")
        
        # If on CPU, avoid computing full attention matrix and limit adaptations
        """
        if cpu_efficient and (not hasattr(self, 'device') or self.device == 'cpu'):
            # Skip expensive attention computation
            current_attention = None
            target_attention = None
            print("Skipping attention computation for CPU efficiency")
        else:
            # If no target attention provided, use the current attention as target
        """
        if target_attention is None:
            current_attention = self.compute_attention(tokens)
            target_attention = current_attention
        else:
            current_attention = self.compute_attention(tokens)
    
        # Extract adaptation parameters
        adaptation_config = self.config.get("adaptation", {})
        mitosis_threshold = adaptation_config.get("mitosis_threshold", 0.1)
        death_threshold = adaptation_config.get("death_threshold", 0.01)
        
        # Set a timeout for the adaptation process
        start_time = time.time()
        max_time = 30  # Maximum seconds for adaptation
        
        # For CPU, use simplified metrics calculation
        if cpu_efficient and (current_attention is None or target_attention is None):
            # Use a simplified metric calculation
            for level in self.hierarchy.levels:
                splats = self.splat_registry.get_splats_at_level(level)
                for splat in splats:
                    # Check timeout
                    if time.time() - start_time > max_time:
                        print("Timeout during metrics calculation")
                        break
                        
                    # Compute a simple activation estimate
                    active_count = 0
                    total_count = 0
                    sample_size = min(20, tokens.shape[0])  # Limit sample size 
                    
                    for i in range(sample_size):
                        # Compute distance from token to splat
                        diff = tokens[i] - splat.position
                        dist = np.sqrt(diff @ splat.covariance_inverse @ diff)
                        
                        # Count as active if within influence
                        if dist < 2.0:
                            active_count += 1
                        total_count += 1
                    
                    # Compute activation ratio
                    activation = active_count / max(1, total_count)
                    self.metrics_tracker.splat_activations[splat.id] = activation
                    
                    # Assign random error contribution for mitosis decisions
                    if np.random.random() < 0.1:  # 10% chance of high error contribution
                        error_contribution = np.random.uniform(mitosis_threshold, mitosis_threshold * 2)
                    else:
                        error_contribution = np.random.uniform(0, mitosis_threshold * 0.5)
                    
                    self.metrics_tracker.splat_error_contributions[splat.id] = error_contribution
        else:
            # Calculate metrics for all splats using standard approach
            for level in self.hierarchy.levels:
                splats = self.splat_registry.get_splats_at_level(level)
                for splat in splats:
                    # Check timeout
                    if time.time() - start_time > max_time:
                        print("Timeout during metrics calculation")
                        break
                        
                    # Compute activation
                    self.metrics_tracker.compute_splat_activation(
                        splat, tokens, current_attention
                    )
                    
                    # Compute error contribution if attention matrices are available
                    if current_attention is not None and target_attention is not None:
                        self.metrics_tracker.compute_splat_error_contribution(
                            splat, tokens, target_attention, current_attention
                        )
        
        # Check if we've timed out
        if time.time() - start_time > max_time:
            print(f"Adaptation metrics timed out after {time.time() - start_time:.2f}s. Using simplified adaptation.")
            # Perform minimal adaptation
            return {"mitosis": 0, "death": 0, "merge": 0, "birth": 0 }
        
        # Check for necessary adaptations with timeout and CPU optimization flags
        max_adaptation_count = 10 if cpu_efficient else 20
        adaptation_actions = check_adaptation_triggers(
            splat_registry=self.splat_registry,
            metrics_tracker=self.metrics_tracker,
            tokens=tokens,  # Pass tokens for mitosis checking
            mitosis_threshold=mitosis_threshold,
            death_threshold=death_threshold,
            max_adaptation_count=max_adaptation_count,
            cpu_optimization=cpu_efficient
        )
        
        # Check if we've timed out
        if time.time() - start_time > max_time * 0.8:
            print(f"Not enough time left for adaptations after {time.time() - start_time:.2f}s.")
            # Return minimal results to continue training
            return {"mitosis": 0, "death": 0, "merge": 0, "birth": 0}
        
        # Perform adaptations if needed
        if adaptation_actions:
            # Count adaptations by type before performing them
            mitosis_count = sum(1 for action, _ in adaptation_actions if action == AdaptationType.MITOSIS)
            death_count = sum(1 for action, _ in adaptation_actions if action == AdaptationType.DEATH)
            birth_count = sum(1 for action, _ in adaptation_actions if action == AdaptationType.BIRTH)
            merge_count = sum(1 for action, _ in adaptation_actions if action == AdaptationType.MERGE)
        
            
            # Update adaptation stats
            self.stats["adaptations"]["mitosis"] += mitosis_count
            self.stats["adaptations"]["death"] += death_count
            self.stats["adaptations"]["merge"] += merge_count
            self.stats["adaptations"]["birth"] += birth_count
            
            # Perform the adaptations with CPU optimization flag
            updated_registry, result = perform_adaptations(
                splat_registry=self.splat_registry,
                adaptations=adaptation_actions,
                tokens=tokens,
                cpu_optimization=cpu_efficient
            )
            
            # Update our registry reference
            self.splat_registry = updated_registry
            
            # Format results for return
            return {
                "birth": birth_count,
                "merge": merge_count,
                "mitosis": mitosis_count,
                "death": death_count
            }
         
    def convert_model(
        self, 
        model: nn.Module, 
        model_type: Optional[str] = None
    ) -> nn.Module:
        """
        Convert a standard transformer model to use HSA.
        
        Args:
            model: The transformer model to convert
            model_type: Optional model type (e.g., 'bert', 'gpt')
            
        Returns:
            Modified model using HSA attention
        """
        # Update model type if provided
        if model_type is not None:
            self.model_adapter = HSAModelAdapter(model_type=model_type)
        
        # Extract model config
        model_config = self.config.get("model", {})
        replace_in_place = model_config.get("replace_in_place", True)
        
        # Convert the model
        modified_model = replace_attention_with_hsa(
            model=model,
            hsa_config=self.config,
            replace_in_place=replace_in_place
        )
        
        return modified_model
    
    def create_transformer_layer(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None
    ) -> HSATransformerLayer:
        """
        Create a transformer layer with HSA attention.
        
        Args:
            dim: Hidden dimension size
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension (defaults to 4*dim)
            
        Returns:
            Transformer layer with HSA attention
        """
        if ffn_dim is None:
            ffn_dim = 4 * dim
        
        return HSATransformerLayer(
            dim=dim,
            hierarchy_config=self.config["hierarchy"],
            num_heads=num_heads,
            ffn_dim=ffn_dim
        )
    
    def create_attention_layer(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None
    ) -> HSAAttention:
        """
        Create an HSA attention layer.
        
        Args:
            dim: Hidden dimension size
            num_heads: Number of attention heads
            head_dim: Dimension per head (defaults to dim/num_heads)
            
        Returns:
            HSA attention layer
        """
        return HSAAttention(
            dim=dim,
            hierarchy_config=self.config["hierarchy"],
            num_heads=num_heads,
            head_dim=head_dim
        )
    
    def train(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        custom_config: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
        """
        Train a model using HSA.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            custom_config: Optional custom training configuration
            
        Returns:
            Training results
        """
        # Get training configuration
        train_config_dict = self.config.get("training", {})
        
        # Override with custom config if provided
        if custom_config:
            train_config_dict.update(custom_config)
        
        # Create TrainingConfig object
        train_config = TrainingConfig(
            epochs=train_config_dict.get("epochs", 20),
            batch_size=train_config_dict.get("batch_size", 32),
            learning_rate=train_config_dict.get("learning_rate", 0.001),
            adaptation_frequency=self.config.get("adaptation", {}).get("adaptation_frequency", 5),
            mitosis_threshold=self.config.get("adaptation", {}).get("mitosis_threshold", 0.1),
            death_threshold=self.config.get("adaptation", {}).get("death_threshold", 0.01),
            early_stopping_patience=train_config_dict.get("early_stopping_patience", 5),
            enable_adaptation=self.config.get("adaptation", {}).get("enable_adaptation", True),
            device=self.device
        )
        
        # Train the model
        results = train_hsa(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            hierarchy=self.hierarchy,
            splat_registry=self.splat_registry
        )
        
        # Update splat registry with the trained one
        self.splat_registry = results.get("splat_registry", self.splat_registry)
        
        return results
    
    def evaluate(
        self,
        model: nn.Module,
        data_loader: Any,
        task_loss_fn: Callable
    ) -> Dict[str, float]:
        """
        Evaluate a model using HSA.
        
        Args:
            model: The model to evaluate
            data_loader: DataLoader for evaluation data
            task_loss_fn: Loss function for the task
            
        Returns:
            Evaluation metrics
        """
        # Perform evaluation
        metrics = evaluate_hsa(
            model=model,
            data_loader=data_loader,
            splat_registry=self.splat_registry,
            attention_computer=self.attention_computer,
            task_loss_fn=task_loss_fn
        )
        
        return metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the HSA system.
        
        Returns:
            Dictionary of statistics
        """
        # Add splat counts to stats
        splat_counts = {}
        
        if self.is_initialized:
            for level in self.hierarchy.levels:
                splats = self.splat_registry.get_splats_at_level(level)
                splat_counts[level] = len(splats)
            
            splat_counts["total"] = len(self.splat_registry.splats)
        
        stats = {
            "splat_counts": splat_counts,
            "adaptations": self.stats["adaptations"],
            "attention_sparsity": self.stats["attention_sparsity"],
            "level_contributions": self.stats["level_contributions"]
        }
        
        return stats
    
    def save(self, path: str) -> None:
        """
        Save the HSA state to disk.
        
        Args:
            path: Path to save the state
        """
        # This is a placeholder - in a full implementation, we would
        # serialize the splat registry and other state
        print(f"Saving HSA state to {path}")
        
        # For now, just save some basic stats
        with open(f"{path}_stats.txt", "w") as f:
            stats = self.get_stats()
            f.write("HSA Statistics:\n")
            f.write(f"Splat counts: {stats['splat_counts']}\n")
            f.write(f"Adaptations: {stats['adaptations']}\n")
            f.write(f"Attention sparsity: {stats['attention_sparsity']:.4f}\n")
            f.write(f"Level contributions: {stats['level_contributions']}\n")
    
    def load(self, path: str) -> None:
        """
        Load the HSA state from disk.
        
        Args:
            path: Path to load the state from
        """
        # This is a placeholder - in a full implementation, we would
        # deserialize the splat registry and other state
        print(f"Loading HSA state from {path}")
        
        # For now, just acknowledge the load
        print("HSA state loaded (placeholder)")
    
    def __repr__(self) -> str:
        """String representation of the HSA instance."""
        hierarchy_info = f"Hierarchy: {self.hierarchy.levels}"
        splat_info = ""
        
        if self.is_initialized:
            total_splats = len(self.splat_registry.splats)
            splat_info = f", Total splats: {total_splats}"
        
        return f"HSA({hierarchy_info}{splat_info})"


def create_hsa(config: Optional[Dict[str, Any]] = None) -> HSA:
    """
    Factory function to create an HSA instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        HSA instance
    """
    return HSA(config=config)


# Simple usage example
if __name__ == "__main__":
    # Create an HSA instance with default configuration
    hsa = create_hsa()
    
    # Display initial state
    print(hsa)
    
    # Create some dummy token embeddings
    tokens = np.random.randn(16, 64)  # 16 tokens, 64-dim embeddings
    
    # Initialize splats
    hsa.initialize(tokens)
    
    # Display state after initialization
    print(hsa)
    
    # Compute attention matrix
    attention_matrix = hsa.compute_attention(tokens)
    print(f"Attention matrix shape: {attention_matrix.shape}")
    
    # Get statistics
    stats = hsa.get_stats()
    print(f"Splat counts: {stats['splat_counts']}")
    print(f"Level contributions: {stats['level_contributions']}")
    
    # Adapt splats
    hsa.adapt(tokens)
    
    # Create a new attention layer using HSA
    attention_layer = hsa.create_attention_layer(dim=64, num_heads=4)
    print(f"Created HSA attention layer: {attention_layer}")
