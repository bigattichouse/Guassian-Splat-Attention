"""
Training utilities for Hierarchical Splat Attention (HSA).

This module provides utility classes and functions for HSA training,
including gradient computation and configuration.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from .splat import Splat
from .registry import SplatRegistry
from .attention_interface import AttentionComputer
from .adaptation_types import AdaptationConfig
from .loss_functions import MSELoss


class TrainingConfig:
    """Configuration for HSA training."""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        num_epochs: int = 10,
        weight_decay: float = 0.0001,
        optimizer_type: str = "adam",
        scheduler_type: Optional[str] = "cosine",
        gradient_clip: float = 1.0,
        regularization_strength: float = 0.001,
        early_stopping_patience: int = 5,
        validation_interval: int = 100,
        adaptation_config: Optional[AdaptationConfig] = None,
        use_curriculum: bool = False,
        distillation_temp: float = 1.0,
        distillation_alpha: float = 0.5,
        use_mixed_precision: bool = False,
        seed: int = 42,
        device: str = "auto",
        dtype: str = "float32",
        save_interval: int = 1000,
        save_path: str = "./models/hsa",
        log_dir: str = "./logs/hsa",
        mixed_precision_dtype: str = "float16"
    ):
        """Initialize training configuration.
        
        Args:
            learning_rate: Base learning rate
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            weight_decay: L2 regularization strength
            optimizer_type: Optimizer type (adam, sgd, etc.)
            scheduler_type: Learning rate scheduler type (cosine, step, etc.)
            gradient_clip: Gradient clipping value
            regularization_strength: Strength of splat regularization
            early_stopping_patience: Patience for early stopping
            validation_interval: Steps between validation runs
            adaptation_config: Configuration for dynamic adaptation
            use_curriculum: Whether to use curriculum learning
            distillation_temp: Temperature for knowledge distillation
            distillation_alpha: Alpha parameter for distillation loss
            use_mixed_precision: Whether to use mixed precision training
            seed: Random seed for reproducibility
            device: Device to use (cpu, cuda, auto)
            dtype: Data type for training (float32, float16, etc.)
            save_interval: Steps between model saving
            save_path: Directory to save models
            log_dir: Directory for logs
            mixed_precision_dtype: Data type for mixed precision
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.gradient_clip = gradient_clip
        self.regularization_strength = regularization_strength
        self.early_stopping_patience = early_stopping_patience
        self.validation_interval = validation_interval
        self.adaptation_config = adaptation_config or AdaptationConfig()
        self.use_curriculum = use_curriculum
        self.distillation_temp = distillation_temp
        self.distillation_alpha = distillation_alpha
        self.use_mixed_precision = use_mixed_precision
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.save_interval = save_interval
        self.save_path = save_path
        self.log_dir = log_dir
        self.mixed_precision_dtype = mixed_precision_dtype


class SplatGradientComputer:
    """Compute gradients for splat parameters from attention gradients."""
    
    def __init__(self, registry: SplatRegistry, attention_computer: AttentionComputer):
        """Initialize gradient computer.
        
        Args:
            registry: SplatRegistry containing splats
            attention_computer: AttentionComputer to use for gradient computation
        """
        self.registry = registry
        self.attention_computer = attention_computer
    
    def compute_gradients(
        self,
        tokens: np.ndarray,
        attention_gradient: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute gradients for all splats.
        
        Args:
            tokens: Token embeddings
            attention_gradient: Gradient of loss with respect to attention matrix
            
        Returns:
            Dictionary mapping splat IDs to parameter gradients
        """
        # Get all splats
        all_splats = self.registry.get_all_splats()
        
        # Initialize gradient dictionary
        gradients = {}
        
        # Compute gradients for each splat
        for splat in all_splats:
            splat_grads = self._compute_splat_gradients(splat, tokens, attention_gradient)
            if splat_grads:
                gradients[splat.id] = splat_grads
        
        return gradients
    
    def _compute_splat_gradients(
        self,
        splat: Splat,
        tokens: np.ndarray,
        attention_gradient: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute gradients for a single splat.
        
        Args:
            splat: Splat to compute gradients for
            tokens: Token embeddings
            attention_gradient: Gradient of loss with respect to attention matrix
            
        Returns:
            Dictionary mapping parameter names to gradients
        """
        # Compute attention contribution of this splat
        splat_attention = self.attention_computer.compute_splat_attention_map(tokens, splat)
        
        # Compute gradient contribution (simpler calculation with finite differences)
        # In production, use analytical gradients for better performance
        
        # Initialize gradients
        gradients = {}
        
        # Compute gradient for position
        position_grad = np.zeros_like(splat.position)
        epsilon = 1e-5
        
        for i in range(splat.dim):
            # Perturb position
            eps_vector = np.zeros(splat.dim)
            eps_vector[i] = epsilon
            
            # Forward perturbation
            splat.update_parameters(position=splat.position + eps_vector)
            attention_plus = self.attention_computer.compute_splat_attention_map(tokens, splat)
            
            # Backward perturbation
            splat.update_parameters(position=splat.position - 2 * eps_vector)
            attention_minus = self.attention_computer.compute_splat_attention_map(tokens, splat)
            
            # Reset position
            splat.update_parameters(position=splat.position + eps_vector)
            
            # Compute gradient using central difference
            attention_diff = (attention_plus - attention_minus) / (2 * epsilon)
            position_grad[i] = np.sum(attention_diff * attention_gradient)
        
        gradients["position"] = position_grad
        
        # Compute gradient for amplitude (simpler)
        original_amplitude = splat.amplitude
        
        # Forward perturbation
        splat.update_parameters(amplitude=original_amplitude + epsilon)
        attention_plus = self.attention_computer.compute_splat_attention_map(tokens, splat)
        
        # Reset amplitude
        splat.update_parameters(amplitude=original_amplitude)
        
        # Compute gradient
        attention_diff = (attention_plus - splat_attention) / epsilon
        amplitude_grad = np.sum(attention_diff * attention_gradient)
        
        gradients["amplitude"] = amplitude_grad
        
        # For covariance, use a simplified approach (diagonal only)
        # In production, use a more sophisticated method for the full matrix
        covariance_grad = np.zeros_like(splat.covariance)
        
        for i in range(splat.dim):
            # Perturb diagonal element
            eps_matrix = np.zeros((splat.dim, splat.dim))
            eps_matrix[i, i] = epsilon
            
            # Forward perturbation
            new_covariance = splat.covariance + eps_matrix
            splat.update_parameters(covariance=new_covariance)
            attention_plus = self.attention_computer.compute_splat_attention_map(tokens, splat)
            
            # Reset covariance
            splat.update_parameters(covariance=splat.covariance - eps_matrix)
            
            # Compute gradient
            attention_diff = (attention_plus - splat_attention) / epsilon
            covariance_grad[i, i] = np.sum(attention_diff * attention_gradient)
        
        gradients["covariance"] = covariance_grad
        
        return gradients
    
    def compute_efficient_gradients(
        self,
        tokens: np.ndarray,
        attention_gradient: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute gradients more efficiently using analytical formulas.
        
        This is a placeholder for a more efficient implementation.
        
        Args:
            tokens: Token embeddings
            attention_gradient: Gradient of loss with respect to attention matrix
            
        Returns:
            Dictionary mapping splat IDs to parameter gradients
        """
        # This would implement analytical gradients for better performance
        # For now, call the simple implementation
        return self.compute_gradients(tokens, attention_gradient)


def compute_convergence_metrics(
    registry: SplatRegistry,
    history: Dict[str, List[float]]
) -> Dict[str, float]:
    """Compute convergence metrics based on training history.
    
    Args:
        registry: Trained SplatRegistry
        history: Training history
        
    Returns:
        Dictionary with convergence metrics
    """
    metrics = {}
    
    # Check if training has converged
    if len(history["loss"]) > 10:
        # Compute mean and std of recent losses
        recent_losses = history["loss"][-10:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)
        
        # Compute convergence score
        if mean_loss > 0:
            cv = std_loss / mean_loss  # Coefficient of variation
            metrics["convergence_score"] = 1.0 - min(1.0, cv * 10)
        else:
            metrics["convergence_score"] = 1.0
        
        # Compute loss improvement
        if len(history["loss"]) > 20:
            prev_losses = history["loss"][-20:-10]
            mean_prev_loss = np.mean(prev_losses)
            
            if mean_prev_loss > 0:
                improvement = (mean_prev_loss - mean_loss) / mean_prev_loss
                metrics["improvement"] = improvement
            else:
                metrics["improvement"] = 0.0
    else:
        metrics["convergence_score"] = 0.0
        metrics["improvement"] = 0.0
    
    # Compute splat distribution metrics
    level_counts = {}
    for level in registry.hierarchy.levels:
        level_counts[level] = registry.count_splats(level)
    
    metrics["level_counts"] = level_counts
    metrics["total_splats"] = sum(level_counts.values())
    
    return metrics


def apply_splat_regularization(
    registry: SplatRegistry,
    alpha: float = 0.01
) -> float:
    """Apply regularization to splat parameters.
    
    Args:
        registry: SplatRegistry to regularize
        alpha: Regularization strength
        
    Returns:
        Regularization loss
    """
    reg_loss = 0.0
    
    # L2 regularization on parameters
    for splat in registry.get_all_splats():
        # Position regularization
        reg_loss += alpha * np.sum(splat.position ** 2)
        
        # Covariance regularization (trace)
        reg_loss += alpha * np.trace(splat.covariance)
        
        # Amplitude regularization
        reg_loss += alpha * splat.amplitude ** 2
    
    return reg_loss


def transfer_from_pretrained(
    source_registry: SplatRegistry,
    target_registry: SplatRegistry
) -> SplatRegistry:
    """Transfer knowledge from a pretrained registry to a target registry.
    
    Args:
        source_registry: Source pretrained registry
        target_registry: Target registry to transfer to
        
    Returns:
        Updated target registry
    """
    # Get all source splats
    source_splats = source_registry.get_all_splats()
    
    # Clear target registry and initialize with same hierarchy
    target_registry.hierarchy = source_registry.hierarchy
    
    # Add all splats to target registry
    for splat in source_splats:
        # Clone the splat
        new_splat = splat.clone()
        
        # Add to target registry
        target_registry.register(new_splat)
    
    # Ensure consistency of parent-child relationships
    parent_map = {}
    for splat in source_splats:
        if splat.parent is not None:
            parent_id = splat.parent.id
            if parent_id in parent_map:
                new_splat = target_registry.get_splat(splat.id)
                new_parent = target_registry.get_splat(parent_map[parent_id])
                new_splat.parent = new_parent
                new_parent.children.add(new_splat)
    
    # Repair any integrity issues
    target_registry.repair_integrity()
    
    return target_registry


def compute_splat_gradients(
    attention_matrix: np.ndarray,
    target_matrix: np.ndarray,
    registry: SplatRegistry,
    attention_computer: AttentionComputer,
    tokens: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute gradients for splat parameters.
    
    Args:
        attention_matrix: Predicted attention matrix
        target_matrix: Target attention matrix
        registry: SplatRegistry containing splats
        attention_computer: AttentionComputer for computing attention
        tokens: Token embeddings
        
    Returns:
        Dictionary mapping splat IDs to parameter gradients
    """
    # Create loss function
    loss_fn = MSELoss()
    
    # Compute loss and attention gradient
    _, attention_gradient = loss_fn.compute_loss(attention_matrix, target_matrix)
    
    # Create gradient computer
    gradient_computer = SplatGradientComputer(registry, attention_computer)
    
    # Compute splat gradients
    return gradient_computer.compute_gradients(tokens, attention_gradient)
