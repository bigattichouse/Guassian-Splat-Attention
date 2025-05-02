"""
Optimizer implementations for Hierarchical Splat Attention (HSA).

This module provides implementations of various optimizers for training
HSA models, including SGD and Adam variants specific to splat parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .splat import Splat


class SplatOptimizer(ABC):
    """Abstract base class for splat parameter optimizers."""
    
    @abstractmethod
    def step(
        self,
        splats: List[Splat],
        gradients: Dict[str, Dict[str, np.ndarray]],
        learning_rate: float
    ) -> None:
        """Perform one optimization step.
        
        Args:
            splats: List of splats to update
            gradients: Gradients for each splat parameter
            learning_rate: Learning rate for this step
        """
        pass


class SplatSGD(SplatOptimizer):
    """Stochastic gradient descent optimizer for splat parameters."""
    
    def __init__(self, weight_decay: float = 0.0001, momentum: float = 0.9):
        """Initialize SGD optimizer.
        
        Args:
            weight_decay: L2 regularization strength
            momentum: Momentum factor
        """
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocity = {}  # Momentum state
    
    def step(
        self,
        splats: List[Splat],
        gradients: Dict[str, Dict[str, np.ndarray]],
        learning_rate: float
    ) -> None:
        """Perform one optimization step.
        
        Args:
            splats: List of splats to update
            gradients: Gradients for each splat parameter
            learning_rate: Learning rate for this step
        """
        for splat in splats:
            if splat.id not in gradients:
                continue
                
            splat_grads = gradients[splat.id]
            
            # Initialize velocity for new splats
            if splat.id not in self.velocity:
                self.velocity[splat.id] = {
                    "position": np.zeros_like(splat.position),
                    "covariance": np.zeros_like(splat.covariance),
                    "amplitude": 0.0
                }
            
            # Update position with momentum
            if "position" in splat_grads:
                grad = splat_grads["position"] + self.weight_decay * splat.position
                self.velocity[splat.id]["position"] = (
                    self.momentum * self.velocity[splat.id]["position"] - learning_rate * grad
                )
                new_position = splat.position + self.velocity[splat.id]["position"]
                
                # Prepare for update
                splat_params = {"position": new_position}
                
                # Update covariance if gradient is available
                if "covariance" in splat_grads:
                    grad = splat_grads["covariance"] + self.weight_decay * splat.covariance
                    self.velocity[splat.id]["covariance"] = (
                        self.momentum * self.velocity[splat.id]["covariance"] - learning_rate * grad
                    )
                    new_covariance = splat.covariance + self.velocity[splat.id]["covariance"]
                    splat_params["covariance"] = new_covariance
                
                # Update amplitude if gradient is available
                if "amplitude" in splat_grads:
                    grad = splat_grads["amplitude"] + self.weight_decay * splat.amplitude
                    self.velocity[splat.id]["amplitude"] = (
                        self.momentum * self.velocity[splat.id]["amplitude"] - learning_rate * grad
                    )
                    new_amplitude = splat.amplitude + self.velocity[splat.id]["amplitude"]
                    new_amplitude = max(0.01, new_amplitude)  # Ensure positive amplitude
                    splat_params["amplitude"] = new_amplitude
                
                # Apply the updates
                splat.update_parameters(**splat_params)


class SplatAdam(SplatOptimizer):
    """Adam optimizer for splat parameters."""
    
    def __init__(
        self,
        weight_decay: float = 0.0001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """Initialize Adam optimizer.
        
        Args:
            weight_decay: L2 regularization strength
            beta1: Exponential decay rate for 1st moment
            beta2: Exponential decay rate for 2nd moment
            epsilon: Small constant for numerical stability
        """
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 1st moment estimates
        self.v = {}  # 2nd moment estimates
        self.t = 0  # Time step
    
    def step(
        self,
        splats: List[Splat],
        gradients: Dict[str, Dict[str, np.ndarray]],
        learning_rate: float
    ) -> None:
        """Perform one optimization step.
        
        Args:
            splats: List of splats to update
            gradients: Gradients for each splat parameter
            learning_rate: Learning rate for this step
        """
        self.t += 1
        
        for splat in splats:
            if splat.id not in gradients:
                continue
                
            splat_grads = gradients[splat.id]
            
            # Initialize state for new splats
            if splat.id not in self.m:
                self.m[splat.id] = {
                    "position": np.zeros_like(splat.position),
                    "covariance": np.zeros_like(splat.covariance),
                    "amplitude": 0.0
                }
                self.v[splat.id] = {
                    "position": np.zeros_like(splat.position),
                    "covariance": np.zeros_like(splat.covariance),
                    "amplitude": 0.0
                }
            
            # Prepare update parameters
            splat_params = {}
            
            # Update position if gradient is available
            if "position" in splat_grads:
                grad = splat_grads["position"] + self.weight_decay * splat.position
                
                # Update moments
                self.m[splat.id]["position"] = (
                    self.beta1 * self.m[splat.id]["position"] + (1 - self.beta1) * grad
                )
                self.v[splat.id]["position"] = (
                    self.beta2 * self.v[splat.id]["position"] + (1 - self.beta2) * (grad ** 2)
                )
                
                # Bias-corrected moments
                m_hat = self.m[splat.id]["position"] / (1 - self.beta1 ** self.t)
                v_hat = self.v[splat.id]["position"] / (1 - self.beta2 ** self.t)
                
                # Update position
                new_position = splat.position - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                splat_params["position"] = new_position
            
            # Update covariance if gradient is available
            if "covariance" in splat_grads:
                grad = splat_grads["covariance"] + self.weight_decay * splat.covariance
                
                # Update moments
                self.m[splat.id]["covariance"] = (
                    self.beta1 * self.m[splat.id]["covariance"] + (1 - self.beta1) * grad
                )
                self.v[splat.id]["covariance"] = (
                    self.beta2 * self.v[splat.id]["covariance"] + (1 - self.beta2) * (grad ** 2)
                )
                
                # Bias-corrected moments
                m_hat = self.m[splat.id]["covariance"] / (1 - self.beta1 ** self.t)
                v_hat = self.v[splat.id]["covariance"] / (1 - self.beta2 ** self.t)
                
                # Update covariance
                new_covariance = splat.covariance - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                splat_params["covariance"] = new_covariance
            
            # Update amplitude if gradient is available
            if "amplitude" in splat_grads:
                grad = splat_grads["amplitude"] + self.weight_decay * splat.amplitude
                
                # Update moments
                self.m[splat.id]["amplitude"] = (
                    self.beta1 * self.m[splat.id]["amplitude"] + (1 - self.beta1) * grad
                )
                self.v[splat.id]["amplitude"] = (
                    self.beta2 * self.v[splat.id]["amplitude"] + (1 - self.beta2) * (grad ** 2)
                )
                
                # Bias-corrected moments
                m_hat = self.m[splat.id]["amplitude"] / (1 - self.beta1 ** self.t)
                v_hat = self.v[splat.id]["amplitude"] / (1 - self.beta2 ** self.t)
                
                # Update amplitude
                new_amplitude = splat.amplitude - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                new_amplitude = max(0.01, new_amplitude)  # Ensure positive amplitude
                splat_params["amplitude"] = new_amplitude
            
            # Apply the updates
            if splat_params:
                splat.update_parameters(**splat_params)


class OptimizerFactory:
    """Factory for creating splat parameter optimizers."""
    
    @staticmethod
    def create_optimizer(optimizer_type: str, **kwargs) -> SplatOptimizer:
        """Create an optimizer of the specified type.
        
        Args:
            optimizer_type: Type of optimizer to create (sgd, adam, etc.)
            **kwargs: Additional arguments for the specific optimizer type
            
        Returns:
            Optimizer instance
            
        Raises:
            ValueError: If optimizer_type is invalid
        """
        if optimizer_type == "sgd":
            return SplatSGD(
                weight_decay=kwargs.get("weight_decay", 0.0001),
                momentum=kwargs.get("momentum", 0.9)
            )
        elif optimizer_type == "adam":
            return SplatAdam(
                weight_decay=kwargs.get("weight_decay", 0.0001),
                beta1=kwargs.get("beta1", 0.9),
                beta2=kwargs.get("beta2", 0.999),
                epsilon=kwargs.get("epsilon", 1e-8)
            )
        else:
            raise ValueError(f"Invalid optimizer type: {optimizer_type}")
