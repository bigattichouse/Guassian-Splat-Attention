"""
Loss function implementations for Hierarchical Splat Attention (HSA).

This module provides loss functions used in training HSA models,
including MSE and KL divergence for attention distillation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class LossFunction(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def compute_loss(
        self, 
        predicted_attention: np.ndarray, 
        target_attention: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute loss and gradients between predicted and target attention.
        
        Args:
            predicted_attention: Predicted attention matrix
            target_attention: Target attention matrix
            
        Returns:
            Tuple of (loss_value, gradient)
        """
        pass


class MSELoss(LossFunction):
    """Mean squared error loss function."""
    
    def compute_loss(
        self, 
        predicted_attention: np.ndarray, 
        target_attention: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute MSE loss and gradients.
        
        Args:
            predicted_attention: Predicted attention matrix
            target_attention: Target attention matrix
            
        Returns:
            Tuple of (loss_value, gradient)
        """
        # Ensure shapes match
        if predicted_attention.shape != target_attention.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted_attention.shape}, " +
                f"target {target_attention.shape}"
            )
        
        # Compute loss
        diff = predicted_attention - target_attention
        loss = np.mean(diff ** 2)
        
        # Compute gradient
        gradient = 2 * diff / np.prod(diff.shape)
        
        return loss, gradient


class KLDivergenceLoss(LossFunction):
    """KL divergence loss function for attention distillation."""
    
    def __init__(self, temperature: float = 1.0, epsilon: float = 1e-8):
        """Initialize KL divergence loss.
        
        Args:
            temperature: Temperature for softening distributions
            epsilon: Small constant for numerical stability
        """
        self.temperature = temperature
        self.epsilon = epsilon
    
    def compute_loss(
        self, 
        predicted_attention: np.ndarray, 
        target_attention: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute KL divergence loss and gradients.
        
        Args:
            predicted_attention: Predicted attention matrix
            target_attention: Target attention matrix
            
        Returns:
            Tuple of (loss_value, gradient)
        """
        # Ensure shapes match
        if predicted_attention.shape != target_attention.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted_attention.shape}, " +
                f"target {target_attention.shape}"
            )
        
        # Apply temperature
        pred_dist = predicted_attention / self.temperature
        target_dist = target_attention / self.temperature
        
        # Apply softmax per row
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        pred_probs = softmax(pred_dist)
        target_probs = softmax(target_dist)
        
        # Add epsilon for numerical stability
        pred_probs = np.maximum(pred_probs, self.epsilon)
        target_probs = np.maximum(target_probs, self.epsilon)
        
        # Normalize (just in case)
        pred_probs = pred_probs / np.sum(pred_probs, axis=1, keepdims=True)
        target_probs = target_probs / np.sum(target_probs, axis=1, keepdims=True)
        
        # Compute KL divergence
        kl_div = np.sum(target_probs * np.log(target_probs / pred_probs), axis=1)
        loss = np.mean(kl_div)
        
        # Compute gradient
        gradient = (pred_probs - target_probs) / (self.temperature * pred_probs.shape[0] * pred_probs.shape[1])
        
        return loss, gradient


class CombinedLoss(LossFunction):
    """Combined loss function for flexible training."""
    
    def __init__(self, losses: List[LossFunction], weights: List[float]):
        """Initialize combined loss.
        
        Args:
            losses: List of loss functions to combine
            weights: Weight for each loss function
            
        Raises:
            ValueError: If lengths of losses and weights don't match
        """
        if len(losses) != len(weights):
            raise ValueError(
                f"Number of losses ({len(losses)}) must match " +
                f"number of weights ({len(weights)})"
            )
        
        self.losses = losses
        self.weights = weights
    
    def compute_loss(
        self, 
        predicted_attention: np.ndarray, 
        target_attention: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute combined loss and gradients.
        
        Args:
            predicted_attention: Predicted attention matrix
            target_attention: Target attention matrix
            
        Returns:
            Tuple of (loss_value, gradient)
        """
        total_loss = 0.0
        total_gradient = None
        
        for loss_fn, weight in zip(self.losses, self.weights):
            loss, gradient = loss_fn.compute_loss(predicted_attention, target_attention)
            
            total_loss += weight * loss
            
            if total_gradient is None:
                total_gradient = weight * gradient
            else:
                total_gradient += weight * gradient
        
        return total_loss, total_gradient


class LossFunctionFactory:
    """Factory for creating loss functions."""
    
    @staticmethod
    def create_loss(loss_type: str, **kwargs) -> LossFunction:
        """Create a loss function of the specified type.
        
        Args:
            loss_type: Type of loss function to create (mse, kl, combined)
            **kwargs: Additional arguments for the specific loss type
            
        Returns:
            Loss function instance
            
        Raises:
            ValueError: If loss_type is invalid
        """
        if loss_type == "mse":
            return MSELoss()
        elif loss_type == "kl":
            return KLDivergenceLoss(
                temperature=kwargs.get("temperature", 1.0),
                epsilon=kwargs.get("epsilon", 1e-8)
            )
        elif loss_type == "combined":
            # Get list of loss types and weights
            loss_types = kwargs.get("loss_types", ["mse"])
            weights = kwargs.get("weights", [1.0])
            
            # Create each loss function
            losses = []
            for lt in loss_types:
                losses.append(LossFunctionFactory.create_loss(lt, **kwargs))
            
            return CombinedLoss(losses, weights)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
