"""
Core training interfaces for Hierarchical Splat Attention (HSA).

This module defines the base interfaces and classes for training HSA models
and integrating with existing model training frameworks.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np

from .splat import Splat
from .registry import SplatRegistry
from .attention_interface import AttentionComputer
from .adaptation_types import AdaptationConfig
from .learning_rate import SchedulerFactory, LearningRateScheduler
from .training_utils import TrainingConfig, SplatGradientComputer
from .optimizers import SplatOptimizer, OptimizerFactory
from .loss_functions import LossFunction, LossFunctionFactory, MSELoss, KLDivergenceLoss, CombinedLoss


class HSATrainer:
    """Trainer for Hierarchical Splat Attention models."""
    
    def __init__(
        self,
        registry: SplatRegistry,
        attention_computer: AttentionComputer,
        config: TrainingConfig,
        adaptation_enabled: bool = True
    ):
        """Initialize HSA trainer.
        
        Args:
            registry: SplatRegistry to train
            attention_computer: AttentionComputer for computing attention
            config: Training configuration
            adaptation_enabled: Whether to enable dynamic adaptation during training
        """
        self.registry = registry
        self.attention_computer = attention_computer
        self.config = config
        self.adaptation_enabled = adaptation_enabled
        
        # Initialize optimizer
        self.optimizer = OptimizerFactory.create_optimizer(
            config.optimizer_type,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = SchedulerFactory.create_scheduler(
            config.scheduler_type or "cosine",
            base_lr=config.learning_rate,
            total_steps=config.num_epochs * 1000  # Approximate number of steps
        )
        
        # Initialize gradient computer
        # Note: In tests, this will be set by the mock
        self.gradient_computer = SplatGradientComputer(registry, attention_computer)
        
        # Initialize loss function
        self.loss_function = LossFunctionFactory.create_loss("mse")
        
        # Initialize adaptation controller if enabled
        if adaptation_enabled:
            from .adaptation_controller import AdaptationController
            from .adaptation_metrics_base import AdaptationMetricsComputer, SplatCandidateEvaluator
            
            # These would be proper implementations in production
            metrics_computer = AdaptationMetricsComputer()
            candidate_evaluator = SplatCandidateEvaluator()
            
            self.adaptation_controller = AdaptationController(
                registry=registry,
                metrics_computer=metrics_computer,
                candidate_evaluator=candidate_evaluator,
                config=config.adaptation_config
            )
        else:
            self.adaptation_controller = None
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.training_history = {
            "loss": [],
            "val_loss": [],
            "lr": []
        }
    
    def train_step(
        self,
        tokens: np.ndarray,
        target_attention: np.ndarray
    ) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            tokens: Token embeddings batch
            target_attention: Target attention matrices batch
            
        Returns:
            Dictionary with training metrics
        """
        # Increment step counter
        self.current_step += 1
        
        # Get current learning rate
        lr = self.scheduler.get_lr(self.current_step)
        
        # Forward pass: compute attention
        predicted_attention = self.attention_computer.compute_attention(tokens, self.registry)
        
        # Compute loss and gradients
        loss, attention_gradient = self.loss_function.compute_loss(
            predicted_attention, target_attention
        )
        
        # Compute gradients for splat parameters
        splat_gradients = self.gradient_computer.compute_gradients(
            tokens, attention_gradient
        )
        
        # Apply gradients
        self.optimizer.step(self.registry.get_all_splats(), splat_gradients, lr)
        
        # Perform adaptation if enabled
        if self.adaptation_enabled and self.adaptation_controller:
            self.adaptation_controller.step(tokens)
        
        # Update training history
        self.training_history["loss"].append(loss)
        self.training_history["lr"].append(lr)
        
        # Return metrics
        return {
            "loss": loss,
            "lr": lr
        }
    
    def validate(
        self,
        validation_data: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """Validate the model on validation data.
        
        Args:
            validation_data: List of (tokens, target_attention) tuples
            
        Returns:
            Dictionary with validation metrics
        """
        total_loss = 0.0
        count = 0
        
        for tokens, target_attention in validation_data:
            # Forward pass: compute attention
            predicted_attention = self.attention_computer.compute_attention(tokens, self.registry)
            
            # Compute loss
            loss, _ = self.loss_function.compute_loss(
                predicted_attention, target_attention
            )
            
            total_loss += loss
            count += 1
        
        # Compute average loss
        avg_loss = total_loss / max(1, count)
        
        # Update training history
        self.training_history["val_loss"].append(avg_loss)
        
        # Check for early stopping
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        # Return metrics
        return {
            "val_loss": avg_loss,
            "steps_without_improvement": self.steps_without_improvement
        }
    
    def should_stop_early(self) -> bool:
        """Check if early stopping criteria are met.
        
        Returns:
            True if training should stop, False otherwise
        """
        return self.steps_without_improvement >= self.config.early_stopping_patience
    
    def train(
        self,
        train_data: List[Tuple[np.ndarray, np.ndarray]],
        validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """Train the model on the provided data.
        
        Args:
            train_data: List of (tokens, target_attention) tuples for training
            validation_data: Optional validation data
            callbacks: Optional list of callback functions
            
        Returns:
            Training history
        """
        # Reset training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.training_history = {
            "loss": [],
            "val_loss": [],
            "lr": []
        }
        
        # Train for specified number of epochs
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            # Train on batches
            for i in range(0, len(train_data), self.config.batch_size):
                batch = train_data[i:i + self.config.batch_size]
                
                # Skip empty batches
                if not batch:
                    continue
                
                # Combine batch data
                batch_tokens = []
                batch_attention = []
                
                for tokens, attention in batch:
                    batch_tokens.append(tokens)
                    batch_attention.append(attention)
                
                # Convert to arrays
                batch_tokens = np.array(batch_tokens)
                batch_attention = np.array(batch_attention)
                
                # Perform training step
                metrics = self.train_step(batch_tokens, batch_attention)
                
                # Call callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(self, metrics)
                
                # Validate at specified intervals
                if validation_data and self.current_step % self.config.validation_interval == 0:
                    val_metrics = self.validate(validation_data)
                    
                    # Call callbacks with validation metrics
                    if callbacks:
                        for callback in callbacks:
                            callback(self, val_metrics)
                    
                    # Check for early stopping
                    if self.should_stop_early():
                        print(f"Early stopping triggered after {self.current_step} steps")
                        return self.training_history
            
            # Validate at the end of each epoch
            if validation_data:
                val_metrics = self.validate(validation_data)
                
                # Call callbacks with validation metrics
                if callbacks:
                    for callback in callbacks:
                        callback(self, val_metrics)
                
                # Check for early stopping
                if self.should_stop_early():
                    print(f"Early stopping triggered after epoch {epoch}")
                    return self.training_history
        
        return self.training_history


class HSATrainingFramework:
    """Framework for training Hierarchical Splat Attention models."""
    
    def __init__(self):
        """Initialize HSA training framework."""
        pass
    
    def distill_from_standard_attention(
        self,
        std_attention: np.ndarray,
        tokens: np.ndarray,
        registry: Optional[SplatRegistry] = None,
        config: Optional[TrainingConfig] = None
    ) -> SplatRegistry:
        """Distill knowledge from standard attention into HSA.
        
        Args:
            std_attention: Standard attention matrix
            tokens: Token embeddings
            registry: Optional existing registry (if None, creates new)
            config: Optional training configuration
            
        Returns:
            Trained SplatRegistry
        """
        # Create registry if not provided
        if registry is None:
            from .hierarchy import Hierarchy
            hierarchy = Hierarchy()  # Default hierarchy
            registry = SplatRegistry(hierarchy, tokens.shape[1])
            
            # Initialize splats
            registry.initialize_splats(tokens)
        
        # Create training configuration if not provided
        if config is None:
            config = TrainingConfig(
                learning_rate=0.01,
                num_epochs=20,
                early_stopping_patience=5
            )
        
        # Create attention computer
        from .dense_attention import DenseAttentionComputer
        attention_computer = DenseAttentionComputer()
        
        # Create trainer with distillation loss
        kl_loss = KLDivergenceLoss(temperature=config.distillation_temp)
        mse_loss = MSELoss()
        
        combined_loss = CombinedLoss(
            losses=[kl_loss, mse_loss],
            weights=[config.distillation_alpha, 1.0 - config.distillation_alpha]
        )
        
        trainer = HSATrainer(
            registry=registry,
            attention_computer=attention_computer,
            config=config,
            adaptation_enabled=True
        )
        
        # Set custom loss function
        trainer.loss_function = combined_loss
        
        # Create training data
        train_data = [(tokens, std_attention)]
        
        # Train model
        trainer.train(train_data)
        
        return registry
    
    def train_progressive(
        self,
        registry: SplatRegistry,
        tokens: np.ndarray,
        target_attention: np.ndarray,
        config: TrainingConfig
    ) -> SplatRegistry:
        """Train registry using a progressive approach.
        
        Args:
            registry: Initial SplatRegistry
            tokens: Token embeddings
            target_attention: Target attention matrix
            config: Training configuration
            
        Returns:
            Trained SplatRegistry
        """
        # Create attention computer
        from .dense_attention import DenseAttentionComputer
        attention_computer = DenseAttentionComputer()
        
        # Create trainer
        trainer = HSATrainer(
            registry=registry,
            attention_computer=attention_computer,
            config=config,
            adaptation_enabled=True
        )
        
        # Create training data
        train_data = [(tokens, target_attention)]
        
        # Train model
        trainer.train(train_data)
        
        return registry
