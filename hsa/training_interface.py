"""
Core training interfaces for Hierarchical Splat Attention (HSA).

This module defines the base interfaces and classes for training HSA models
and integrating with existing model training frameworks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np

from .splat import Splat
from .registry import SplatRegistry
from .attention_interface import AttentionComputer
from .adaptation_types import AdaptationConfig
from .optimizers import SplatOptimizer, OptimizerFactory
from .loss_functions import LossFunction, LossFunctionFactory, MSELoss, KLDivergenceLoss, CombinedLoss


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


class LearningRateScheduler(ABC):
    """Abstract base class for learning rate schedulers."""
    
    @abstractmethod
    def get_lr(self, step: int) -> float:
        """Get learning rate for the current step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate for this step
        """
        pass


class CosineDecayScheduler(LearningRateScheduler):
    """Cosine decay learning rate scheduler."""
    
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int = 0, min_lr: float = 0.0):
        """Initialize cosine decay scheduler.
        
        Args:
            base_lr: Base learning rate
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps with linearly increasing LR
            min_lr: Minimum learning rate
        """
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for the current step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate for this step
        """
        # Warmup phase
        if step < self.warmup_steps:
            return self.base_lr * (step / max(1, self.warmup_steps))
        
        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        decay_step = min(step - self.warmup_steps, decay_steps)
        
        # Cosine function from 1 to 0
        cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / decay_steps))
        
        # Scale and shift to get LR
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


class StepDecayScheduler(LearningRateScheduler):
    """Step decay learning rate scheduler."""
    
    def __init__(self, base_lr: float, decay_steps: int, decay_rate: float = 0.1, warmup_steps: int = 0):
        """Initialize step decay scheduler.
        
        Args:
            base_lr: Base learning rate
            decay_steps: Steps between decays
            decay_rate: Learning rate decay rate
            warmup_steps: Number of warmup steps with linearly increasing LR
        """
        self.base_lr = base_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for the current step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate for this step
        """
        # Warmup phase
        if step < self.warmup_steps:
            return self.base_lr * (step / max(1, self.warmup_steps))
        
        # Step decay phase
        decay_factor = self.decay_rate ** (step // self.decay_steps)
        return self.base_lr * decay_factor


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(
        scheduler_type: str,
        base_lr: float,
        total_steps: int,
        **kwargs
    ) -> LearningRateScheduler:
        """Create a learning rate scheduler of the specified type.
        
        Args:
            scheduler_type: Type of scheduler to create (cosine, step, etc.)
            base_lr: Base learning rate
            total_steps: Total number of training steps
            **kwargs: Additional arguments for the specific scheduler type
            
        Returns:
            Learning rate scheduler instance
            
        Raises:
            ValueError: If scheduler_type is invalid
        """
        if scheduler_type == "cosine":
            return CosineDecayScheduler(
                base_lr=base_lr,
                total_steps=total_steps,
                warmup_steps=kwargs.get("warmup_steps", 0),
                min_lr=kwargs.get("min_lr", 0.0)
            )
        elif scheduler_type == "step":
            return StepDecayScheduler(
                base_lr=base_lr,
                decay_steps=kwargs.get("decay_steps", total_steps // 3),
                decay_rate=kwargs.get("decay_rate", 0.1),
                warmup_steps=kwargs.get("warmup_steps", 0)
            )
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")


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
    
    def setup_optimizer(self, optimizer_type: str, params: Dict[str, Any]) -> SplatOptimizer:
        """Set up an optimizer for HSA training.
        
        Args:
            optimizer_type: Type of optimizer to use
            params: Optimizer parameters
            
        Returns:
            Configured optimizer
        """
        return OptimizerFactory.create_optimizer(optimizer_type, **params)
    
    def create_parameter_specific_lr_schedule(
        self,
        base_schedule: LearningRateScheduler,
        parameter_multipliers: Dict[str, float]
    ) -> Callable[[str, int], float]:
        """Create a parameter-specific learning rate schedule.
        
        Args:
            base_schedule: Base learning rate scheduler
            parameter_multipliers: Multipliers for specific parameters
            
        Returns:
            Function to get learning rate for a parameter at a specific step
        """
        def get_lr(param_name: str, step: int) -> float:
            base_lr = base_schedule.get_lr(step)
            multiplier = parameter_multipliers.get(param_name, 1.0)
            return base_lr * multiplier
        
        return get_lr
    
    def compute_splat_gradients(
        self,
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
    
    def compute_convergence_metrics(
        self,
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
        self,
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
        self,
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
