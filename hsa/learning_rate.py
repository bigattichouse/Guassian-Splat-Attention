"""
Learning rate schedulers for Hierarchical Splat Attention (HSA).

This module provides learning rate schedulers for HSA training to control
the learning rate over time for better convergence.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, Any, Callable


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


class ExponentialDecayScheduler(LearningRateScheduler):
    """Exponential decay learning rate scheduler."""
    
    def __init__(self, base_lr: float, decay_rate: float = 0.96, decay_steps: int = 1000, warmup_steps: int = 0):
        """Initialize exponential decay scheduler.
        
        Args:
            base_lr: Base learning rate
            decay_rate: Exponential decay rate
            decay_steps: Steps per decay
            warmup_steps: Number of warmup steps with linearly increasing LR
        """
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
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
        
        # Exponential decay phase
        adjusted_step = step - self.warmup_steps
        decay_factor = self.decay_rate ** (adjusted_step / self.decay_steps)
        return self.base_lr * decay_factor


class LinearDecayScheduler(LearningRateScheduler):
    """Linear decay learning rate scheduler."""
    
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int = 0, min_lr: float = 0.0):
        """Initialize linear decay scheduler.
        
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
        
        # Linear decay phase
        decay_steps = self.total_steps - self.warmup_steps
        decay_step = min(step - self.warmup_steps, decay_steps)
        
        # Linear function from 1 to 0
        linear_decay = 1.0 - (decay_step / decay_steps)
        
        # Scale and shift to get LR
        return self.min_lr + (self.base_lr - self.min_lr) * linear_decay


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
        elif scheduler_type == "exponential":
            return ExponentialDecayScheduler(
                base_lr=base_lr,
                decay_rate=kwargs.get("decay_rate", 0.96),
                decay_steps=kwargs.get("decay_steps", total_steps // 10),
                warmup_steps=kwargs.get("warmup_steps", 0)
            )
        elif scheduler_type == "linear":
            return LinearDecayScheduler(
                base_lr=base_lr,
                total_steps=total_steps,
                warmup_steps=kwargs.get("warmup_steps", 0),
                min_lr=kwargs.get("min_lr", 0.0)
            )
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")


def create_parameter_specific_lr_schedule(
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
