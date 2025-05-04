import unittest
import numpy as np
from hsa.learning_rate import (
    LearningRateScheduler, CosineDecayScheduler, StepDecayScheduler,
    ExponentialDecayScheduler, LinearDecayScheduler, SchedulerFactory,
    create_parameter_specific_lr_schedule
)


class TestLearningRateSchedulers(unittest.TestCase):
    """Test learning rate schedulers."""

    def test_cosine_decay_scheduler_basic(self):
        """Test basic functionality of cosine decay scheduler."""
        base_lr = 0.1
        total_steps = 100
        scheduler = CosineDecayScheduler(base_lr=base_lr, total_steps=total_steps)
        
        # Initial learning rate should be base_lr
        self.assertEqual(scheduler.get_lr(0), base_lr)
        
        # Halfway through should be approximately half of base_lr
        mid_lr = scheduler.get_lr(total_steps // 2)
        self.assertAlmostEqual(mid_lr, base_lr * 0.5, delta=0.1)
        
        # Final learning rate should be close to zero
        final_lr = scheduler.get_lr(total_steps)
        self.assertAlmostEqual(final_lr, 0.0, delta=0.01)

    def test_cosine_decay_scheduler_with_warmup(self):
        """Test cosine decay scheduler with warmup."""
        base_lr = 0.1
        total_steps = 100
        warmup_steps = 20
        scheduler = CosineDecayScheduler(
            base_lr=base_lr, 
            total_steps=total_steps,
            warmup_steps=warmup_steps
        )
        
        # Start of warmup should be very low
        self.assertAlmostEqual(scheduler.get_lr(0), 0.0, delta=0.01)
        
        # End of warmup should be base_lr
        self.assertEqual(scheduler.get_lr(warmup_steps), base_lr)
        
        # Final learning rate should be close to zero
        final_lr = scheduler.get_lr(total_steps)
        self.assertAlmostEqual(final_lr, 0.0, delta=0.01)

    def test_cosine_decay_scheduler_with_min_lr(self):
        """Test cosine decay scheduler with minimum learning rate."""
        base_lr = 0.1
        min_lr = 0.01
        total_steps = 100
        scheduler = CosineDecayScheduler(
            base_lr=base_lr, 
            total_steps=total_steps,
            min_lr=min_lr
        )
        
        # Initial learning rate should be base_lr
        self.assertEqual(scheduler.get_lr(0), base_lr)
        
        # Final learning rate should be min_lr
        final_lr = scheduler.get_lr(total_steps)
        self.assertAlmostEqual(final_lr, min_lr, delta=0.001)

    def test_step_decay_scheduler(self):
        """Test step decay scheduler."""
        base_lr = 0.1
        decay_steps = 20
        decay_rate = 0.5
        scheduler = StepDecayScheduler(
            base_lr=base_lr, 
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
        
        # Initial learning rate should be base_lr
        self.assertEqual(scheduler.get_lr(0), base_lr)
        
        # After one decay step, should be base_lr * decay_rate
        self.assertEqual(scheduler.get_lr(decay_steps), base_lr * decay_rate)
        
        # After two decay steps, should be base_lr * decay_rate^2
        self.assertEqual(scheduler.get_lr(2 * decay_steps), base_lr * decay_rate**2)

    def test_step_decay_scheduler_with_warmup(self):
        """Test step decay scheduler with warmup."""
        base_lr = 0.1
        decay_steps = 20
        decay_rate = 0.5
        warmup_steps = 10
        scheduler = StepDecayScheduler(
            base_lr=base_lr, 
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            warmup_steps=warmup_steps
        )
        
        # Start of warmup should be very low
        self.assertAlmostEqual(scheduler.get_lr(0), 0.0, delta=0.01)
        
        # End of warmup should be base_lr
        self.assertEqual(scheduler.get_lr(warmup_steps), base_lr)
        
        # After one decay step (plus warmup), should be base_lr * decay_rate
        self.assertEqual(scheduler.get_lr(warmup_steps + decay_steps), base_lr * decay_rate)

    def test_exponential_decay_scheduler(self):
        """Test exponential decay scheduler."""
        base_lr = 0.1
        decay_rate = 0.9
        decay_steps = 10
        scheduler = ExponentialDecayScheduler(
            base_lr=base_lr, 
            decay_rate=decay_rate,
            decay_steps=decay_steps
        )
        
        # Initial learning rate should be base_lr
        self.assertEqual(scheduler.get_lr(0), base_lr)
        
        # After decay_steps, should be base_lr * decay_rate
        self.assertAlmostEqual(scheduler.get_lr(decay_steps), base_lr * decay_rate, delta=0.001)
        
        # After 2*decay_steps, should be base_lr * decay_rate^2
        self.assertAlmostEqual(scheduler.get_lr(2 * decay_steps), base_lr * decay_rate**2, delta=0.001)

    def test_exponential_decay_scheduler_with_warmup(self):
        """Test exponential decay scheduler with warmup."""
        base_lr = 0.1
        decay_rate = 0.9
        decay_steps = 10
        warmup_steps = 5
        scheduler = ExponentialDecayScheduler(
            base_lr=base_lr, 
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            warmup_steps=warmup_steps
        )
        
        # Start of warmup should be very low
        self.assertAlmostEqual(scheduler.get_lr(0), 0.0, delta=0.01)
        
        # End of warmup should be base_lr
        self.assertEqual(scheduler.get_lr(warmup_steps), base_lr)
        
        # After decay_steps (plus warmup), should take decay into account
        expected_lr = base_lr * decay_rate
        self.assertAlmostEqual(scheduler.get_lr(warmup_steps + decay_steps), expected_lr, delta=0.001)

    def test_linear_decay_scheduler(self):
        """Test linear decay scheduler."""
        base_lr = 0.1
        total_steps = 100
        scheduler = LinearDecayScheduler(base_lr=base_lr, total_steps=total_steps)
        
        # Initial learning rate should be base_lr
        self.assertEqual(scheduler.get_lr(0), base_lr)
        
        # Halfway through should be halfway between base_lr and 0
        self.assertAlmostEqual(scheduler.get_lr(total_steps // 2), base_lr * 0.5, delta=0.001)
        
        # Final learning rate should be 0
        self.assertAlmostEqual(scheduler.get_lr(total_steps), 0.0, delta=0.001)

    def test_linear_decay_scheduler_with_min_lr(self):
        """Test linear decay scheduler with minimum learning rate."""
        base_lr = 0.1
        min_lr = 0.01
        total_steps = 100
        scheduler = LinearDecayScheduler(
            base_lr=base_lr, 
            total_steps=total_steps,
            min_lr=min_lr
        )
        
        # Initial learning rate should be base_lr
        self.assertEqual(scheduler.get_lr(0), base_lr)
        
        # Halfway through should be halfway between base_lr and min_lr
        expected_mid_lr = min_lr + (base_lr - min_lr) * 0.5
        self.assertAlmostEqual(scheduler.get_lr(total_steps // 2), expected_mid_lr, delta=0.001)
        
        # Final learning rate should be min_lr
        self.assertAlmostEqual(scheduler.get_lr(total_steps), min_lr, delta=0.001)

    def test_scheduler_factory_cosine(self):
        """Test scheduler factory for cosine scheduler."""
        scheduler = SchedulerFactory.create_scheduler(
            "cosine", base_lr=0.1, total_steps=100
        )
        self.assertIsInstance(scheduler, CosineDecayScheduler)
        self.assertEqual(scheduler.base_lr, 0.1)
        self.assertEqual(scheduler.total_steps, 100)

    def test_scheduler_factory_step(self):
        """Test scheduler factory for step scheduler."""
        scheduler = SchedulerFactory.create_scheduler(
            "step", base_lr=0.1, total_steps=100, decay_rate=0.5
        )
        self.assertIsInstance(scheduler, StepDecayScheduler)
        self.assertEqual(scheduler.base_lr, 0.1)
        self.assertEqual(scheduler.decay_rate, 0.5)

    def test_scheduler_factory_exponential(self):
        """Test scheduler factory for exponential scheduler."""
        scheduler = SchedulerFactory.create_scheduler(
            "exponential", base_lr=0.1, total_steps=100, decay_rate=0.9
        )
        self.assertIsInstance(scheduler, ExponentialDecayScheduler)
        self.assertEqual(scheduler.base_lr, 0.1)
        self.assertEqual(scheduler.decay_rate, 0.9)

    def test_scheduler_factory_linear(self):
        """Test scheduler factory for linear scheduler."""
        scheduler = SchedulerFactory.create_scheduler(
            "linear", base_lr=0.1, total_steps=100, min_lr=0.01
        )
        self.assertIsInstance(scheduler, LinearDecayScheduler)
        self.assertEqual(scheduler.base_lr, 0.1)
        self.assertEqual(scheduler.min_lr, 0.01)

    def test_scheduler_factory_invalid(self):
        """Test scheduler factory with invalid scheduler type."""
        with self.assertRaises(ValueError):
            SchedulerFactory.create_scheduler(
                "invalid", base_lr=0.1, total_steps=100
            )

    def test_parameter_specific_lr_schedule(self):
        """Test parameter-specific learning rate schedule."""
        base_scheduler = CosineDecayScheduler(base_lr=0.1, total_steps=100)
        parameter_multipliers = {
            "position": 0.5,
            "covariance": 0.2,
            "amplitude": 1.0
        }
        
        param_scheduler = create_parameter_specific_lr_schedule(
            base_scheduler, parameter_multipliers
        )
        
        # Test specific parameters
        self.assertEqual(param_scheduler("position", 0), 0.1 * 0.5)
        self.assertEqual(param_scheduler("covariance", 0), 0.1 * 0.2)
        self.assertEqual(param_scheduler("amplitude", 0), 0.1 * 1.0)
        
        # Test default multiplier for unknown parameter
        self.assertEqual(param_scheduler("unknown", 0), 0.1 * 1.0)


if __name__ == "__main__":
    unittest.main()
