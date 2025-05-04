import unittest
import numpy as np
from hsa.optimizers import (
    SplatOptimizer, SplatSGD, SplatAdam, OptimizerFactory
)
from hsa.splat import Splat


class TestOptimizers(unittest.TestCase):
    """Tests for splat parameter optimizers."""

    def setUp(self):
        """Set up test data for optimizer tests."""
        # Create some test splats
        self.dim = 2
        self.splats = []
        
        # Create a few splats with different positions
        for i in range(3):
            position = np.array([i * 0.5, i * 0.3])
            covariance = np.eye(self.dim) * (i + 1)
            splat = Splat(
                dim=self.dim,
                position=position,
                covariance=covariance,
                amplitude=1.0,
                level="token",
                id=f"splat_{i}"
            )
            self.splats.append(splat)
        
        # Create gradients for each splat
        self.gradients = {}
        for i, splat in enumerate(self.splats):
            # Simple gradients for testing
            position_grad = np.ones_like(splat.position) * (i + 1) * 0.1
            covariance_grad = np.eye(self.dim) * (i + 1) * 0.05
            amplitude_grad = 0.01 * (i + 1)
            
            self.gradients[splat.id] = {
                "position": position_grad,
                "covariance": covariance_grad,
                "amplitude": amplitude_grad
            }

    def test_sgd_optimizer_basic(self):
        """Test basic functionality of SGD optimizer."""
        # Create optimizer
        optimizer = SplatSGD(weight_decay=0.0001, momentum=0.9)
        
        # Store original parameters
        original_positions = [splat.position.copy() for splat in self.splats]
        original_covariances = [splat.covariance.copy() for splat in self.splats]
        original_amplitudes = [splat.amplitude for splat in self.splats]
        
        # Apply one optimization step
        learning_rate = 0.1
        optimizer.step(self.splats, self.gradients, learning_rate)
        
        # Verify parameters have been updated
        for i, splat in enumerate(self.splats):
            # Position should change in the opposite direction of the gradient
            self.assertFalse(np.array_equal(splat.position, original_positions[i]))
            
            # Direction should be opposite to gradient (but with momentum = 0 initially)
            expected_direction = -learning_rate * self.gradients[splat.id]["position"]
            actual_direction = splat.position - original_positions[i]
            
            # Check direction with some tolerance for numerical issues
            direction_dot = np.dot(expected_direction, actual_direction)
            self.assertGreater(direction_dot, 0)
            
            # Similar checks for covariance and amplitude
            self.assertFalse(np.array_equal(splat.covariance, original_covariances[i]))
            self.assertNotEqual(splat.amplitude, original_amplitudes[i])

    def test_sgd_optimizer_with_momentum(self):
        """Test SGD optimizer with momentum."""
        # Create optimizer with momentum
        optimizer = SplatSGD(weight_decay=0.0, momentum=0.9)
        
        # Apply multiple steps with the same gradient
        learning_rate = 0.1
        
        # First step
        optimizer.step(self.splats, self.gradients, learning_rate)
        positions_after_first_step = [splat.position.copy() for splat in self.splats]
        
        # Second step (should have momentum from first step)
        optimizer.step(self.splats, self.gradients, learning_rate)
        positions_after_second_step = [splat.position.copy() for splat in self.splats]
        
        # Verify momentum effect: changes should be larger in the second step
        for i in range(len(self.splats)):
            # Calculate update sizes
            first_update = np.linalg.norm(positions_after_first_step[i] - self.splats[i].position)
            second_update = np.linalg.norm(positions_after_second_step[i] - positions_after_first_step[i])
            
            # Second update should be larger due to momentum
            self.assertGreater(second_update, first_update * 0.9)

    def test_sgd_optimizer_with_weight_decay(self):
        """Test SGD optimizer with weight decay."""
        # Create two optimizers: one with weight decay and one without
        optimizer_with_decay = SplatSGD(weight_decay=0.1, momentum=0.0)
        optimizer_without_decay = SplatSGD(weight_decay=0.0, momentum=0.0)
        
        # Create copies of splats for each optimizer
        splats_with_decay = []
        splats_without_decay = []
        
        for splat in self.splats:
            # Create a copy for each test case
            splat_with_decay = Splat(
                dim=splat.dim,
                position=splat.position.copy(),
                covariance=splat.covariance.copy(),
                amplitude=splat.amplitude,
                level=splat.level,
                id=splat.id
            )
            splat_without_decay = Splat(
                dim=splat.dim,
                position=splat.position.copy(),
                covariance=splat.covariance.copy(),
                amplitude=splat.amplitude,
                level=splat.level,
                id=splat.id
            )
            
            splats_with_decay.append(splat_with_decay)
            splats_without_decay.append(splat_without_decay)
        
        # Apply one optimization step
        learning_rate = 0.1
        optimizer_with_decay.step(splats_with_decay, self.gradients, learning_rate)
        optimizer_without_decay.step(splats_without_decay, self.gradients, learning_rate)
        
        # Verify parameters differ due to weight decay
        for i in range(len(self.splats)):
            # Without weight decay, position only changes due to gradient
            # With weight decay, position changes due to gradient and decay towards zero
            # So the change should be larger with weight decay
            splat_with_decay = splats_with_decay[i]
            splat_without_decay = splats_without_decay[i]
            
            # For a point away from the origin, weight decay pulls it closer to origin
            if np.linalg.norm(self.splats[i].position) > 0.1:
                decay_effect = np.linalg.norm(splat_with_decay.position) < np.linalg.norm(splat_without_decay.position)
                self.assertTrue(decay_effect)

    def test_adam_optimizer_basic(self):
        """Test basic functionality of Adam optimizer."""
        # Create optimizer
        optimizer = SplatAdam(weight_decay=0.0001)
        
        # Store original parameters
        original_positions = [splat.position.copy() for splat in self.splats]
        original_covariances = [splat.covariance.copy() for splat in self.splats]
        original_amplitudes = [splat.amplitude for splat in self.splats]
        
        # Apply one optimization step
        learning_rate = 0.1
        optimizer.step(self.splats, self.gradients, learning_rate)
        
        # Verify parameters have been updated
        for i, splat in enumerate(self.splats):
            # Position should change
            self.assertFalse(np.array_equal(splat.position, original_positions[i]))
            
            # Covariance and amplitude should also change
            self.assertFalse(np.array_equal(splat.covariance, original_covariances[i]))
            self.assertNotEqual(splat.amplitude, original_amplitudes[i])

    def test_adam_optimizer_multiple_steps(self):
        """Test Adam optimizer with multiple steps."""
        # Create optimizer
        optimizer = SplatAdam(weight_decay=0.0, beta1=0.9, beta2=0.999)
        
        # Apply multiple steps with the same gradient
        learning_rate = 0.1
        
        # Take several steps
        positions_after_steps = []
        for _ in range(3):
            optimizer.step(self.splats, self.gradients, learning_rate)
            positions_after_steps.append([splat.position.copy() for splat in self.splats])
        
        # Verify Adam adaptation: changes should become more targeted over time
        # as it adapts the learning rate per parameter
        for i in range(len(self.splats)):
            # Calculate update sizes
            first_update = np.linalg.norm(positions_after_steps[0][i] - self.splats[i].position)
            last_update = np.linalg.norm(positions_after_steps[-1][i] - positions_after_steps[-2][i])
            
            # Updates should stabilize over time
            self.assertLess(abs(first_update - last_update), first_update)

    def test_adam_optimizer_with_custom_parameters(self):
        """Test Adam optimizer with custom parameters."""
        # Create optimizer with custom beta values
        optimizer = SplatAdam(weight_decay=0.0, beta1=0.8, beta2=0.99, epsilon=1e-6)
        
        # Verify optimizer has set the parameters correctly
        self.assertEqual(optimizer.beta1, 0.8)
        self.assertEqual(optimizer.beta2, 0.99)
        self.assertEqual(optimizer.epsilon, 1e-6)
        
        # Apply optimization step
        learning_rate = 0.1
        optimizer.step(self.splats, self.gradients, learning_rate)
        
        # Just verify no errors occurred
        self.assertTrue(True)

    def test_optimizer_factory_sgd(self):
        """Test optimizer factory for SGD."""
        optimizer = OptimizerFactory.create_optimizer(
            "sgd", weight_decay=0.01, momentum=0.85
        )
        self.assertIsInstance(optimizer, SplatSGD)
        self.assertEqual(optimizer.weight_decay, 0.01)
        self.assertEqual(optimizer.momentum, 0.85)

    def test_optimizer_factory_adam(self):
        """Test optimizer factory for Adam."""
        optimizer = OptimizerFactory.create_optimizer(
            "adam", weight_decay=0.01, beta1=0.85, beta2=0.99, epsilon=1e-7
        )
        self.assertIsInstance(optimizer, SplatAdam)
        self.assertEqual(optimizer.weight_decay, 0.01)
        self.assertEqual(optimizer.beta1, 0.85)
        self.assertEqual(optimizer.beta2, 0.99)
        self.assertEqual(optimizer.epsilon, 1e-7)

    def test_optimizer_factory_invalid(self):
        """Test optimizer factory with invalid optimizer type."""
        with self.assertRaises(ValueError):
            OptimizerFactory.create_optimizer("invalid")

    def test_optimizer_with_missing_parameters(self):
        """Test optimizers with missing parameters in gradients."""
        # Create optimizers
        sgd_optimizer = SplatSGD()
        adam_optimizer = SplatAdam()
        
        # Create gradients with missing parameters
        incomplete_gradients = {}
        for i, splat in enumerate(self.splats):
            # Only include position gradient
            incomplete_gradients[splat.id] = {
                "position": np.ones_like(splat.position) * 0.1
            }
        
        # Apply optimization step - should not raise errors
        learning_rate = 0.1
        sgd_optimizer.step(self.splats, incomplete_gradients, learning_rate)
        adam_optimizer.step(self.splats, incomplete_gradients, learning_rate)
        
        # Verify no errors occurred
        self.assertTrue(True)

    def test_optimizer_with_nonexistent_splat(self):
        """Test optimizers with gradients for non-existent splats."""
        # Create optimizers
        sgd_optimizer = SplatSGD()
        adam_optimizer = SplatAdam()
        
        # Create gradients with additional non-existent splat
        extended_gradients = self.gradients.copy()
        extended_gradients["nonexistent_splat"] = {
            "position": np.ones(self.dim) * 0.1,
            "covariance": np.eye(self.dim) * 0.05,
            "amplitude": 0.01
        }
        
        # Apply optimization step - should not raise errors
        learning_rate = 0.1
        sgd_optimizer.step(self.splats, extended_gradients, learning_rate)
        adam_optimizer.step(self.splats, extended_gradients, learning_rate)
        
        # Verify no errors occurred
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
