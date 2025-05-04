import unittest
import numpy as np
from hsa.loss_functions import (
    LossFunction, MSELoss, KLDivergenceLoss, CombinedLoss,
    LossFunctionFactory
)


class TestLossFunctions(unittest.TestCase):
    """Test loss functions for HSA training."""

    def setUp(self):
        """Set up test data."""
        # Create sample attention matrices
        self.predicted = np.array([
            [0.9, 0.1, 0.0],
            [0.2, 0.7, 0.1],
            [0.0, 0.3, 0.7]
        ])
        
        self.target = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Ensure they have the same shape
        assert self.predicted.shape == self.target.shape

    def test_mse_loss(self):
        """Test mean squared error loss."""
        loss_fn = MSELoss()
        
        # Compute loss and gradient
        loss, gradient = loss_fn.compute_loss(self.predicted, self.target)
        
        # Expected MSE loss: mean((predicted - target)^2)
        expected_loss = np.mean((self.predicted - self.target) ** 2)
        self.assertAlmostEqual(loss, expected_loss, places=6)
        
        # Expected gradient: 2 * (predicted - target) / n_elements
        expected_gradient = 2 * (self.predicted - self.target) / np.prod(self.predicted.shape)
        np.testing.assert_array_almost_equal(gradient, expected_gradient)

    def test_mse_loss_shape_mismatch(self):
        """Test MSE loss with shape mismatch."""
        loss_fn = MSELoss()
        
        # Create matrices with different shapes
        predicted = np.random.rand(3, 4)
        target = np.random.rand(3, 3)
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            loss_fn.compute_loss(predicted, target)

    def test_kl_divergence_loss(self):
        """Test KL divergence loss."""
        loss_fn = KLDivergenceLoss(temperature=1.0)
        
        # Create normalized probability distributions
        predicted = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        target = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])
        
        # Compute loss and gradient
        loss, gradient = loss_fn.compute_loss(predicted, target)
        
        # Expected KL divergence: sum(target * log(target / predicted))
        # This is just a rough approximation for the test
        epsilon = loss_fn.epsilon
        kl_div = np.sum(target * np.log((target + epsilon) / (predicted + epsilon)), axis=1)
        expected_loss = np.mean(kl_div)
        
        # The loss should be positive
        self.assertGreater(loss, 0)
        
        # Gradient shape should match input shape
        self.assertEqual(gradient.shape, predicted.shape)

    def test_kl_divergence_loss_with_temperature(self):
        """Test KL divergence loss with different temperature."""
        # With higher temperature, the loss should be lower
        loss_fn_high_temp = KLDivergenceLoss(temperature=2.0)
        loss_fn_low_temp = KLDivergenceLoss(temperature=0.5)
        
        # Create test data
        predicted = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        target = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])
        
        # Compute losses
        loss_high_temp, _ = loss_fn_high_temp.compute_loss(predicted, target)
        loss_low_temp, _ = loss_fn_low_temp.compute_loss(predicted, target)
        
        # Higher temperature should give more smoothed distributions and lower loss
        self.assertLess(loss_high_temp, loss_low_temp)

    def test_combined_loss(self):
        """Test combined loss function."""
        mse_loss = MSELoss()
        kl_loss = KLDivergenceLoss()
        
        # Create combined loss with equal weights
        combined_loss = CombinedLoss(losses=[mse_loss, kl_loss], weights=[0.5, 0.5])
        
        # Compute individual losses
        mse_loss_val, mse_gradient = mse_loss.compute_loss(self.predicted, self.target)
        kl_loss_val, kl_gradient = kl_loss.compute_loss(self.predicted, self.target)
        
        # Compute combined loss
        combined_loss_val, combined_gradient = combined_loss.compute_loss(self.predicted, self.target)
        
        # Expected combined loss and gradient
        expected_loss = 0.5 * mse_loss_val + 0.5 * kl_loss_val
        expected_gradient = 0.5 * mse_gradient + 0.5 * kl_gradient
        
        # Check loss
        self.assertAlmostEqual(combined_loss_val, expected_loss, places=6)
        
        # Check gradient
        np.testing.assert_array_almost_equal(combined_gradient, expected_gradient)

    def test_combined_loss_unequal_weights(self):
        """Test combined loss function with unequal weights."""
        mse_loss = MSELoss()
        kl_loss = KLDivergenceLoss()
        
        # Create combined loss with unequal weights
        combined_loss = CombinedLoss(losses=[mse_loss, kl_loss], weights=[0.7, 0.3])
        
        # Compute individual losses
        mse_loss_val, mse_gradient = mse_loss.compute_loss(self.predicted, self.target)
        kl_loss_val, kl_gradient = kl_loss.compute_loss(self.predicted, self.target)
        
        # Compute combined loss
        combined_loss_val, combined_gradient = combined_loss.compute_loss(self.predicted, self.target)
        
        # Expected combined loss and gradient
        expected_loss = 0.7 * mse_loss_val + 0.3 * kl_loss_val
        expected_gradient = 0.7 * mse_gradient + 0.3 * kl_gradient
        
        # Check loss
        self.assertAlmostEqual(combined_loss_val, expected_loss, places=6)
        
        # Check gradient
        np.testing.assert_array_almost_equal(combined_gradient, expected_gradient)

    def test_combined_loss_mismatched_lengths(self):
        """Test combined loss with mismatched lengths of losses and weights."""
        losses = [MSELoss(), KLDivergenceLoss()]
        weights = [0.5]  # Only one weight
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            CombinedLoss(losses=losses, weights=weights)

    def test_loss_function_factory_mse(self):
        """Test LossFunctionFactory for MSE loss."""
        loss_fn = LossFunctionFactory.create_loss("mse")
        self.assertIsInstance(loss_fn, MSELoss)

    def test_loss_function_factory_kl(self):
        """Test LossFunctionFactory for KL divergence loss."""
        loss_fn = LossFunctionFactory.create_loss("kl", temperature=2.0, epsilon=1e-9)
        self.assertIsInstance(loss_fn, KLDivergenceLoss)
        self.assertEqual(loss_fn.temperature, 2.0)
        self.assertEqual(loss_fn.epsilon, 1e-9)

    def test_loss_function_factory_combined(self):
        """Test LossFunctionFactory for combined loss."""
        loss_fn = LossFunctionFactory.create_loss(
            "combined", 
            loss_types=["mse", "kl"], 
            weights=[0.7, 0.3],
            temperature=2.0
        )
        self.assertIsInstance(loss_fn, CombinedLoss)
        self.assertEqual(len(loss_fn.losses), 2)
        self.assertIsInstance(loss_fn.losses[0], MSELoss)
        self.assertIsInstance(loss_fn.losses[1], KLDivergenceLoss)
        self.assertEqual(loss_fn.weights, [0.7, 0.3])
        self.assertEqual(loss_fn.losses[1].temperature, 2.0)

    def test_loss_function_factory_invalid(self):
        """Test LossFunctionFactory with invalid loss type."""
        with self.assertRaises(ValueError):
            LossFunctionFactory.create_loss("invalid")


if __name__ == "__main__":
    unittest.main()
