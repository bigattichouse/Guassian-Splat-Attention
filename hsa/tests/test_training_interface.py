"""
Tests for the training interface in Hierarchical Splat Attention.

This module contains tests for the HSATrainer, HSATrainingFramework,
and related classes for training HSA models.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from hsa.training_interface import (
    HSATrainer,
    HSATrainingFramework
)
from hsa.training_utils import TrainingConfig
from hsa.adaptation_types import AdaptationConfig
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa.attention_interface import AttentionComputer
from hsa.loss_functions import LossFunction, MSELoss, KLDivergenceLoss, CombinedLoss
from hsa.dense_attention import DenseAttentionComputer


class MockLossFunction(LossFunction):
    """Mock loss function for testing."""
    
    def compute_loss(self, predicted_attention, target_attention):
        """Return mock loss and gradients."""
        # Just return a scalar loss and gradient of same shape as input
        return 0.1, np.ones_like(predicted_attention)


class TestHSATrainer:
    """Tests for the HSATrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create registry and attention computer
        self.hierarchy = Hierarchy(levels=["token", "document"])
        self.registry = SplatRegistry(self.hierarchy, embedding_dim=4)
        self.attention_computer = MagicMock(spec=AttentionComputer)
        
        # Create test splats
        for i in range(3):
            splat = Splat(
                dim=4,
                position=np.random.rand(4),
                covariance=np.eye(4) * 0.5,
                amplitude=1.0,
                level="token" if i < 2 else "document"
            )
            self.registry.register(splat)
        
        # Create training config
        self.config = TrainingConfig(
            learning_rate=0.01,
            batch_size=2,
            num_epochs=2,
            early_stopping_patience=2,
            validation_interval=1
        )
        
        # Mock the gradient computer
        self.mock_gradient_computer = MagicMock()
        self.mock_gradient_computer.compute_gradients.return_value = {
            splat.id: {
                "position": np.ones(4),
                "covariance": np.eye(4),
                "amplitude": np.array(0.1)
            } for splat in self.registry.get_all_splats()
        }
        
        # Create mock optimizer
        self.mock_optimizer = MagicMock()
        
        # Create mock scheduler
        self.mock_scheduler = MagicMock()
        self.mock_scheduler.get_lr.return_value = 0.01
        
        # Mock loss function
        self.mock_loss_function = MockLossFunction()
        
        # Create trainer with mocks (disable adaptation for testing)
        # First, create the trainer with normal dependencies
        with patch('hsa.optimizers.OptimizerFactory.create_optimizer', return_value=self.mock_optimizer), \
             patch('hsa.learning_rate.SchedulerFactory.create_scheduler', return_value=self.mock_scheduler):
            self.trainer = HSATrainer(
                registry=self.registry,
                attention_computer=self.attention_computer,
                config=self.config,
                adaptation_enabled=False
            )
            
        # Then directly set the mocked gradient computer and loss function
        self.trainer.gradient_computer = self.mock_gradient_computer
        self.trainer.loss_function = self.mock_loss_function
        
        # Mock attention computation to return something with correct shape
        self.attention_computer.compute_attention.return_value = np.random.rand(5, 5)
        
        # Create test data
        self.tokens = np.random.rand(5, 4)  # 5 tokens with dim 4
        self.target_attention = np.random.rand(5, 5)  # 5x5 attention matrix
    
    def test_init(self):
        """Test initialization of the trainer."""
        # Check that the trainer was initialized correctly
        assert self.trainer.registry is self.registry
        assert self.trainer.attention_computer is self.attention_computer
        assert self.trainer.config is self.config
        assert self.trainer.adaptation_enabled is False
        
        # Check that components were created
        assert self.trainer.optimizer is self.mock_optimizer
        assert self.trainer.scheduler is self.mock_scheduler
        assert self.trainer.gradient_computer is self.mock_gradient_computer
        assert isinstance(self.trainer.loss_function, MockLossFunction)
        assert self.trainer.adaptation_controller is None  # disabled
        
        # Check initial state
        assert self.trainer.current_step == 0
        assert self.trainer.current_epoch == 0
        assert self.trainer.best_loss == float('inf')
        assert self.trainer.steps_without_improvement == 0
        assert "loss" in self.trainer.training_history
        assert "val_loss" in self.trainer.training_history
        assert "lr" in self.trainer.training_history
    
    def test_train_step(self):
        """Test a single training step."""
        # Perform a training step
        metrics = self.trainer.train_step(self.tokens, self.target_attention)
        
        # Check that the trainer's state was updated
        assert self.trainer.current_step == 1
        assert len(self.trainer.training_history["loss"]) == 1
        assert len(self.trainer.training_history["lr"]) == 1
        
        # Check that the returned metrics have the expected keys
        assert "loss" in metrics
        assert "lr" in metrics
        
        # Check that necessary methods were called
        self.attention_computer.compute_attention.assert_called_once()
        self.mock_gradient_computer.compute_gradients.assert_called_once()
        self.mock_optimizer.step.assert_called_once()
    
    @patch('hsa.loss_functions.MSELoss.compute_loss')
    def test_validate(self, mock_compute_loss):
        """Test validation."""
        # Make compute_loss return a scalar loss and gradient
        mock_compute_loss.return_value = (0.1, np.ones((5, 5)))
        
        # Create validation data
        val_data = [
            (np.random.rand(5, 4), np.random.rand(5, 5)),
            (np.random.rand(5, 4), np.random.rand(5, 5))
        ]
        
        # Perform validation
        metrics = self.trainer.validate(val_data)
        
        # Check that validation metrics were returned
        assert "val_loss" in metrics
        assert "steps_without_improvement" in metrics
        
        # Check that training history was updated
        assert len(self.trainer.training_history["val_loss"]) == 1
        
        # Check that the attention computer was called for each validation sample
        assert self.attention_computer.compute_attention.call_count == 2
    
    def test_should_stop_early(self):
        """Test early stopping check."""
        # Initially should not stop
        assert self.trainer.should_stop_early() is False
        
        # Simulate some steps without improvement
        self.trainer.steps_without_improvement = 1
        assert self.trainer.should_stop_early() is False
        
        # Simulate enough steps without improvement to trigger early stopping
        self.trainer.steps_without_improvement = self.config.early_stopping_patience
        assert self.trainer.should_stop_early() is True
    
    @patch('hsa.training_interface.HSATrainer.train_step')
    @patch('hsa.training_interface.HSATrainer.validate')
    def test_train(self, mock_validate, mock_train_step):
        """Test the full training loop."""
        # Mock train_step and validate to avoid issues
        mock_train_step.return_value = {"loss": 0.1, "lr": 0.01}
        mock_validate.return_value = {"val_loss": 0.2, "steps_without_improvement": 0}
        
        # Create training data - matching expected shape
        train_data = [
            (np.random.rand(5, 4), np.random.rand(5, 5)),
            (np.random.rand(5, 4), np.random.rand(5, 5)),
            (np.random.rand(5, 4), np.random.rand(5, 5)),
            (np.random.rand(5, 4), np.random.rand(5, 5))
        ]
        
        # Create validation data
        val_data = [
            (np.random.rand(5, 4), np.random.rand(5, 5)),
            (np.random.rand(5, 4), np.random.rand(5, 5))
        ]
        
        # Create a callback to track calls
        callback_calls = []
        def callback(trainer, metrics):
            callback_calls.append(metrics)
        
        # Train the model
        history = self.trainer.train(train_data, val_data, [callback])
        
        # Check that history was returned
        assert "loss" in history
        assert "val_loss" in history
        assert "lr" in history
        
        # Check that callbacks were called
        assert len(callback_calls) > 0
        
        # Check that train_step and validate were called
        assert mock_train_step.called
        assert mock_validate.called


class TestHSATrainingFramework:
    """Tests for the HSATrainingFramework class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.framework = HSATrainingFramework()
        
        # Create registry
        self.hierarchy = Hierarchy(levels=["token", "document"])
        self.registry = SplatRegistry(self.hierarchy, embedding_dim=4)
        
        # Create test data
        self.tokens = np.random.rand(5, 4)  # 5 tokens with dim 4
        self.std_attention = np.random.rand(5, 5)  # 5x5 attention matrix
    
    @patch("hsa.dense_attention.DenseAttentionComputer")
    @patch("hsa.training_interface.HSATrainer")
    def test_distill_from_standard_attention(self, mock_trainer_class, mock_attention_class):
        """Test distillation from standard attention."""
        # Mock HSATrainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock DenseAttentionComputer
        mock_attention = MagicMock()
        mock_attention_class.return_value = mock_attention
        
        # Call the method
        result = self.framework.distill_from_standard_attention(
            self.std_attention, self.tokens, self.registry
        )
        
        # Check that necessary methods were called
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
        
        # Check that result is the expected registry
        assert result is self.registry
    
    @patch("hsa.dense_attention.DenseAttentionComputer")
    @patch("hsa.training_interface.HSATrainer")
    @patch("hsa.registry.SplatRegistry.initialize_splats")
    def test_distill_with_new_registry(self, mock_initialize, mock_trainer_class, mock_attention_class):
        """Test distillation with a new registry."""
        # Mock HSATrainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock DenseAttentionComputer
        mock_attention = MagicMock()
        mock_attention_class.return_value = mock_attention
        
        # Call the method without passing a registry
        result = self.framework.distill_from_standard_attention(
            self.std_attention, self.tokens
        )
        
        # Check that a registry was created
        assert result is not None
        assert isinstance(result, SplatRegistry)
        
        # Check that necessary methods were called
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_initialize.assert_called_once()
    
    @patch("hsa.dense_attention.DenseAttentionComputer")
    @patch("hsa.training_interface.HSATrainer")
    def test_train_progressive(self, mock_trainer_class, mock_attention_class):
        """Test progressive training."""
        # Mock HSATrainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock DenseAttentionComputer
        mock_attention = MagicMock()
        mock_attention_class.return_value = mock_attention
        
        # Create config
        config = TrainingConfig()
        
        # Call the method
        result = self.framework.train_progressive(
            self.registry, self.tokens, self.std_attention, config
        )
        
        # Check that necessary methods were called
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
        
        # Check that result is the expected registry
        assert result is self.registry


class TestWithRealComponents:
    """Integration tests with real components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create real components
        self.hierarchy = Hierarchy(levels=["token", "document"])
        self.registry = SplatRegistry(self.hierarchy, embedding_dim=4)
        
        # Mock attention computer to avoid implementation details
        self.attention_computer = MagicMock(spec=DenseAttentionComputer)
        self.attention_computer.compute_attention.return_value = np.eye(4)
        
        # Create test splats
        for i in range(3):
            splat = Splat(
                dim=4,
                position=np.random.rand(4),
                covariance=np.eye(4) * 0.5,
                amplitude=1.0,
                level="token" if i < 2 else "document"
            )
            self.registry.register(splat)
        
        # Create training config with minimal iterations
        self.config = TrainingConfig(
            learning_rate=0.01,
            batch_size=2,
            num_epochs=1,
            validation_interval=1
        )
        
        # Create test data
        self.tokens = np.random.rand(4, 4)  # 4 tokens with dim 4
        self.target_attention = np.eye(4)  # 4x4 identity matrix as target
    
    def test_mse_loss_training(self):
        """Test training with MSE loss."""
        # Create real components but patch the gradient computer
        mock_gradient_computer = MagicMock()
        
        # Return properly structured gradients
        mock_gradient_computer.compute_gradients.return_value = {
            splat.id: {
                "position": np.ones(4),
                "covariance": np.eye(4),
                "amplitude": np.array(0.1)  # Scalar numpy array
            } for splat in self.registry.get_all_splats()
        }
        
        # Create a mock optimizer
        mock_optimizer = MagicMock()
        
        # Create trainer and set the mocks directly
        with patch("hsa.optimizers.OptimizerFactory.create_optimizer", return_value=mock_optimizer):
            trainer = HSATrainer(
                registry=self.registry,
                attention_computer=self.attention_computer,
                config=self.config,
                adaptation_enabled=False
            )
        
        # Set the mocks directly on the trainer
        trainer.gradient_computer = mock_gradient_computer
        
        # Mock the loss function to return known values
        mock_loss_function = MagicMock()
        mock_loss_function.compute_loss.return_value = (0.1, np.ones((4, 4)))
        trainer.loss_function = mock_loss_function
        
        # Train for one step
        trainer.train_step(self.tokens, self.target_attention)
        
        # Check that training history was updated
        assert len(trainer.training_history["loss"]) == 1
        assert trainer.current_step == 1
        
        # Check that methods were called
        mock_gradient_computer.compute_gradients.assert_called_once()
        mock_optimizer.step.assert_called_once()
    
    @patch("hsa.training_interface.HSATrainer")
    def test_training_framework_integration(self, mock_trainer_class):
        """Test integration of training framework with real registry and attention computer."""
        # Create training framework
        framework = HSATrainingFramework()
        
        # Mock HSATrainer to avoid actual training
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Perform distillation
        framework.distill_from_standard_attention(
            self.target_attention, self.tokens, self.registry
        )
        
        # Check that trainer was configured correctly
        mock_trainer_class.assert_called_once()
        
        # Check that train was called
        mock_trainer.train.assert_called_once()
        
        # Get training data used
        train_data = mock_trainer.train.call_args[0][0]
        
        # Verify data structure
        assert isinstance(train_data, list)
        assert len(train_data) == 1
        assert len(train_data[0]) == 2
        assert np.array_equal(train_data[0][0], self.tokens)
        assert np.array_equal(train_data[0][1], self.target_attention)
