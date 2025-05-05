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
from hsa.loss_functions import MSELoss, KLDivergenceLoss, CombinedLoss
from hsa.dense_attention import DenseAttentionComputer


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
        
        # Create trainer (disable adaptation for testing)
        self.trainer = HSATrainer(
            registry=self.registry,
            attention_computer=self.attention_computer,
            config=self.config,
            adaptation_enabled=False
        )
        
        # Mock attention computation
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
        assert self.trainer.optimizer is not None
        assert self.trainer.scheduler is not None
        assert self.trainer.gradient_computer is not None
        assert self.trainer.loss_function is not None
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
        # Mock necessary methods to avoid actual computations
        self.trainer.gradient_computer.compute_gradients.return_value = {
            splat.id: {
                "position": np.zeros(4),
                "covariance": np.zeros((4, 4)),
                "amplitude": 0.0
            } for splat in self.registry.get_all_splats()
        }
        
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
        self.trainer.gradient_computer.compute_gradients.assert_called_once()
    
    def test_validate(self):
        """Test validation."""
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
    
    def test_train(self):
        """Test the full training loop."""
        # Create training data
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
        
        # Mock necessary methods to avoid actual computations
        self.trainer.gradient_computer.compute_gradients.return_value = {
            splat.id: {
                "position": np.zeros(4),
                "covariance": np.zeros((4, 4)),
                "amplitude": 0.0
            } for splat in self.registry.get_all_splats()
        }
        
        # Create a callback to track calls
        callback_calls = []
        def callback(trainer, metrics):
            callback_calls.append(metrics)
        
        # Train the model
        history = self.trainer.train(train_data, val_data, [callback])
        
        # Check that training was performed
        assert self.trainer.current_step > 0
        assert self.trainer.current_epoch > 0
        
        # Check that history was returned
        assert "loss" in history
        assert "val_loss" in history
        assert "lr" in history
        
        # Check that callbacks were called
        assert len(callback_calls) > 0


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
    def test_distill_with_new_registry(self, mock_trainer_class, mock_attention_class):
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
        self.attention_computer = DenseAttentionComputer()
        
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
        self.target_attention = np.identity(4)  # 4x4 identity matrix as target
    
    def test_mse_loss_training(self):
        """Test training with MSE loss."""
        # Create trainer with MSE loss
        trainer = HSATrainer(
            registry=self.registry,
            attention_computer=self.attention_computer,
            config=self.config,
            adaptation_enabled=False
        )
        trainer.loss_function = MSELoss()
        
        # Create training data
        train_data = [(self.tokens, self.target_attention)]
        
        # Train for one step
        trainer.train_step(self.tokens, self.target_attention)
        
        # Check that training history was updated
        assert len(trainer.training_history["loss"]) == 1
        assert trainer.current_step == 1
    
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
        assert np.array_equal(train_data[0][1], self.target_attention).train_step(self.tokens, self.target_attention)
        
        # Check that training history was updated
        assert len(trainer.training_history["loss"]) == 1
        assert trainer.current_step == 1
    
    def test_kl_loss_training(self):
        """Test training with KL divergence loss."""
        # Create trainer with KL loss
        trainer = HSATrainer(
            registry=self.registry,
            attention_computer=self.attention_computer,
            config=self.config,
            adaptation_enabled=False
        )
        trainer.loss_function = KLDivergenceLoss(temperature=2.0)
        
        # Train for one step
        trainer.train_step(self.tokens, self.target_attention)
        
        # Check that training history was updated
        assert len(trainer.training_history["loss"]) == 1
        assert trainer.current_step == 1
    
    def test_combined_loss_training(self):
        """Test training with combined loss."""
        # Create trainer with combined loss
        trainer = HSATrainer(
            registry=self.registry,
            attention_computer=self.attention_computer,
            config=self.config,
            adaptation_enabled=False
        )
        trainer.loss_function = CombinedLoss(
            losses=[MSELoss(), KLDivergenceLoss()],
            weights=[0.7, 0.3]
        )
        
        # Train for one step
        trainer
