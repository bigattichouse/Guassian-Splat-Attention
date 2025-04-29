"""
Unit tests for the HSA training module.

These tests verify the training components of Hierarchical Splat Attention:
- TrainingConfig initialization and parameter validation
- HSATrainer initialization and functionality
- Training process with mock data
- Validation and evaluation processes
- Adaptation during training
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory to path to import HSA modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import HSA modules
from hsa.data_structures import Splat, Hierarchy, SplatRegistry
from hsa.initialization import initialize_splats
from hsa.attention import AttentionComputer, SplatAttentionMetrics, create_attention_computer
from hsa.training import TrainingConfig, HSATrainer, train_hsa, evaluate_hsa
from hsa.adaptation import check_adaptation_triggers, perform_adaptations

class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, num_batches=10, batch_size=4, seq_len=16, dim=64):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dim = dim
        
    def __iter__(self):
        self.current_batch = 0
        return self
        
    def __next__(self):
        if self.current_batch < self.num_batches:
            # Generate random tokens and targets
            tokens = np.random.randn(self.batch_size, self.seq_len, self.dim)
            targets = np.random.randn(self.batch_size, self.seq_len, self.dim)
            
            self.current_batch += 1
            return tokens, targets
        else:
            raise StopIteration
            
    def __len__(self):
        return self.num_batches

class TestTrainingConfig(unittest.TestCase):
    """Tests for the TrainingConfig class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        config = TrainingConfig()
        
        # Check default values
        self.assertEqual(config.epochs, 20)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.adaptation_frequency, 5)
        self.assertEqual(config.mitosis_threshold, 0.1)
        self.assertEqual(config.death_threshold, 0.01)
        self.assertEqual(config.early_stopping_patience, 5)
        self.assertTrue(config.enable_adaptation)
        self.assertEqual(config.device, "cpu")
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        custom_config = TrainingConfig(
            epochs=10,
            batch_size=16,
            learning_rate=0.0005,
            adaptation_frequency=3,
            mitosis_threshold=0.2,
            death_threshold=0.02,
            early_stopping_patience=3,
            enable_adaptation=False,
            device="cuda"
        )
        
        # Check custom values
        self.assertEqual(custom_config.epochs, 10)
        self.assertEqual(custom_config.batch_size, 16)
        self.assertEqual(custom_config.learning_rate, 0.0005)
        self.assertEqual(custom_config.adaptation_frequency, 3)
        self.assertEqual(custom_config.mitosis_threshold, 0.2)
        self.assertEqual(custom_config.death_threshold, 0.02)
        self.assertEqual(custom_config.early_stopping_patience, 3)
        self.assertFalse(custom_config.enable_adaptation)
        self.assertEqual(custom_config.device, "cuda")
    
    def test_repr(self):
        """Test string representation."""
        config = TrainingConfig(epochs=5, batch_size=8)
        repr_str = repr(config)
        
        # Check that essential parameters are included in string representation
        self.assertIn("epochs=5", repr_str)
        self.assertIn("batch_size=8", repr_str)
        self.assertIn("adaptation_frequency=", repr_str)
        self.assertIn("enable_adaptation=", repr_str)

class TestHSATrainer(unittest.TestCase):
    """Tests for the HSATrainer class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create hierarchy
        self.hierarchy = Hierarchy(
            levels=["Token", "Phrase"],
            init_splats_per_level=[10, 5],
            level_weights=[0.7, 0.3]
        )
        
        # Create training config
        self.config = TrainingConfig(
            epochs=3,
            batch_size=4,
            learning_rate=0.001,
            adaptation_frequency=2
        )
        
        # Mock loss function
        self.loss_fn = MagicMock(return_value=0.5)
        
        # Mock attention computer
        self.attention_computer = MagicMock()
        self.attention_computer.compute_attention.return_value = np.random.rand(16, 16)
        
        # Mock metrics tracker
        self.metrics_tracker = MagicMock()
        self.metrics_tracker.compute_splat_activation.return_value = 0.5
        self.metrics_tracker.compute_splat_error_contribution.return_value = 0.2
    
    def test_init(self):
        """Test initialization of trainer."""
        trainer = HSATrainer(
            hierarchy=self.hierarchy,
            config=self.config,
            task_loss_fn=self.loss_fn,
            attention_computer=self.attention_computer,
            metrics_tracker=self.metrics_tracker
        )
        
        # Check initialization
        self.assertEqual(trainer.hierarchy, self.hierarchy)
        self.assertEqual(trainer.config, self.config)
        self.assertEqual(trainer.task_loss_fn, self.loss_fn)
        self.assertEqual(trainer.attention_computer, self.attention_computer)
        self.assertEqual(trainer.metrics_tracker, self.metrics_tracker)
        
        # Check initial state
        self.assertEqual(trainer.current_epoch, 0)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.best_validation_loss, float('inf'))
        self.assertEqual(trainer.epochs_without_improvement, 0)
        self.assertEqual(trainer.train_losses, [])
        self.assertEqual(trainer.val_losses, [])
        self.assertEqual(trainer.adaptation_history, [])
    
    @patch('hsa.attention.create_attention_computer')
    def test_create_attention_computer(self, mock_create):
        """Test creation of attention computer."""
        # Create trainer without specifying attention computer
        mock_create.return_value = self.attention_computer
        
        trainer = HSATrainer(
            hierarchy=self.hierarchy,
            config=self.config,
            task_loss_fn=self.loss_fn
        )
        
        # Check that _create_attention_computer method was called
        # and it used the mocked create_attention_computer function
        mock_create.assert_called_once()
        self.assertEqual(trainer.attention_computer, self.attention_computer)
    
    # FIX: Patch the correct path for the imported functions in training.py
    @patch('hsa.training.perform_adaptations')
    @patch('hsa.training.check_adaptation_triggers')
    def test_train_epoch(self, mock_check_triggers, mock_perform_adaptations):
        """Test training for one epoch."""
        # Create mock data loader
        data_loader = MockDataLoader(num_batches=5, batch_size=4, seq_len=16, dim=64)
        
        # Create splat registry
        splat_registry = SplatRegistry(self.hierarchy)
        for i in range(10):
            splat = Splat(
                position=np.random.randn(64),
                covariance=np.eye(64),
                amplitude=1.0,
                level="Token"
            )
            splat_registry.register(splat)
        
        # Create a non-empty tokens array to avoid empty array issues
        non_empty_tokens = np.random.randn(4, 64)  # Batch of 4 tokens
        
        # Set up the mock to return values matching what the test expects
        # Use a splat from the registry for adaptation
        sample_splat = list(splat_registry.splats.values())[0]
        mock_check_triggers.return_value = [("mitosis", sample_splat)]
        
        # Set up the mock to return values matching what the test expects
        # The perform_adaptations function should return a tuple (splat_registry, result)
        mock_adaptation_result = MagicMock()  # Create a mock result object
        mock_adaptation_result.birth_count = 0
        mock_adaptation_result.death_count = 0
        mock_adaptation_result.mitosis_count = 1

        def side_effect(splat_registry, adaptations, tokens):
            # Verify tokens are not empty
            self.assertTrue(tokens.size > 0, "Empty tokens passed to perform_adaptations")
            # Return both the registry and the result object
            return splat_registry, mock_adaptation_result

        mock_perform_adaptations.side_effect = side_effect
        
        # Modify the config to match what the test expects
        # The test expects adaptation_frequency=2 to result in 2 calls
        # to check_adaptation_triggers during 5 batches
        self.config.adaptation_frequency = 2
        
        # Create model and optimizer mocks
        model = MagicMock()
        
        # Mock the output of the model to return non-empty tensors
        model.return_value = np.random.randn(4, 16, 64)  # Non-empty output
        
        optimizer = MagicMock()
        
        # Create a real metrics tracker instead of a mock to avoid type errors
        real_metrics_tracker = SplatAttentionMetrics()
        
        # Set up metrics tracker with float values instead of MagicMock objects
        # This ensures that comparisons like "activation < death_threshold" work
        metrics_dict = {
            "activation": 0.5,  # A real float value
            "error_contribution": 0.2  # A real float value
        }
        
        # Use a MagicMock with specific get_splat_metrics behavior
        mock_metrics_tracker = MagicMock()
        mock_metrics_tracker.get_splat_metrics.return_value = metrics_dict
        
        # Create trainer with proper metrics tracker
        trainer = HSATrainer(
            hierarchy=self.hierarchy,
            config=self.config,
            task_loss_fn=self.loss_fn,
            attention_computer=self.attention_computer,
            metrics_tracker=mock_metrics_tracker
        )
        
        # Reset global_step to ensure we start from 0
        trainer.global_step = 0
        
        # Run one epoch of training
        epoch_metrics = trainer._train_epoch(
            model=model,
            train_loader=data_loader,
            optimizer=optimizer,
            splat_registry=splat_registry
        )
        
        # Check that training ran and returned metrics
        self.assertIn('train_loss', epoch_metrics)
        self.assertIn('adaptations', epoch_metrics)
        
        # Verify that check_triggers was called exactly 2 times
        # (once every 2 steps for 5 batches = 2 calls)
        self.assertEqual(mock_check_triggers.call_count, 2)
    
    def test_validate(self):
        """Test validation process."""
        # Create mock data loader
        data_loader = MockDataLoader(num_batches=3, batch_size=4)
        
        # Create splat registry
        splat_registry = SplatRegistry(self.hierarchy)
        
        # Create model mock
        model = MagicMock()
        
        # Create trainer
        trainer = HSATrainer(
            hierarchy=self.hierarchy,
            config=self.config,
            task_loss_fn=self.loss_fn
        )
        
        # Run validation
        val_metrics = trainer._validate(
            model=model,
            val_loader=data_loader,
            splat_registry=splat_registry
        )
        
        # Check that validation ran and returned metrics
        self.assertIn('val_loss', val_metrics)

class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for training process."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create hierarchy
        self.hierarchy = Hierarchy(
            levels=["Token", "Phrase"],
            init_splats_per_level=[4, 2],
            level_weights=[0.7, 0.3]
        )
        
        # Create sample tokens
        self.tokens = np.random.randn(16, 64)
        
        # Initialize splats
        self.splat_registry = initialize_splats(
            tokens=self.tokens,
            hierarchy_config={
                "levels": self.hierarchy.levels,
                "init_splats_per_level": self.hierarchy.init_splats_per_level,
                "level_weights": self.hierarchy.level_weights
            }
        )
        
        # Mock loss function
        self.loss_fn = MagicMock(return_value=0.5)
        
        # Create model mock
        self.model = MagicMock()
        
        # Create data loaders
        self.train_loader = MockDataLoader(num_batches=3, batch_size=2, seq_len=16, dim=64)
        self.val_loader = MockDataLoader(num_batches=2, batch_size=2, seq_len=16, dim=64)
    
    @patch('hsa.training.HSATrainer')
    def test_train_hsa_function(self, mock_trainer_class):
        """Test train_hsa function."""
        # Setup mock trainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            'model': self.model,
            'splat_registry': self.splat_registry,
            'train_losses': [0.5, 0.4, 0.3],
            'val_losses': [0.6, 0.5, 0.4],
            'best_validation_loss': 0.4
        }
        mock_trainer_class.return_value = mock_trainer
        
        # Call train_hsa
        results = train_hsa(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            hierarchy=self.hierarchy,
            task_loss_fn=self.loss_fn,
            splat_registry=self.splat_registry
        )
        
        # Check that trainer was created and used
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
        
        # Check results
        self.assertEqual(results['model'], self.model)
        self.assertEqual(results['splat_registry'], self.splat_registry)
        self.assertEqual(len(results['train_losses']), 3)
        self.assertEqual(len(results['val_losses']), 3)
        self.assertEqual(results['best_validation_loss'], 0.4)
    
    @patch('hsa.attention.AttentionComputer')
    def test_evaluate_hsa_function(self, mock_attention_computer_class):
        """Test evaluate_hsa function."""
        # Setup mock attention computer
        mock_attention_computer = MagicMock()
        mock_attention_computer.compute_attention.return_value = np.random.rand(16, 16)
        mock_attention_computer_class.return_value = mock_attention_computer
        
        # Call evaluate_hsa
        metrics = evaluate_hsa(
            model=self.model,
            data_loader=self.val_loader,
            splat_registry=self.splat_registry,
            attention_computer=mock_attention_computer,
            task_loss_fn=self.loss_fn
        )
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('attention_sparsity', metrics)
        self.assertIn('level_contributions', metrics)

if __name__ == '__main__':
    unittest.main()
