"""
Tests for the training utilities in Hierarchical Splat Attention.

This module contains tests for the training utility classes and functions
including TrainingConfig, SplatGradientComputer, and related components.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from hsa.training_utils import (
    TrainingConfig, 
    SplatGradientComputer, 
    compute_convergence_metrics,
    apply_splat_regularization,
    transfer_from_pretrained,
    compute_splat_gradients
)
from hsa.adaptation_types import AdaptationConfig
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa.attention_interface import AttentionComputer


class TestTrainingConfig:
    """Tests for the TrainingConfig class."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        config = TrainingConfig()
        
        # Check default values
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.num_epochs == 10
        assert config.weight_decay == 0.0001
        assert config.optimizer_type == "adam"
        assert config.scheduler_type == "cosine"
        assert config.gradient_clip == 1.0
        assert config.early_stopping_patience == 5
        assert config.validation_interval == 100
        assert isinstance(config.adaptation_config, AdaptationConfig)
        assert config.use_curriculum is False
        assert config.distillation_temp == 1.0
        assert config.distillation_alpha == 0.5
        assert config.use_mixed_precision is False
        assert config.seed == 42
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        custom_adaptation_config = AdaptationConfig(
            low_activation_threshold=0.005,
            high_activation_threshold=0.9
        )
        
        config = TrainingConfig(
            learning_rate=0.05,
            batch_size=64,
            num_epochs=20,
            weight_decay=0.0005,
            optimizer_type="sgd",
            scheduler_type="step",
            gradient_clip=0.5,
            early_stopping_patience=10,
            validation_interval=200,
            adaptation_config=custom_adaptation_config,
            use_curriculum=True,
            distillation_temp=2.0,
            distillation_alpha=0.8,
            use_mixed_precision=True,
            seed=123
        )
        
        # Check custom values
        assert config.learning_rate == 0.05
        assert config.batch_size == 64
        assert config.num_epochs == 20
        assert config.weight_decay == 0.0005
        assert config.optimizer_type == "sgd"
        assert config.scheduler_type == "step"
        assert config.gradient_clip == 0.5
        assert config.early_stopping_patience == 10
        assert config.validation_interval == 200
        assert config.adaptation_config is custom_adaptation_config
        assert config.adaptation_config.low_activation_threshold == 0.005
        assert config.adaptation_config.high_activation_threshold == 0.9
        assert config.use_curriculum is True
        assert config.distillation_temp == 2.0
        assert config.distillation_alpha == 0.8
        assert config.use_mixed_precision is True
        assert config.seed == 123


class TestSplatGradientComputer:
    """Tests for the SplatGradientComputer class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock registry and attention computer
        self.hierarchy = Hierarchy(levels=["token", "document"])
        self.registry = SplatRegistry(self.hierarchy, embedding_dim=4)
        self.attention_computer = MagicMock(spec=AttentionComputer)
        
        # Create a gradient computer
        self.gradient_computer = SplatGradientComputer(
            self.registry, self.attention_computer
        )
        
        # Create a test splat
        self.splat = Splat(
            dim=4,
            position=np.array([0.1, 0.2, 0.3, 0.4]),
            covariance=np.eye(4) * 0.5,
            amplitude=1.0,
            level="token"
        )
        self.registry.register(self.splat)
        
        # Create test tokens and attention gradient
        self.tokens = np.random.rand(5, 4)  # 5 tokens with dim 4
        self.attention_gradient = np.random.rand(5, 5)  # 5x5 attention matrix
        
    def test_compute_gradients(self):
        """Test computation of gradients for all splats."""
        # Set up mock compute_splat_attention_map
        splat_attention = np.random.rand(5, 5)
        self.attention_computer.compute_splat_attention_map.return_value = splat_attention
        
        # Call compute_gradients
        gradients = self.gradient_computer.compute_gradients(
            self.tokens, self.attention_gradient
        )
        
        # Check that the result is a dictionary
        assert isinstance(gradients, dict)
        assert self.splat.id in gradients
        
        # Check that the gradient dict has the expected parameters
        splat_grads = gradients[self.splat.id]
        assert "position" in splat_grads
        assert "covariance" in splat_grads
        assert "amplitude" in splat_grads
        
        # Check gradient shapes
        assert splat_grads["position"].shape == (4,)
        assert splat_grads["covariance"].shape == (4, 4)
        assert isinstance(splat_grads["amplitude"], (float, np.float64))
    
    def test_compute_splat_gradients(self):
        """Test computation of gradients for a single splat."""
        # Set up mock compute_splat_attention_map
        splat_attention = np.random.rand(5, 5)
        self.attention_computer.compute_splat_attention_map.return_value = splat_attention
        
        # Call _compute_splat_gradients
        splat_grads = self.gradient_computer._compute_splat_gradients(
            self.splat, self.tokens, self.attention_gradient
        )
        
        # Check that the result has the expected parameters
        assert "position" in splat_grads
        assert "covariance" in splat_grads
        assert "amplitude" in splat_grads
        
        # Check gradient shapes
        assert splat_grads["position"].shape == (4,)
        assert splat_grads["covariance"].shape == (4, 4)
        assert isinstance(splat_grads["amplitude"], (float, np.float64))


class TestConvergenceMetrics:
    """Tests for the compute_convergence_metrics function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = Hierarchy(levels=["token", "document"])
        self.registry = SplatRegistry(self.hierarchy, embedding_dim=4)
        
        # Create some test splats
        for i in range(3):
            splat = Splat(
                dim=4,
                position=np.random.rand(4),
                covariance=np.eye(4) * 0.5,
                amplitude=1.0,
                level="token" if i < 2 else "document"
            )
            self.registry.register(splat)
    
    def test_compute_convergence_metrics_with_history(self):
        """Test computation of convergence metrics with training history."""
        # Create a history dict
        history = {
            "loss": [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.19, 0.18, 0.17, 
                     0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.095, 0.09, 0.085, 0.08]
        }
        
        # Compute metrics
        metrics = compute_convergence_metrics(self.registry, history)
        
        # Check that the result is a dictionary with expected keys
        assert isinstance(metrics, dict)
        assert "convergence_score" in metrics
        assert "improvement" in metrics
        assert "level_counts" in metrics
        assert "total_splats" in metrics
        
        # Check values
        assert 0 <= metrics["convergence_score"] <= 1.0
        assert metrics["total_splats"] == 3
        assert metrics["level_counts"]["token"] == 2
        assert metrics["level_counts"]["document"] == 1
    
    def test_compute_convergence_metrics_insufficient_history(self):
        """Test with insufficient history data."""
        # Create a history dict with few entries
        history = {
            "loss": [0.5, 0.4, 0.3]
        }
        
        # Compute metrics
        metrics = compute_convergence_metrics(self.registry, history)
        
        # Check default values for insufficient history
        assert metrics["convergence_score"] == 0.0
        assert metrics["improvement"] == 0.0
        assert metrics["total_splats"] == 3


class TestSplatRegularization:
    """Tests for the apply_splat_regularization function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = Hierarchy(levels=["token", "document"])
        self.registry = SplatRegistry(self.hierarchy, embedding_dim=4)
        
        # Create some test splats
        for i in range(3):
            splat = Splat(
                dim=4,
                position=np.random.rand(4),
                covariance=np.eye(4) * 0.5,
                amplitude=1.0,
                level="token" if i < 2 else "document"
            )
            self.registry.register(splat)
    
    def test_apply_splat_regularization(self):
        """Test application of regularization to splat parameters."""
        # Apply regularization with default alpha
        reg_loss = apply_splat_regularization(self.registry)
        
        # Check that the result is a non-negative float
        assert isinstance(reg_loss, float)
        assert reg_loss >= 0
        
        # Apply regularization with custom alpha
        custom_alpha = 0.05
        reg_loss_custom = apply_splat_regularization(self.registry, alpha=custom_alpha)
        
        # Check that the result scales with alpha
        ratio = reg_loss_custom / reg_loss
        assert np.isclose(ratio, custom_alpha / 0.01, rtol=1e-5)


class TestTransferFromPretrained:
    """Tests for the transfer_from_pretrained function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = Hierarchy(levels=["token", "document"])
        self.source_registry = SplatRegistry(self.hierarchy, embedding_dim=4)
        
        # Create some test splats in source registry
        for i in range(3):
            splat = Splat(
                dim=4,
                position=np.random.rand(4),
                covariance=np.eye(4) * 0.5,
                amplitude=1.0,
                level="token" if i < 2 else "document"
            )
            self.source_registry.register(splat)
            
        # Create a parent-child relationship
        splats = self.source_registry.get_all_splats()
        splats[0].parent = splats[2]  # token splat's parent is document splat
        splats[2].children.add(splats[0])
        
        # Create an empty target registry
        self.target_registry = SplatRegistry(self.hierarchy, embedding_dim=4)
    
    def test_transfer_from_pretrained(self):
        """Test transferring from a pretrained registry to a target registry."""
        # Perform transfer
        updated_registry = transfer_from_pretrained(
            self.source_registry, self.target_registry
        )
        
        # Check that the target registry was updated
        assert updated_registry is self.target_registry
        
        # Check that all splats were transferred
        assert updated_registry.count_splats() == self.source_registry.count_splats()
        
        # Check that level counts match
        assert (updated_registry.count_splats("token") == 
                self.source_registry.count_splats("token"))
        assert (updated_registry.count_splats("document") == 
                self.source_registry.count_splats("document"))
        
        # Check that parent-child relationships were preserved
        new_splats = updated_registry.get_all_splats()
        
        # Find token and document splats in transferred registry
        token_splats = [s for s in new_splats if s.level == "token"]
        doc_splats = [s for s in new_splats if s.level == "document"]
        
        # Check that at least one token splat has a parent
        has_parent = any(s.parent is not None for s in token_splats)
        assert has_parent
        
        # Check that at least one document splat has children
        has_children = any(len(s.children) > 0 for s in doc_splats)
        assert has_children


class TestComputeSplatGradients:
    """Tests for the compute_splat_gradients function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = Hierarchy(levels=["token", "document"])
        self.registry = SplatRegistry(self.hierarchy, embedding_dim=4)
        self.attention_computer = MagicMock(spec=AttentionComputer)
        
        # Create a test splat
        self.splat = Splat(
            dim=4,
            position=np.random.rand(4),
            covariance=np.eye(4) * 0.5,
            amplitude=1.0,
            level="token"
        )
        self.registry.register(self.splat)
        
        # Create test tokens and attention matrices
        self.tokens = np.random.rand(5, 4)  # 5 tokens with dim 4
        self.attention_matrix = np.random.rand(5, 5)  # 5x5 attention matrix
        self.target_matrix = np.random.rand(5, 5)  # 5x5 target matrix
        
        # Mock the compute_splat_attention_map method
        self.attention_computer.compute_splat_attention_map.return_value = np.random.rand(5, 5)
    
    def test_compute_splat_gradients(self):
        """Test computation of gradients for splat parameters."""
        # Compute gradients
        gradients = compute_splat_gradients(
            self.attention_matrix,
            self.target_matrix,
            self.registry,
            self.attention_computer,
            self.tokens
        )
        
        # Check that the result is a dictionary
        assert isinstance(gradients, dict)
        assert self.splat.id in gradients
        
        # Check that the gradient dict has the expected parameters
        splat_grads = gradients[self.splat.id]
        assert "position" in splat_grads
        assert "covariance" in splat_grads
        assert "amplitude" in splat_grads
        
        # Check gradient shapes
        assert splat_grads["position"].shape == (4,)
        assert splat_grads["covariance"].shape == (4, 4)
        assert isinstance(splat_grads["amplitude"], (float, np.float64))
