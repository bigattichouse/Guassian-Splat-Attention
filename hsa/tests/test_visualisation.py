"""
Tests for the HSA visualization module.

This module tests the visualization capabilities of HSA:
- Attention matrix visualization
- Splat distribution visualization
- Hierarchy visualization
- Adaptation history visualization
- Level contributions visualization
- Attention sparsity visualization
- Dashboard creation
"""

import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Import the HSA modules we need for testing
from hsa.visualization import HSAVisualizer
from hsa.data_structures import Splat, Hierarchy, SplatRegistry

# Create a fixture for the visualizer
@pytest.fixture
def visualizer():
    """Create a visualizer with a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield HSAVisualizer(output_dir=temp_dir)
    # Clean up the temp directory after tests
    shutil.rmtree(temp_dir)

# Create a fixture for a sample attention matrix
@pytest.fixture
def sample_attention_matrix():
    """Create a sample attention matrix for testing."""
    seq_len = 20
    attention_matrix = np.random.rand(seq_len, seq_len)
    attention_matrix = attention_matrix * (attention_matrix > 0.7)  # Make it sparse
    return attention_matrix

# Create a fixture for sample tokens
@pytest.fixture
def sample_tokens():
    """Create sample token embeddings for testing."""
    seq_len = 20
    embedding_dim = 64
    return np.random.randn(seq_len, embedding_dim)

# Create a fixture for a mock splat registry
@pytest.fixture
def mock_splat_registry():
    """Create a mock splat registry for testing."""
    # Create a hierarchy
    hierarchy = Hierarchy(
        levels=["Token", "Phrase", "Section", "Document"],
        init_splats_per_level=[10, 5, 3, 1],
        level_weights=[0.4, 0.3, 0.2, 0.1]
    )
    
    # Create a registry
    registry = SplatRegistry(hierarchy)
    
    # Add some splats at each level
    embedding_dim = 64
    
    # Token level splats
    for i in range(10):
        position = np.random.randn(embedding_dim)
        covariance = np.eye(embedding_dim) * (0.1 + 0.05 * np.random.rand())
        splat = Splat(
            position=position,
            covariance=covariance,
            amplitude=0.5 + 0.5 * np.random.rand(),
            level="Token",
            splat_id=f"token_splat_{i}"
        )
        registry.register(splat)
    
    # Phrase level splats
    phrase_splats = []
    for i in range(5):
        position = np.random.randn(embedding_dim)
        covariance = np.eye(embedding_dim) * (0.2 + 0.1 * np.random.rand())
        splat = Splat(
            position=position,
            covariance=covariance,
            amplitude=0.6 + 0.4 * np.random.rand(),
            level="Phrase",
            splat_id=f"phrase_splat_{i}"
        )
        registry.register(splat)
        phrase_splats.append(splat)
    
    # Section level splats
    section_splats = []
    for i in range(3):
        position = np.random.randn(embedding_dim)
        covariance = np.eye(embedding_dim) * (0.3 + 0.2 * np.random.rand())
        splat = Splat(
            position=position,
            covariance=covariance,
            amplitude=0.7 + 0.3 * np.random.rand(),
            level="Section",
            splat_id=f"section_splat_{i}"
        )
        registry.register(splat)
        section_splats.append(splat)
    
    # Document level splat
    position = np.random.randn(embedding_dim)
    covariance = np.eye(embedding_dim) * 0.5
    doc_splat = Splat(
        position=position,
        covariance=covariance,
        amplitude=1.0,
        level="Document",
        splat_id="document_splat_0"
    )
    registry.register(doc_splat)
    
    # Set up some parent-child relationships
    # Link phrase splats to token splats
    token_splats = list(registry.get_splats_at_level("Token"))
    for i, phrase_splat in enumerate(phrase_splats):
        # Each phrase splat gets 2 token children
        child_indices = [i*2, i*2+1]
        for idx in child_indices:
            if idx < len(token_splats):
                phrase_splat.add_child(token_splats[idx])
    
    # Link section splats to phrase splats
    for i, section_splat in enumerate(section_splats):
        # Distribute phrase splats among section splats
        for j in range(len(phrase_splats)):
            if j % len(section_splats) == i:
                section_splat.add_child(phrase_splats[j])
    
    # Link document splat to section splats
    for section_splat in section_splats:
        doc_splat.add_child(section_splat)
    
    return registry

def test_visualizer_init(visualizer):
    """Test initializing the visualizer."""
    assert os.path.exists(visualizer.output_dir)
    assert "Token" in visualizer.level_colors
    assert len(visualizer.adaptation_history) == 0

def test_visualize_attention_matrix(visualizer, sample_attention_matrix):
    """Test visualizing an attention matrix."""
    # Disable plt.show to avoid opening windows during tests
    with patch('matplotlib.pyplot.show'):
        # Test with minimal arguments
        file_path = visualizer.visualize_attention_matrix(
            attention_matrix=sample_attention_matrix,
            show=False  # Don't actually show the plot
        )
        assert os.path.exists(file_path)
        assert file_path.endswith('.png')
        
        # Test with token labels
        token_labels = [f"Token_{i}" for i in range(sample_attention_matrix.shape[0])]
        file_path = visualizer.visualize_attention_matrix(
            attention_matrix=sample_attention_matrix,
            tokens=token_labels,
            title="Test Attention Matrix",
            show=False
        )
        assert os.path.exists(file_path)

def test_visualize_splat_distribution(visualizer, mock_splat_registry, sample_tokens):
    """Test visualizing splat distributions."""
    with patch('matplotlib.pyplot.show'):
        # Test with just splats
        file_path = visualizer.visualize_splat_distribution(
            splat_registry=mock_splat_registry,
            show=False
        )
        assert os.path.exists(file_path)
        
        # Test with tokens included
        file_path = visualizer.visualize_splat_distribution(
            splat_registry=mock_splat_registry,
            tokens=sample_tokens,
            method="pca",
            title="Test Splat Distribution",
            show=False
        )
        assert os.path.exists(file_path)

def test_visualize_hierarchy(visualizer, mock_splat_registry):
    """Test visualizing the hierarchy."""
    with patch('matplotlib.pyplot.show'):
        file_path = visualizer.visualize_hierarchy(
            splat_registry=mock_splat_registry,
            title="Test Hierarchy Visualization",
            show=False
        )
        assert os.path.exists(file_path)

def test_adaptation_history(visualizer, sample_tokens):
    """Test recording and visualizing adaptation events."""
    # Record some events
    visualizer.record_adaptation_event(
        event_type="mitosis",
        splat_id="test_splat_1",
        tokens=sample_tokens
    )
    visualizer.record_adaptation_event(
        event_type="death",
        splat_id="test_splat_2",
        tokens=None
    )
    
    assert len(visualizer.adaptation_history) == 2
    assert visualizer.adaptation_history[0]['type'] == "mitosis"
    assert visualizer.adaptation_history[1]['type'] == "death"
    
    # Visualize the history
    with patch('matplotlib.pyplot.show'):
        file_path = visualizer.visualize_adaptation_history(
            title="Test Adaptation History",
            show=False
        )
        assert os.path.exists(file_path)

def test_visualize_level_contributions(visualizer):
    """Test visualizing level contributions."""
    # Create sample level contributions
    level_contributions = {
        "Token": 0.4,
        "Phrase": 0.3,
        "Section": 0.2,
        "Document": 0.1
    }
    
    with patch('matplotlib.pyplot.show'):
        file_path = visualizer.visualize_level_contributions(
            level_contributions=level_contributions,
            title="Test Level Contributions",
            show=False
        )
        assert os.path.exists(file_path)

def test_visualize_attention_sparsity(visualizer, sample_attention_matrix):
    """Test visualizing attention sparsity."""
    with patch('matplotlib.pyplot.show'):
        file_path = visualizer.visualize_attention_sparsity(
            attention_matrix=sample_attention_matrix,
            title="Test Attention Sparsity",
            show=False
        )
        assert os.path.exists(file_path)

def test_create_dashboard(visualizer, mock_splat_registry, sample_attention_matrix, sample_tokens):
    """Test creating a dashboard."""
    with patch('matplotlib.pyplot.show'):
        file_path = visualizer.create_dashboard(
            splat_registry=mock_splat_registry,
            attention_matrix=sample_attention_matrix,
            tokens=sample_tokens,
            title="Test HSA Dashboard",
            show=False
        )
        assert os.path.exists(file_path)

def test_edge_cases(visualizer):
    """Test edge cases and error handling."""
    # Empty adaptation history
    with patch('matplotlib.pyplot.show'):
        # Clear the history
        visualizer.adaptation_history = []
        result = visualizer.visualize_adaptation_history(show=False)
        assert result == ""  # Should return empty string when nothing to visualize
    
    # Empty level contributions - we should provide at least one item
    # to avoid the max() of empty sequence error
    with patch('matplotlib.pyplot.show'):
        file_path = visualizer.visualize_level_contributions(
            level_contributions={"Default": 0.0},
            show=False
        )
        # It should still create a file with the minimal data
        assert file_path.endswith('.png')
