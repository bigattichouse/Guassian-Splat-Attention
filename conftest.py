"""
Common fixtures for HSA tests.
"""

import pytest
import numpy as np

from hsa.splat import Splat
from hsa.hierarchy import Hierarchy
from hsa.registry import SplatRegistry


@pytest.fixture
def simple_hierarchy() -> Hierarchy:
    """Create a simple hierarchy for testing."""
    return Hierarchy(
        levels=["token", "phrase", "sentence"],
        init_splats_per_level=[10, 5, 2],
        level_weights=[0.6, 0.3, 0.1]
    )


@pytest.fixture
def empty_registry(simple_hierarchy) -> SplatRegistry:
    """Create an empty registry for testing."""
    return SplatRegistry(hierarchy=simple_hierarchy, embedding_dim=3)


@pytest.fixture
def sample_splats() -> list:
    """Create a list of sample splats for testing."""
    splats = []
    
    # Create splats at token level
    for i in range(3):
        splat = Splat(
            dim=3,
            position=np.array([float(i), 0.0, 0.0]),
            level="token",
            id=f"token_{i}"
        )
        splats.append(splat)
    
    # Create splats at phrase level
    for i in range(2):
        splat = Splat(
            dim=3,
            position=np.array([float(i), 1.0, 0.0]),
            level="phrase",
            id=f"phrase_{i}"
        )
        splats.append(splat)
    
    # Create splat at sentence level
    splat = Splat(
        dim=3,
        position=np.array([0.0, 0.0, 1.0]),
        level="sentence",
        id="sentence_0"
    )
    splats.append(splat)
    
    return splats


@pytest.fixture
def populated_registry(empty_registry, sample_splats) -> SplatRegistry:
    """Create a populated registry for testing."""
    registry = empty_registry
    for splat in sample_splats:
        registry.register(splat)
    return registry


@pytest.fixture
def sample_tokens() -> np.ndarray:
    """Create sample token embeddings for testing."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
