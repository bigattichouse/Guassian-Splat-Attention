"""
Failure type definitions for Hierarchical Splat Attention (HSA).

This module provides the enumeration of failure types and basic structures
for failure detection in HSA.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Union


class FailureType(Enum):
    """Types of failures that can be detected in HSA."""
    NUMERICAL_INSTABILITY = auto()     # Numerical issues in computation
    EMPTY_LEVEL = auto()               # Hierarchy level with no splats
    ORPHANED_SPLAT = auto()            # Splat with incorrect parent-child relationship
    ADAPTATION_STAGNATION = auto()     # No useful adaptation occurring
    PATHOLOGICAL_CONFIGURATION = auto() # Problematic splat arrangement
    ATTENTION_COLLAPSE = auto()        # Attention concentrating too much
    INFORMATION_BOTTLENECK = auto()    # Information flow blockage
    MEMORY_OVERFLOW = auto()           # Too many splats or high memory usage
    GRADIENT_INSTABILITY = auto()      # Unstable gradients during training
