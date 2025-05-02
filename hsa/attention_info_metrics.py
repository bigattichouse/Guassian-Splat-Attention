"""
Information-theoretic metrics for adaptation in Hierarchical Splat Attention (HSA).

This module provides metrics based on information theory to guide the
adaptation process and evaluate the quality of the attention mechanism.

This is the main interface module that exports components from the more
specialized implementation modules.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import logging

# Import from implementation modules
from .metrics_computation import InfoTheoreticMetricsComputer
from .candidate_evaluation import InfoTheoreticCandidateEvaluator

# Re-export classes for simpler imports by consumers
__all__ = [
    'InfoTheoreticMetricsComputer',
    'InfoTheoreticCandidateEvaluator'
]
