"""
Configuration file for pytest.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to make imports work
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
