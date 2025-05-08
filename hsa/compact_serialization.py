"""
Compact serialization utilities for Hierarchical Splat Attention (HSA).

This module provides efficient serialization functions that focus on minimal 
file size while preserving essential model state.
"""

import os
import json
import zlib
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union

from .splat import Splat
from .registry import SplatRegistry
from .hierarchy import Hierarchy

logger = logging.getLogger(__name__)


def save_registry_compact(
    registry: SplatRegistry, 
    filepath: str, 
    format: str = "binary", 
    include_history: bool = False,
    compression_level: int = 9
) -> float:
    """Save registry in compact format to minimize file size.
    
    Args:
        registry: SplatRegistry to save
        filepath: Path to save the file
        format: Output format ('binary', 'json', or 'pickle')
        include_history: Whether to include activation history
        compression_level: Compression level (0-9, higher = smaller but slower)
        
    Returns:
        Size of saved file in MB
    """
    # Create compact representation
    data = _create_compact_representation(registry, include_history)
    
    # Add metadata
    from datetime import datetime
    data["_metadata"] = {
        "version": "1.0.0-compact",
        "timestamp": datetime.now().isoformat(),
        "format": "hsa_registry_compact"
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Save in specified format
    if format.lower() == 'json':
        _save_json_format(data, filepath, compression_level)
    elif format.lower() == 'pickle':
        _save_pickle_format(data, filepath, compression_level)
    else:  # Default to binary
        _save_binary_format(data, filepath, compression_level)
    
    # Report file size
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    logger.info(f"Saved compact registry to {filepath} ({size_mb:.2f} MB) using format: {format}")
    
    return size_mb


def _create_compact_representation(registry: SplatRegistry, include_history: bool = False) -> Dict:
    """Create a compact dictionary representation of the registry.
    
    Args:
        registry: SplatRegistry to convert
        include_history: Whether to include activation history
        
    Returns:
        Compact dictionary representation
    """
    data = {
        "dim": registry.embedding_dim,
        "hierarchy": {
            "levels": registry.hierarchy.levels,
            "weights": [float(w) for w in registry.hierarchy.level_weights],
            "counts": [int(c) for c in registry.hierarchy.init_splats_per_level]
        },
        "splats": []
    }
    
    # Process all splats
    for splat in registry.get_all_splats():
        # Create compact splat representation
        splat_data = {
            "id": splat.id,
            "level": splat.level,
            "amp": float(splat.amplitude),
            "lifetime": int(splat.lifetime),
            # Use parent ID reference instead of full object
            "parent": splat.parent.id if splat.parent else None
        }
        
        # Efficient position encoding (direct list)
        splat_data["pos"] = [float(x) for x in splat.position]
        
        # Efficient covariance encoding
        splat_data.update(_encode_covariance_compact(splat.covariance))
        
        # Include activation data
        if include_history:
            # Include full history if requested
            splat_data["act_hist"] = [float(x) for x in splat.activation_history.get_values()]
        else:
            # Just current activation value
            splat_data["act"] = float(splat.get_average_activation())
        
        data["splats"].append(splat_data)
    
    return data


def _encode_covariance_compact(covariance: np.ndarray) -> Dict:
    """Encode covariance matrix in the most compact form possible.
    
    Args:
        covariance: Covariance matrix
        
    Returns:
        Dictionary with compact covariance representation
    """
    # Get diagonal and dimension
    diag = np.diag(covariance)
    dim = len(diag)
    
    # Check if matrix is approximately spherical (all diagonal elements equal)
    if np.allclose(diag, diag[0] * np.ones(dim), rtol=1e-3):
        return {
            "cov_type": "spherical",
            "cov": float(diag[0])
        }
    
    # Check if matrix is approximately diagonal
    if np.allclose(covariance, np.diag(diag), rtol=1e-3):
        return {
            "cov_type": "diagonal",
            "cov": [float(x) for x in diag]
        }
    
    # Check if matrix can be efficiently represented with eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # If most variance is explained by few components, use PCA representation
        total_var = np.sum(eigenvalues)
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Find how many components explain 99% of variance
        cum_var = np.cumsum(eigenvalues) / total_var
        n_components = np.sum(cum_var < 0.99) + 1
        n_components = max(1, min(n_components, dim))
        
        # If we can achieve good compression (less than 50% of full matrix)
        if n_components < dim // 2:
            return {
                "cov_type": "pca",
                "cov_k": int(n_components),
                "cov_vals": [float(x) for x in eigenvalues[:n_components]],
                "cov_vecs": [[float(x) for x in vec] for vec in eigenvectors[:, :n_components].T]
            }
    except np.linalg.LinAlgError:
        # Fall back to full matrix if eigendecomposition fails
        pass
    
    # Default: encode as full matrix (flattened to save on JSON nesting)
    return {
        "cov_type": "full", 
        "cov": [float(x) for x in covariance.flatten()]
    }


def _save_binary_format(data: Dict, filepath: str, compression_level: int):
    """Save data in binary format with compression.
    
    Args:
        data: Data to save
        filepath: Path to save the file
        compression_level: Compression level
    """
    # Convert to JSON string
    json_str = json.dumps(data)
    
    # Compress
    compressed = zlib.compress(json_str.encode('utf-8'), level=compression_level)
    
    # Write to file
    with open(filepath, 'wb') as f:
        f.write(compressed)


def _save_json_format(data: Dict, filepath: str, compression_level: int):
    """Save data in readable JSON format.
    
    Args:
        data: Data to save
        filepath: Path to save the file
        compression_level: Compression level (ignored for JSON)
    """
    # For .json extension, we'll use uncompressed JSON for readability
    # Add .json extension if not present
    if not filepath.endswith('.json'):
        filepath += '.json'
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def _save_pickle_format(data: Dict, filepath: str, compression_level: int):
    """Save data in pickle format with compression.
    
    Args:
        data: Data to save
        filepath: Path to save the file
        compression_level: Compression level
    """
    import pickle
    
    # Add .pkl extension if not present
    if not filepath.endswith('.pkl') and not filepath.endswith('.pickle'):
        filepath += '.pkl'
    
    # Write to file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_registry_compact(filepath: str) -> SplatRegistry:
    """Load registry from compact format file.
    
    Args:
        filepath: Path to the saved file
        
    Returns:
        Loaded SplatRegistry
    """
    # Detect format from extension
    format = 'binary'
    if filepath.endswith('.json'):
        format = 'json'
    elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
        format = 'pickle'
    
    # Load data based on format
    if format == 'json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif format == 'pickle':
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:  # binary
        try:
            with open(filepath, 'rb') as f:
                compressed = f.read()
            json_str = zlib.decompress(compressed).decode('utf-8')
            data = json.loads(json_str)
        except zlib.error:
            # Try without decompression
            with open(filepath, 'rb') as f:
                json_str = f.read().decode('utf-8')
            data = json.loads(json_str)
    
    # Create registry from compact data
    return _create_registry_from_compact(data)


def _create_registry_from_compact(data: Dict) -> SplatRegistry:
    """Create registry from compact representation.
    
    Args:
        data: Compact dictionary representation
        
    Returns:
        Reconstructed SplatRegistry
    """
    # Create hierarchy
    hierarchy = Hierarchy(
        levels=data["hierarchy"]["levels"],
        init_splats_per_level=data["hierarchy"]["counts"],
        level_weights=data["hierarchy"]["weights"]
    )
    
    # Create registry
    registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=data["dim"])
    
    # First pass: create all splats
    splats_by_id = {}
    for splat_data in data["splats"]:
        # Recreate position
        position = np.array(splat_data["pos"], dtype=np.float64)
        
        # Recreate covariance based on type
        covariance = _decode_covariance_compact(splat_data, data["dim"])
        
        # Create splat
        splat = Splat(
            dim=data["dim"],
            position=position,
            covariance=covariance,
            amplitude=splat_data["amp"],
            level=splat_data["level"],
            id=splat_data["id"]
        )
        
        # Set lifetime
        if "lifetime" in splat_data:
            splat.lifetime = splat_data["lifetime"]
        
        # Set activation data
        if "act_hist" in splat_data:
            # Set full history if provided
            for value in splat_data["act_hist"]:
                splat.activation_history.add(value)
        elif "act" in splat_data:
            # Just set average activation
            splat.activation_history.add(splat_data["act"])
        
        # Add to registry and map
        registry.register(splat)
        splats_by_id[splat.id] = splat
    
    # Second pass: establish parent-child relationships
    for splat_data in data["splats"]:
        if splat_data["parent"] and splat_data["parent"] in splats_by_id:
            child = splats_by_id[splat_data["id"]]
            parent = splats_by_id[splat_data["parent"]]
            
            # Set parent-child relationship
            child.parent = parent
            parent.children.add(child)
    
    # Ensure registry integrity
    registry.repair_integrity()
    
    return registry


def _decode_covariance_compact(splat_data: Dict, dim: int) -> np.ndarray:
    """Decode covariance matrix from compact representation.
    
    Args:
        splat_data: Splat data dictionary
        dim: Dimensionality of embedding space
        
    Returns:
        Covariance matrix
    """
    cov_type = splat_data.get("cov_type", "full")
    
    if cov_type == "spherical":
        # Single value for all diagonal elements
        return np.eye(dim) * splat_data["cov"]
    
    elif cov_type == "diagonal":
        # Diagonal matrix
        return np.diag(splat_data["cov"])
    
    elif cov_type == "pca":
        # PCA representation (eigenvalues and eigenvectors)
        k = splat_data["cov_k"]
        eigenvalues = np.array(splat_data["cov_vals"])
        eigenvectors = np.array(splat_data["cov_vecs"])
        
        # Reconstruct covariance
        full_eigenvectors = np.zeros((dim, k))
        for i in range(k):
            full_eigenvectors[:, i] = eigenvectors[i]
        
        # Compute C = V * D * V^T
        return full_eigenvectors @ np.diag(eigenvalues) @ full_eigenvectors.T
    
    else:  # full
        # Full matrix (flattened)
        return np.array(splat_data["cov"]).reshape((dim, dim))
