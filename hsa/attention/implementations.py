"""
Attention implementation module for Hierarchical Splat Attention (HSA).

This module provides concrete implementations of attention computers:
- Dense attention for small sequences
- Sparse attention for efficient computation 
- Spatial attention for very large sequences

Each implementation is optimized for different sequence lengths and computational resources.
"""

import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union, Any
import math
from scipy.sparse import csr_matrix, lil_matrix
import time
from sklearn.neighbors import NearestNeighbors
import torch

# Import from base module
from hsa.attention.base import AttentionComputer, mahalanobis_batch, gauss_kernel_batch, find_relevant_tokens

# Import core data structures
from hsa.data_structures import Splat, Hierarchy, SplatRegistry


class DenseAttentionComputer(AttentionComputer):
    """
    Computes attention using a dense approach, suitable for small sequences.
    
    This implementation computes the full attention matrix without optimizations,
    which is more straightforward but less efficient for large sequences.
    """
    
    def compute_attention(
        self, 
        tokens: np.ndarray, 
        splat_registry: SplatRegistry
    ) -> np.ndarray:
        """
        Compute attention using the dense approach.
        
        Args:
            tokens: Token embeddings of shape [sequence_length, embedding_dim]
            splat_registry: Registry containing all splats
            
        Returns:
            Attention matrix of shape [sequence_length, sequence_length]
        """
        sequence_length = tokens.shape[0]
        attention_matrix = np.zeros((sequence_length, sequence_length))
        
        # Start timing the computation
        start_time = time.time()
        
        # Process each hierarchical level
        for level in self.hierarchy.levels:
            # Check for timeout
            if time.time() - start_time > self.max_computation_time:
                print(f"Warning: Attention computation timeout. Returning partial results.")
                break
                
            level_weight = self.hierarchy.get_level_weight(level)
            level_splats = splat_registry.get_splats_at_level(level)
            
            # Skip if no splats at this level
            if not level_splats:
                continue
            
            # Initialize level contribution
            level_contrib = np.zeros((sequence_length, sequence_length))
            
            # Process each splat
            for splat in level_splats:
                # Extract and convert necessary splat parameters
                pos = splat.position
                cov_inv = splat.covariance_inverse
                amp = splat.amplitude
                
                # Compute token-to-splat distances
                diffs = tokens - pos
                token_dists = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
                
                # Find tokens that fall within the splat's influence
                relevant_tokens = np.where(token_dists < self.max_splat_radius)[0]
                
                # Only compute attention for tokens within the splat's influence
                for i in relevant_tokens:
                    for j in relevant_tokens:
                        # Compute distance between tokens relative to this splat
                        diff = (tokens[i] - tokens[j]) - pos
                        dist = np.sqrt(diff @ cov_inv @ diff)
                        
                        # Compute attention score using Gaussian kernel
                        score = amp * np.exp(-dist**2)
                        
                        # Add to level contribution
                        level_contrib[i, j] += score
            
            # Apply top-k sparsification
            level_contrib = self._apply_topk_sparsity(level_contrib)
            
            # Add weighted contribution to final matrix
            attention_matrix += level_contrib * level_weight
        
        return attention_matrix


class SparseAttentionComputer(AttentionComputer):
    """
    Computes attention using a sparse approach for O(n*k) complexity.
    
    This implementation efficiently computes attention by:
    1. Only tracking non-zero elements
    2. Avoiding computation for token pairs with negligible attention
    3. Using sparse matrix operations
    """
    
    def __init__(self, hierarchy: Hierarchy, sparse_topk: int = 64, max_splat_radius: float = 3.0):
        """
        Initialize the sparse attention computer.
        
        Args:
            hierarchy: Hierarchy configuration for the attention mechanism
            sparse_topk: Number of top attention scores to keep per token
            max_splat_radius: Maximum radius of influence for splats
        """
        super().__init__(hierarchy, sparse_topk, max_splat_radius)
        self.use_cpu_optimizations = not torch.cuda.is_available()
    
    def compute_attention(
        self, 
        tokens: np.ndarray, 
        splat_registry: SplatRegistry
    ) -> np.ndarray:
        """
        Compute attention using sparse optimizations.
        
        Args:
            tokens: Token embeddings of shape [sequence_length, embedding_dim]
            splat_registry: Registry containing all splats
            
        Returns:
            Attention matrix of shape [sequence_length, sequence_length]
        """
        sequence_length = tokens.shape[0]
        
        # For small sequence lengths, use the dense method for simplicity
        if sequence_length <= 64:
            dense_computer = DenseAttentionComputer(
                self.hierarchy, self.sparse_topk, self.max_splat_radius
            )
            return dense_computer.compute_attention(tokens, splat_registry)
        
        # Start timing the computation
        start_time = time.time()
        
        # For larger sequences on CPU, use subsample if too large
        if self.use_cpu_optimizations and sequence_length > 200:
            # Subsample tokens for attention computation
            subsample_size = min(200, sequence_length)
            if subsample_size < sequence_length:
                print(f"CPU optimization: Using {subsample_size}/{sequence_length} tokens for attention")
                # Use evenly spaced tokens for better coverage
                step = sequence_length // subsample_size
                indices = np.arange(0, sequence_length, step)[:subsample_size]
                sub_tokens = tokens[indices]
                
                # Compute on subset
                sub_attention = self._compute_attention_sparse_optimized(sub_tokens, splat_registry)
                
                # Expand to full size (nearest neighbor interpolation)
                attention_matrix = np.zeros((sequence_length, sequence_length))
                for i in range(sequence_length):
                    i_sub = min(i // step, sub_attention.shape[0]-1)
                    for j in range(sequence_length):
                        j_sub = min(j // step, sub_attention.shape[1]-1)
                        attention_matrix[i, j] = sub_attention[i_sub, j_sub]
                
                return attention_matrix
        
        # Create a sparse matrix for the final attention
        # Use lil_matrix for efficient incremental construction
        attention_matrix = lil_matrix((sequence_length, sequence_length), dtype=np.float32)
        
        # Process each hierarchical level
        for level in self.hierarchy.levels:
            # Check for timeout
            if time.time() - start_time > self.max_computation_time:
                print(f"Warning: Attention computation timeout. Returning partial results.")
                break
                
            level_weight = self.hierarchy.get_level_weight(level)
            level_splats = splat_registry.get_splats_at_level(level)
            
            # Skip if no splats at this level
            if not level_splats:
                continue
            
            # Compute the level's contribution to attention
            level_contrib = self._compute_level_attention_sparse(tokens, level_splats)
            
            # Apply top-k sparsification if needed
            if level_contrib.nnz > sequence_length * self.sparse_topk:
                level_contrib = self._apply_topk_sparsity_sparse(level_contrib)
            
            # Add weighted contribution to the final attention matrix
            # Scale the values by level weight
            level_contrib_lil = level_contrib.tolil()
            rows, cols = level_contrib_lil.nonzero()
            for i, j in zip(rows, cols):
                attention_matrix[i, j] += level_contrib_lil[i, j] * level_weight
        
        # Convert to dense format for compatibility with the rest of the code
        return attention_matrix.toarray()
    
    def _compute_attention_sparse_optimized(
        self, 
        tokens: np.ndarray, 
        splat_registry: SplatRegistry
    ) -> np.ndarray:
        """
        Optimized sparse attention computation for a subset of tokens.
        
        Args:
            tokens: Token embeddings
            splat_registry: Registry containing all splats
            
        Returns:
            Attention matrix
        """
        sequence_length = tokens.shape[0]
        attention_matrix = np.zeros((sequence_length, sequence_length))
        
        # Process each hierarchical level
        for level in self.hierarchy.levels:
            level_weight = self.hierarchy.get_level_weight(level)
            level_splats = splat_registry.get_splats_at_level(level)
            
            # Skip if no splats at this level
            if not level_splats:
                continue
            
            # Process each splat - with fewer iterations for CPU
            max_splats = 10 if self.use_cpu_optimizations else len(level_splats)
            processed_count = 0
            
            for splat in level_splats:
                if processed_count >= max_splats:
                    break
                
                # Extract parameters for efficiency
                pos = splat.position
                cov_inv = splat.covariance_inverse
                amp = splat.amplitude
                
                # Calculate differences and distances using vectorized operations
                diffs = tokens - pos
                
                # Use einsum for efficient matrix calculations
                # This computes the Mahalanobis distance between each token and the splat
                distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
                
                # Find all tokens within the influence radius
                relevant_indices = np.where(distances < self.max_splat_radius)[0]
                
                # Limit to at most 50 tokens per splat for CPU efficiency
                if self.use_cpu_optimizations and len(relevant_indices) > 50:
                    # Keep the closest tokens
                    closest_indices = np.argsort(distances[relevant_indices])[:50]
                    relevant_indices = relevant_indices[closest_indices]
                
                # Skip if no relevant tokens
                if len(relevant_indices) == 0:
                    continue
                
                # Get relevant tokens
                relevant_tokens = tokens[relevant_indices]
                
                # For each pair of relevant tokens, compute attention
                for i_idx, i in enumerate(relevant_indices):
                    token_i = tokens[i]
                    for j_idx, j in enumerate(relevant_indices):
                        token_j = tokens[j]
                        
                        # Compute the difference vector
                        diff = (token_i - token_j) - pos
                        
                        # Compute Mahalanobis distance
                        dist = np.sqrt(diff @ cov_inv @ diff)
                        
                        # Compute attention score and add to matrix
                        score = amp * np.exp(-dist**2)
                        if score > 0.01:  # Filter small values
                            attention_matrix[i, j] += score * level_weight
                
                processed_count += 1
            
            # Apply top-k sparsification to each row
            for i in range(sequence_length):
                row = attention_matrix[i]
                if np.sum(row > 0) > self.sparse_topk:
                    # Keep only top-k values
                    threshold = np.partition(row, -self.sparse_topk)[-self.sparse_topk]
                    row[row < threshold] = 0
        
        return attention_matrix
    
    def _compute_level_attention_sparse(
        self, 
        tokens: np.ndarray, 
        splats: Set[Splat]
    ) -> csr_matrix:
        """
        Compute attention contribution for a level using a sparse approach.
        
        Args:
            tokens: Token embeddings [sequence_length, embedding_dim]
            splats: Set of splats at the current level
            
        Returns:
            Sparse attention matrix for this level
        """
        sequence_length = tokens.shape[0]
        
        # Use lil_matrix for efficient incremental construction
        level_contrib = lil_matrix((sequence_length, sequence_length), dtype=np.float32)
        
        # Convert splats to list for indexing
        splat_list = list(splats)
        
        # For CPU, limit the number of splats processed
        if self.use_cpu_optimizations and len(splat_list) > 20:
            # Select a subset of splats
            splat_list = list(np.random.choice(splat_list, size=20, replace=False))
            print(f"CPU optimization: Using 20/{len(splats)} splats for attention")
        
        # First, for each splat, identify tokens within its influence
        splat_to_tokens = {}
        for s_idx, splat in enumerate(splat_list):
            # Extract splat parameters
            pos = splat.position
            cov_inv = splat.covariance_inverse
            
            # Compute differences for all tokens at once
            diffs = tokens - pos
            
            # Efficient Mahalanobis distance calculation using einsum
            distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
            
            # Find tokens within the influence radius
            relevant_indices = np.where(distances < self.max_splat_radius)[0]
            
            # Limit the number of tokens per splat for CPU efficiency
            max_tokens_per_splat = min(50, sequence_length // 4)
            if len(relevant_indices) > max_tokens_per_splat:
                # Sort by distance and take closest
                sorted_indices = np.argsort(distances[relevant_indices])
                relevant_indices = relevant_indices[sorted_indices[:max_tokens_per_splat]]
            
            splat_to_tokens[s_idx] = relevant_indices
        
        # Now compute attention between tokens influenced by the same splat
        for s_idx, splat in enumerate(splat_list):
            # Get tokens influenced by this splat
            relevant_indices = splat_to_tokens[s_idx]
            
            # Skip if no relevant tokens
            if len(relevant_indices) == 0:
                continue
            
            # Extract splat parameters
            pos = splat.position
            cov_inv = splat.covariance_inverse
            amp = splat.amplitude
            
            # Compute attention between all pairs of relevant tokens
            for i_idx in relevant_indices:
                for j_idx in relevant_indices:
                    # Skip if already computed a significant value
                    if level_contrib[i_idx, j_idx] > 0.1:
                        continue
                    
                    # Compute distance between tokens relative to this splat
                    diff = (tokens[i_idx] - tokens[j_idx]) - pos
                    dist = np.sqrt(diff @ cov_inv @ diff)
                    
                    # Compute attention score
                    score = amp * np.exp(-dist**2)
                    
                    # Only store significant attention scores
                    if score > 0.01:
                        level_contrib[i_idx, j_idx] += score
        
        # Convert to CSR format for efficient operations
        return level_contrib.tocsr()


class SpatialAttentionComputer(SparseAttentionComputer):
    """
    Computes attention using spatial indexing for very large sequences.
    
    This implementation adds spatial indexing on top of sparse attention for even
    greater efficiency with extremely long sequences.
    """
    
    def __init__(self, hierarchy: Hierarchy, sparse_topk: int = 64, max_splat_radius: float = 3.0):
        """
        Initialize the spatial attention computer.
        
        Args:
            hierarchy: Hierarchy configuration for the attention mechanism
            sparse_topk: Number of top attention scores to keep per token
            max_splat_radius: Maximum radius of influence for splats
        """
        super().__init__(hierarchy, sparse_topk, max_splat_radius)
        
        # Additional attributes for spatial indexing
        self.token_index = None
        self.splat_indices = {}
    
    def compute_attention(
        self, 
        tokens: np.ndarray, 
        splat_registry: SplatRegistry
    ) -> np.ndarray:
        """
        Compute attention using spatial indexing for very efficient computation.
        
        Args:
            tokens: Token embeddings of shape [sequence_length, embedding_dim]
            splat_registry: Registry containing all splats
            
        Returns:
            Attention matrix of shape [sequence_length, sequence_length]
        """
        sequence_length = tokens.shape[0]
        
        # For small sequences, use the simpler method
        if sequence_length <= 256:
            return super().compute_attention(tokens, splat_registry)
        
        # For CPU with large sequences, use extreme optimization
        if self.use_cpu_optimizations and sequence_length > 500:
            print(f"CPU optimization: Using simplified attention for large sequence ({sequence_length} tokens)")
            return self._compute_simplified_attention(tokens, splat_registry)
        
        # Start timing the computation
        start_time = time.time()
        
        # Build spatial index for tokens (if needed)
        if self.token_index is None or self.token_index.n_samples != len(tokens):
            # For CPU, only build index for a subset
            if self.use_cpu_optimizations and sequence_length > 1000:
                sample_size = min(1000, sequence_length)
                sample_indices = np.random.choice(sequence_length, size=sample_size, replace=False)
                self._build_token_index(tokens[sample_indices])
                print(f"CPU optimization: Built token index with {sample_size}/{sequence_length} tokens")
            else:
                self._build_token_index(tokens)
        
        # Initialize sparse attention matrix
        attention_matrix = lil_matrix((sequence_length, sequence_length), dtype=np.float32)
        
        # Process each hierarchical level
        for level in self.hierarchy.levels:
            # Check for timeout
            if time.time() - start_time > self.max_computation_time:
                print(f"Warning: Attention computation timeout. Returning partial results.")
                break
                
            level_weight = self.hierarchy.get_level_weight(level)
            level_splats = list(splat_registry.get_splats_at_level(level))
            
            # Skip if no splats at this level
            if not level_splats:
                continue
            
            # For CPU with many splats, limit the processing
            if self.use_cpu_optimizations and len(level_splats) > 20:
                # Select a random subset of splats for efficiency
                selected_indices = np.random.choice(len(level_splats), size=20, replace=False)
                level_splats = [level_splats[i] for i in selected_indices]
                print(f"CPU optimization: Processing 20/{len(splat_registry.get_splats_at_level(level))} splats for level {level}")
            
            # Build spatial index for splats at this level
            self._build_splat_index(level_splats, level)
            
            # For each token, find relevant splats and target tokens
            # For CPU, process a subset of tokens
            max_tokens = sequence_length
            if self.use_cpu_optimizations and sequence_length > 200:
                max_tokens = 200
                token_indices = np.random.choice(sequence_length, size=max_tokens, replace=False)
                print(f"CPU optimization: Processing {max_tokens}/{sequence_length} tokens for attention")
            else:
                token_indices = range(sequence_length)
            
            for i in token_indices:
                # Check for timeout
                if time.time() - start_time > self.max_computation_time:
                    print(f"Warning: Attention computation timeout during token processing.")
                    break
                
                # Find splats that influence this token
                relevant_splats = self._find_relevant_splats(tokens[i], level_splats, level)
                
                # Only process if there are relevant splats
                if not relevant_splats:
                    continue
                
                # Find other tokens influenced by these splats
                target_tokens = set()
                for splat, _ in relevant_splats:
                    # Find tokens within this splat's influence
                    tokens_for_splat = self._find_tokens_for_splat(splat, tokens)
                    target_tokens.update(tokens_for_splat)
                
                # Limit the number of target tokens for CPU efficiency
                if self.use_cpu_optimizations and len(target_tokens) > 50:
                    target_tokens = set(list(target_tokens)[:50])
                
                # Compute attention scores for this token to target tokens
                for j in target_tokens:
                    # Skip diagonal if needed
                    if i == j:
                        continue
                    
                    # Compute combined attention from all relevant splats
                    score = 0.0
                    for splat, _ in relevant_splats:
                        # Compute attention score
                        diff = (tokens[i] - tokens[j]) - splat.position
                        dist = np.sqrt(diff @ splat.covariance_inverse @ diff)
                        score += splat.amplitude * np.exp(-dist**2)
                    
                    # Only store significant attention scores
                    if score > 0.01:
                        attention_matrix[i, j] += score * level_weight
        
        # Apply top-k sparsification if needed
        if attention_matrix.nnz > sequence_length * self.sparse_topk:
            attention_matrix = self._apply_topk_sparsity_sparse(attention_matrix)
        
        # Handle case where attention is computed for subset of tokens
        if self.use_cpu_optimizations and max_tokens < sequence_length:
            # Expand computed attention to full size using nearest neighbors
            full_matrix = lil_matrix((sequence_length, sequence_length), dtype=np.float32)
            computed_rows = np.array(list(token_indices))
            
            # For each token, find nearest computed token
            for i in range(sequence_length):
                if i in token_indices:
                    # Copy the row as is
                    full_matrix.data[i] = attention_matrix.data[list(token_indices).index(i)]
                    full_matrix.rows[i] = attention_matrix.rows[list(token_indices).index(i)]
                else:
                    # Find nearest computed token
                    distances = np.abs(computed_rows - i)
                    nearest_idx = computed_rows[np.argmin(distances)]
                    # Copy its row
                    nearest_row_idx = list(token_indices).index(nearest_idx)
                    full_matrix.data[i] = attention_matrix.data[nearest_row_idx]
                    full_matrix.rows[i] = attention_matrix.rows[nearest_row_idx]
            
            attention_matrix = full_matrix
        
        # Return dense matrix for compatibility
        return attention_matrix.toarray()
    
    def _compute_simplified_attention(
        self,
        tokens: np.ndarray,
        splat_registry: SplatRegistry
    ) -> np.ndarray:
        """
        Compute a simplified approximation of attention for very large sequences on CPU.
        
        Args:
            tokens: Token embeddings
            splat_registry: Registry containing all splats
            
        Returns:
            Approximate attention matrix
        """
        sequence_length = tokens.shape[0]
        
        # Create a simplified attention matrix
        # Use a coarse grid approach
        grid_size = min(200, sequence_length)
        grid_step = sequence_length // grid_size
        
        # Sample tokens for the grid
        grid_indices = np.arange(0, sequence_length, grid_step)[:grid_size]
        grid_tokens = tokens[grid_indices]
        
        # Compute attention for the grid
        grid_attention = np.zeros((grid_size, grid_size))
        
        # Process each level
        for level in self.hierarchy.levels:
            level_weight = self.hierarchy.get_level_weight(level)
            level_splats = list(splat_registry.get_splats_at_level(level))
            
            # Skip if no splats
            if not level_splats:
                continue
            
            # Limit splats for efficiency
            max_splats = min(10, len(level_splats))
            selected_splats = level_splats[:max_splats]
            
            # For each splat, compute contribution
            for splat in selected_splats:
                # Extract parameters
                pos = splat.position
                cov_inv = splat.covariance_inverse
                amp = splat.amplitude
                
                # Find relevant grid tokens
                diffs = grid_tokens - pos
                distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
                relevant_indices = np.where(distances < self.max_splat_radius)[0]
                
                # Compute attention for relevant tokens
                for i_idx in relevant_indices:
                    for j_idx in relevant_indices:
                        diff = (grid_tokens[i_idx] - grid_tokens[j_idx]) - pos
                        dist = np.sqrt(diff @ cov_inv @ diff)
                        score = amp * np.exp(-dist**2)
                        
                        if score > 0.01:
                            grid_attention[i_idx, j_idx] += score * level_weight
        
        # Apply top-k sparsification
        for i in range(grid_size):
            row = grid_attention[i]
            if np.sum(row > 0) > self.sparse_topk:
                # Keep only top-k values
                threshold = np.partition(row, -self.sparse_topk)[-self.sparse_topk]
                row[row < threshold] = 0
        
        # Expand to full size
        full_attention = np.zeros((sequence_length, sequence_length))
        
        # Simple nearest neighbor expansion
        for i in range(sequence_length):
            i_grid = min(i // grid_step, grid_size-1)
            for j in range(sequence_length):
                j_grid = min(j // grid_step, grid_size-1)
                full_attention[i, j] = grid_attention[i_grid, j_grid]
        
        return full_attention
    
    def _build_token_index(self, tokens: np.ndarray) -> None:
        """
        Build a spatial index for tokens for efficient nearest neighbor queries.
        
        Args:
            tokens: Token embeddings
        """
        # Only rebuild if not already built or tokens shape changed
        if self.token_index is None or self.token_index.n_samples != len(tokens):
            self.token_index = NearestNeighbors(
                n_neighbors=min(50, len(tokens)),
                algorithm='auto',
                metric='euclidean'
            ).fit(tokens)
    
    def _build_splat_index(self, splats: List[Splat], level: str) -> None:
        """
        Build a spatial index for splats at a specific level.
        
        Args:
            splats: List of splats at this level
            level: Level name
        """
        # Extract splat positions
        positions = np.array([splat.position for splat in splats])
        
        # Only rebuild if needed
        if level not in self.splat_indices or len(positions) != self.splat_indices[level].n_samples:
            self.splat_indices[level] = NearestNeighbors(
                n_neighbors=min(20, len(splats)),
                algorithm='auto',
                metric='euclidean'
            ).fit(positions)
    
    def _find_relevant_splats(
        self, 
        token: np.ndarray, 
        splats: List[Splat],
        level: str
    ) -> List[Tuple[Splat, float]]:
        """
        Find splats that are relevant for a specific token.
        
        Args:
            token: Token embedding
            splats: List of splats to search
            level: Level name
            
        Returns:
            List of (splat, relevance) pairs sorted by relevance
        """
        # For small numbers of splats, check all of them
        if len(splats) < 20:
            relevant_splats = []
            for splat in splats:
                # Compute distance from token to splat center
                diff = token - splat.position
                dist = np.sqrt(diff @ splat.covariance_inverse @ diff)
                
                # Include splat if token is within its influence
                if dist < self.max_splat_radius:
                    relevance = splat.amplitude * np.exp(-dist**2)
                    relevant_splats.append((splat, relevance))
            
            # Sort by relevance (descending)
            relevant_splats.sort(key=lambda x: x[1], reverse=True)
            return relevant_splats
        
        # For many splats, use the spatial index
        if level in self.splat_indices:
            # Find nearest splats
            token_reshaped = token.reshape(1, -1)
            distances, indices = self.splat_indices[level].kneighbors(token_reshaped)
            
            # Convert to list of (splat, relevance) pairs
            relevant_splats = []
            for idx, dist in zip(indices[0], distances[0]):
                splat = splats[idx]
                
                # Refine distance using the splat's covariance
                diff = token - splat.position
                mahal_dist = np.sqrt(diff @ splat.covariance_inverse @ diff)
                
                # Include splat if token is within its influence
                if mahal_dist < self.max_splat_radius:
                    relevance = splat.amplitude * np.exp(-mahal_dist**2)
                    relevant_splats.append((splat, relevance))
            
            # Sort by relevance (descending)
            relevant_splats.sort(key=lambda x: x[1], reverse=True)
            return relevant_splats
        
        # Fallback to checking all splats
        return self._find_relevant_splats_simple(token, splats)
    
    def _find_relevant_splats_simple(
        self, 
        token: np.ndarray, 
        splats: List[Splat]
    ) -> List[Tuple[Splat, float]]:
        """
        Simple implementation of finding relevant splats without using spatial index.
        
        Args:
            token: Token embedding
            splats: List of splats to search
            
        Returns:
            List of (splat, relevance) pairs sorted by relevance
        """
        relevant_splats = []
        for splat in splats:
            # Compute distance from token to splat center
            diff = token - splat.position
            dist = np.sqrt(diff @ splat.covariance_inverse @ diff)
            
            # Include splat if token is within its influence
            if dist < self.max_splat_radius:
                relevance = splat.amplitude * np.exp(-dist**2)
                relevant_splats.append((splat, relevance))
        
        # Sort by relevance (descending)
        relevant_splats.sort(key=lambda x: x[1], reverse=True)
        return relevant_splats
    
    def _find_tokens_for_splat(
        self,
        splat: Splat,
        tokens: np.ndarray
    ) -> List[int]:
        """
        Find tokens that are influenced by a specific splat.
        
        Args:
            splat: The splat
            tokens: Token embeddings
            
        Returns:
            List of token indices
        """
        # Extract splat parameters
        pos = splat.position
        cov_inv = splat.covariance_inverse
        
        # Use spatial index for efficiency
        if self.token_index is not None:
            # First find tokens near the splat's center using Euclidean distance
            pos_reshaped = pos.reshape(1, -1)
            
            # Scale search radius based on splat covariance
            search_radius = self.max_splat_radius * np.sqrt(np.trace(splat.covariance) / tokens.shape[1])
            
            # Query for tokens within the radius
            neighbor_indices = self.token_index.radius_neighbors(
                pos_reshaped, 
                radius=search_radius,
                return_distance=False
            )[0]
            
            # Refine using Mahalanobis distance
            refined_indices = []
            for idx in neighbor_indices:
                diff = tokens[idx] - pos
                mahal_dist = np.sqrt(diff @ cov_inv @ diff)
                
                if mahal_dist < self.max_splat_radius:
                    refined_indices.append(idx)
            
            return refined_indices
        
        # Fallback to checking all tokens
        relevant_tokens = []
        for idx in range(len(tokens)):
            diff = tokens[idx] - pos
            dist = np.sqrt(diff @ cov_inv @ diff)
            
            if dist < self.max_splat_radius:
                relevant_tokens.append(idx)
        
        return relevant_tokens
