"""
Gaussian Splat Attention - Core Splat Implementation

This module contains the core Splat class which represents a single Gaussian distribution
in embedding space. Each splat is defined by its position (center), covariance matrix
(shape and orientation), and amplitude (overall influence).

The Splat class provides methods to compute attention between tokens via this splat,
track activation history, and update parameters.
"""

import numpy as np
import torch
from .ring_buffer import RingBuffer
from .numeric_utils import (
    ensure_positive_definite,
    compute_gaussian,
    stable_matrix_inverse,
    bounded_amplitude
)

class Splat:
    """
    Represents a single Gaussian 'splat' in embedding space.
    
    A splat is defined by:
    - Position: Center in embedding space
    - Covariance: Determines shape and orientation
    - Amplitude: Controls overall influence
    - Activation: Tracks recent usage/importance
    """
    
    def __init__(self, 
                 position, 
                 covariance, 
                 amplitude=1.0, 
                 activation_buffer_size=10, 
                 splat_id=None):
        """
        Initialize a new Splat.
        
        Args:
            position (torch.Tensor): Center of the Gaussian in embedding space
                                    Shape: [embedding_dim]
            covariance (torch.Tensor): Covariance matrix that defines the shape
                                      Shape: [embedding_dim, embedding_dim]
            amplitude (float): Scalar that controls the overall influence
            activation_buffer_size (int): Size of the activation history buffer
            splat_id (str, optional): Unique identifier for this splat
        """
        # Validate and store parameters
        self.embedding_dim = position.shape[0]
        
        # Ensure position is correctly shaped
        if position.dim() != 1:
            raise ValueError(f"Position must be a 1D tensor, got shape {position.shape}")
        self.position = position
        
        # Ensure covariance is correctly shaped and positive definite
        if covariance.shape != (self.embedding_dim, self.embedding_dim):
            raise ValueError(f"Covariance must be a square matrix of size {self.embedding_dim}, got {covariance.shape}")
        
        # Ensure the covariance matrix is positive definite
        self.covariance = ensure_positive_definite(covariance)
        
        # Calculate inverse covariance matrix (precision matrix) for faster computation
        self.precision = stable_matrix_inverse(self.covariance)
        
        # Set and bound the amplitude
        self.amplitude = bounded_amplitude(amplitude)
        
        # Initialize activation history
        self.activation_history = RingBuffer(capacity=activation_buffer_size)
        
        # Assign ID if provided, otherwise use a random UUID
        self.id = splat_id if splat_id else str(np.random.randint(0, 1000000))
        
        # Metadata for tracking
        self.creation_time = 0
        self.last_update_time = 0
        self.birth_embedding = position.clone()  # Keep original position for analysis
    
    def compute_attention(self, token_embedding_a, token_embedding_b):
        """
        Compute attention weight between two tokens via this splat.
        
        The attention is computed as:
        attention = amplitude * exp(-0.5 * (a-μ)ᵀΣ⁻¹(a-μ)) * exp(-0.5 * (b-μ)ᵀΣ⁻¹(b-μ))
        
        Args:
            token_embedding_a (torch.Tensor): Embedding of first token
            token_embedding_b (torch.Tensor): Embedding of second token
            
        Returns:
            torch.Tensor: Scalar attention weight
        """
        # Compute Gaussian values for both tokens
        gaussian_a = compute_gaussian(
            token_embedding_a, 
            self.position, 
            self.precision
        )
        
        gaussian_b = compute_gaussian(
            token_embedding_b, 
            self.position, 
            self.precision
        )
        
        # Attention is the product of individual Gaussian values, scaled by amplitude
        attention = self.amplitude * gaussian_a * gaussian_b
        
        return attention
    
    def compute_attention_batch(self, token_embeddings):
        """
        Compute attention weights for a batch of tokens.
        
        Args:
            token_embeddings (torch.Tensor): Batch of token embeddings
                                           Shape: [batch_size, embedding_dim]
                                           
        Returns:
            torch.Tensor: Attention matrix
                         Shape: [batch_size, batch_size]
        """
        batch_size = token_embeddings.shape[0]
        
        # Compute Gaussian values for all tokens in batch
        # Shape: [batch_size]
        gaussian_values = torch.zeros(batch_size, device=token_embeddings.device)
        
        for i in range(batch_size):
            gaussian_values[i] = compute_gaussian(
                token_embeddings[i], 
                self.position, 
                self.precision
            )
        
        # Compute outer product to get attention matrix
        # Shape: [batch_size, batch_size]
        attention_matrix = self.amplitude * torch.outer(gaussian_values, gaussian_values)
        
        return attention_matrix
    
    def update_activation(self, activation_value):
        """
        Update the activation history with a new value.
        
        Args:
            activation_value (float): New activation value to add
        """
        self.activation_history.append(activation_value)
    
    def get_average_activation(self):
        """
        Get the average activation over the history buffer.
        
        Returns:
            float: Average activation value
        """
        if len(self.activation_history) == 0:
            return 0.0
        
        return sum(self.activation_history) / len(self.activation_history)
    
    def update_parameters(self, position_delta, covariance_delta, amplitude_delta):
        """
        Update the splat parameters based on deltas.
        
        Args:
            position_delta (torch.Tensor): Change in position
            covariance_delta (torch.Tensor): Change in covariance
            amplitude_delta (float): Change in amplitude
        """
        # Update position
        self.position = self.position + position_delta
        
        # Update covariance and ensure it remains positive definite
        new_covariance = self.covariance + covariance_delta
        self.covariance = ensure_positive_definite(new_covariance)
        
        # Recompute precision matrix
        self.precision = stable_matrix_inverse(self.covariance)
        
        # Update amplitude and ensure it remains within bounds
        new_amplitude = self.amplitude + amplitude_delta
        self.amplitude = bounded_amplitude(new_amplitude)
        
        # Update last update time
        self.last_update_time += 1
    
    def compute_distance_to(self, other_splat):
        """
        Compute the Mahalanobis distance to another splat.
        
        Args:
            other_splat (Splat): Another splat instance
            
        Returns:
            float: Distance between the two splats
        """
        diff = self.position - other_splat.position
        avg_precision = (self.precision + other_splat.precision) / 2
        distance = torch.sqrt(torch.dot(diff, torch.mv(avg_precision, diff)))
        return distance.item()
    
    def to_dict(self):
        """
        Convert the splat to a dictionary for serialization.
        
        Returns:
            dict: Dictionary representation of the splat
        """
        # Use higher precision for serialization to avoid round-trip issues
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'covariance': self.covariance.clone().detach().tolist(),  # Use clone to ensure exact copy
            'amplitude': float(self.amplitude),
            'creation_time': self.creation_time,
            'last_update_time': self.last_update_time,
            'average_activation': self.get_average_activation()
        }
    
    @classmethod
    def from_dict(cls, data, device=None):
        """
        Create a splat from a dictionary representation.
        
        Args:
            data (dict): Dictionary with splat parameters
            device (torch.device, optional): Device to place tensors on
            
        Returns:
            Splat: New splat instance
        """
        # Use double precision during conversion to maintain accuracy
        position = torch.tensor(data['position'], dtype=torch.float64, device=device)
        covariance = torch.tensor(data['covariance'], dtype=torch.float64, device=device)
        
        # Convert back to float32 for consistency with the rest of the system
        position = position.float()
        covariance = covariance.float()
        
        splat = cls(
            position=position,
            covariance=covariance,
            amplitude=data['amplitude'],
            splat_id=data['id']
        )
        
        splat.creation_time = data['creation_time']
        splat.last_update_time = data['last_update_time']
        
        return splat
