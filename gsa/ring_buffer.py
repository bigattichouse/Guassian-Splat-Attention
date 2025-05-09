"""
Gaussian Splat Attention - Ring Buffer Implementation

This module implements a fixed-size buffer (ring buffer) for tracking activation history.
This is used by the Splat class to maintain a history of recent activations,
which helps determine when splats should be removed from the collection.
"""

import collections
import numpy as np

class RingBuffer:
    """
    A fixed-size buffer that overwrites oldest values when full.
    
    This class implements a circular buffer (ring buffer) with a fixed capacity.
    New elements are added at the end, and when the buffer reaches its capacity,
    the oldest elements are overwritten.
    """
    
    def __init__(self, capacity):
        """
        Initialize a new RingBuffer with the specified capacity.
        
        Args:
            capacity (int): Maximum number of elements the buffer can hold
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
    
    def append(self, value):
        """
        Add a new value to the buffer.
        
        If the buffer is at capacity, the oldest value will be removed.
        
        Args:
            value: The value to add to the buffer
        """
        self.buffer.append(value)
    
    def clear(self):
        """
        Remove all values from the buffer.
        """
        self.buffer.clear()
    
    def get_values(self):
        """
        Get all values currently in the buffer.
        
        Returns:
            list: List of all values in the buffer
        """
        return list(self.buffer)
    
    def get_statistics(self):
        """
        Calculate statistical measures of the values in the buffer.
        
        Returns:
            dict: Dictionary containing mean, median, min, max, and std
        """
        if not self.buffer:
            return {
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        values = np.array(self.buffer)
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values))
        }
    
    def is_full(self):
        """
        Check if the buffer is at capacity.
        
        Returns:
            bool: True if the buffer contains exactly capacity elements
        """
        return len(self.buffer) == self.capacity
    
    def __len__(self):
        """
        Get the current number of elements in the buffer.
        
        Returns:
            int: Number of elements currently in the buffer
        """
        return len(self.buffer)
    
    def __iter__(self):
        """
        Allow iteration over the buffer's values.
        
        Returns:
            iterator: Iterator over the buffer's values
        """
        return iter(self.buffer)
    
    def __getitem__(self, index):
        """
        Access an element by index.
        
        Args:
            index (int): Index of the element to retrieve
            
        Returns:
            The element at the specified index
            
        Raises:
            IndexError: If the index is out of range
        """
        if index >= len(self.buffer) or index < -len(self.buffer):
            raise IndexError("Index out of range")
        return self.buffer[index]
    
    def rolling_average(self, window_size=None):
        """
        Calculate the rolling average of values in the buffer.
        
        Args:
            window_size (int, optional): Size of the rolling window.
                                      If None, uses the entire buffer.
                                      
        Returns:
            list: List of rolling averages
        """
        if not self.buffer:
            return []
        
        if window_size is None or window_size > len(self.buffer):
            window_size = len(self.buffer)
        
        values = np.array(self.buffer)
        result = []
        
        for i in range(len(values) - window_size + 1):
            window = values[i:i+window_size]
            result.append(float(np.mean(window)))
        
        return result
    
    def rolling_variance(self, window_size=None):
        """
        Calculate the rolling variance of values in the buffer.
        
        Args:
            window_size (int, optional): Size of the rolling window.
                                      If None, uses the entire buffer.
                                      
        Returns:
            list: List of rolling variances
        """
        if not self.buffer:
            return []
        
        if window_size is None or window_size > len(self.buffer):
            window_size = len(self.buffer)
        
        values = np.array(self.buffer)
        result = []
        
        for i in range(len(values) - window_size + 1):
            window = values[i:i+window_size]
            result.append(float(np.var(window)))
        
        return result
    
    def trend_analysis(self):
        """
        Analyze the trend of values in the buffer.
        
        Returns:
            dict: Dictionary with trend analysis results
        """
        if len(self.buffer) < 2:
            return {
                'increasing': False,
                'decreasing': False,
                'stable': True,
                'slope': 0.0
            }
        
        values = np.array(self.buffer)
        x = np.arange(len(values))
        
        # Calculate linear regression slope
        mean_x = np.mean(x)
        mean_y = np.mean(values)
        
        numerator = np.sum((x - mean_x) * (values - mean_y))
        denominator = np.sum((x - mean_x) ** 2)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend based on slope
        threshold = 0.05  # Threshold for detecting trends
        
        # Convert numpy booleans to Python booleans to avoid test failures
        is_increasing = bool(slope > threshold)
        is_decreasing = bool(slope < -threshold)
        is_stable = bool(abs(slope) <= threshold)
        
        return {
            'increasing': is_increasing,
            'decreasing': is_decreasing,
            'stable': is_stable,
            'slope': float(slope)
        }
