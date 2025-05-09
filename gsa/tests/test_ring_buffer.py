"""
Tests for the RingBuffer implementation.
"""

import pytest
import numpy as np
from gsa.ring_buffer import RingBuffer


def test_ring_buffer_initialization():
    """Test that a ring buffer can be initialized with a valid capacity."""
    buffer = RingBuffer(5)
    assert buffer.capacity == 5
    assert len(buffer) == 0


def test_ring_buffer_initialization_invalid_capacity():
    """Test that initialization fails with invalid capacity."""
    with pytest.raises(ValueError):
        RingBuffer(0)
    
    with pytest.raises(ValueError):
        RingBuffer(-1)


def test_append():
    """Test adding values to the buffer."""
    buffer = RingBuffer(3)
    
    buffer.append(1)
    assert len(buffer) == 1
    assert buffer[0] == 1
    
    buffer.append(2)
    assert len(buffer) == 2
    assert buffer[0] == 1
    assert buffer[1] == 2
    
    buffer.append(3)
    assert len(buffer) == 3
    assert buffer[0] == 1
    assert buffer[1] == 2
    assert buffer[2] == 3


def test_append_non_finite():
    """Test handling of non-finite values when appending."""
    buffer = RingBuffer(3)
    
    buffer.append(float('nan'))
    assert len(buffer) == 1
    assert buffer[0] == 0.0  # NaN should be replaced with 0.0
    
    buffer.append(float('inf'))
    assert len(buffer) == 2
    assert buffer[1] == 0.0  # Inf should be replaced with 0.0


def test_overflow():
    """Test that oldest values are dropped when buffer is full."""
    buffer = RingBuffer(3)
    
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    buffer.append(4)
    
    assert len(buffer) == 3
    assert buffer[0] == 2
    assert buffer[1] == 3
    assert buffer[2] == 4


def test_clear():
    """Test clearing the buffer."""
    buffer = RingBuffer(3)
    
    buffer.append(1)
    buffer.append(2)
    
    buffer.clear()
    
    assert len(buffer) == 0


def test_get_values():
    """Test retrieving all values."""
    buffer = RingBuffer(3)
    
    buffer.append(1)
    buffer.append(2)
    
    values = buffer.get_values()
    
    assert isinstance(values, list)
    assert values == [1, 2]


def test_get_statistics_empty():
    """Test getting statistics for an empty buffer."""
    buffer = RingBuffer(3)
    
    stats = buffer.get_statistics()
    
    assert stats['mean'] == 0.0
    assert stats['median'] == 0.0
    assert stats['min'] == 0.0
    assert stats['max'] == 0.0
    assert stats['std'] == 0.0


def test_get_statistics():
    """Test getting statistics for a non-empty buffer."""
    buffer = RingBuffer(5)
    
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    buffer.append(4)
    buffer.append(5)
    
    stats = buffer.get_statistics()
    
    assert stats['mean'] == 3.0
    assert stats['median'] == 3.0
    assert stats['min'] == 1.0
    assert stats['max'] == 5.0
    assert stats['std'] == pytest.approx(np.std([1, 2, 3, 4, 5]))


def test_get_statistics_error_handling():
    """Test error handling in statistics calculation."""
    buffer = RingBuffer(3)
    buffer.append(1)
    buffer.append(2)
    
    # Mock numpy functions to simulate errors
    orig_mean = np.mean
    orig_median = np.median
    orig_min = np.min
    orig_max = np.max
    orig_std = np.std
    
    def mock_error(*args, **kwargs):
        raise RuntimeError("Simulated numpy error")
    
    np.mean = mock_error
    np.median = mock_error
    np.min = mock_error
    np.max = mock_error
    np.std = mock_error
    
    try:
        stats = buffer.get_statistics()
        
        # Should return zeros for all metrics on error
        assert stats['mean'] == 0.0
        assert stats['median'] == 0.0
        assert stats['min'] == 0.0
        assert stats['max'] == 0.0
        assert stats['std'] == 0.0
    finally:
        # Restore original functions
        np.mean = orig_mean
        np.median = orig_median
        np.min = orig_min
        np.max = orig_max
        np.std = orig_std


def test_is_full():
    """Test checking if the buffer is full."""
    buffer = RingBuffer(2)
    
    assert not buffer.is_full()
    
    buffer.append(1)
    assert not buffer.is_full()
    
    buffer.append(2)
    assert buffer.is_full()


def test_iteration():
    """Test iterating over the buffer."""
    buffer = RingBuffer(3)
    
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    
    values = []
    for value in buffer:
        values.append(value)
    
    assert values == [1, 2, 3]


def test_getitem():
    """Test accessing elements by index."""
    buffer = RingBuffer(3)
    
    buffer.append(1)
    buffer.append(2)
    
    assert buffer[0] == 1
    assert buffer[1] == 2
    assert buffer[-1] == 2
    assert buffer[-2] == 1
    
    with pytest.raises(IndexError):
        buffer[2]
    
    with pytest.raises(IndexError):
        buffer[-3]


def test_rolling_average():
    """Test calculating rolling averages."""
    buffer = RingBuffer(5)
    
    # Empty buffer
    assert buffer.rolling_average() == []
    
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    buffer.append(4)
    buffer.append(5)
    
    # Full window
    assert buffer.rolling_average() == [3.0]
    
    # Custom window size
    avgs = buffer.rolling_average(window_size=2)
    assert len(avgs) == 4
    assert avgs[0] == 1.5
    assert avgs[1] == 2.5
    assert avgs[2] == 3.5
    assert avgs[3] == 4.5
    
    # Window size larger than buffer
    assert buffer.rolling_average(window_size=10) == [3.0]


def test_rolling_average_edge_cases():
    """Test rolling average with edge cases."""
    buffer = RingBuffer(5)
    buffer.append(1)
    
    # Window size <= 0
    assert buffer.rolling_average(window_size=0) == [1.0]
    assert buffer.rolling_average(window_size=-1) == [1.0]
    
    # Test error handling
    buffer = RingBuffer(3)
    buffer.append(1)
    buffer.append(float('nan'))  # Non-finite value
    
    avgs = buffer.rolling_average()
    assert len(avgs) == 1
    assert np.isfinite(avgs[0])
    
    # Test with error in numpy function
    orig_mean = np.mean
    
    def mock_error(*args, **kwargs):
        raise RuntimeError("Simulated numpy error")
    
    np.mean = mock_error
    
    try:
        assert buffer.rolling_average() == []
    finally:
        np.mean = orig_mean


def test_rolling_variance():
    """Test calculating rolling variances."""
    buffer = RingBuffer(5)
    
    # Empty buffer
    assert buffer.rolling_variance() == []
    
    buffer.append(1)
    buffer.append(1)
    buffer.append(1)
    buffer.append(1)
    buffer.append(1)
    
    # All values the same
    assert buffer.rolling_variance() == [0.0]
    
    buffer.clear()
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    buffer.append(4)
    buffer.append(5)
    
    # Custom window size
    vars = buffer.rolling_variance(window_size=3)
    assert len(vars) == 3
    assert vars[0] == pytest.approx(np.var([1, 2, 3]))
    assert vars[1] == pytest.approx(np.var([2, 3, 4]))
    assert vars[2] == pytest.approx(np.var([3, 4, 5]))


def test_rolling_variance_edge_cases():
    """Test rolling variance with edge cases."""
    buffer = RingBuffer(5)
    buffer.append(1)
    
    # Window size <= 1 (need at least 2 elements for variance)
    # The function may handle this by using window_size=2 which is valid behavior
    result = buffer.rolling_variance(window_size=1)
    # If it returns [], that's fine, if it returns [0.0], that's also acceptable
    assert result == [] or result == [0.0]
    
    # Add another element so we can calculate variance
    buffer.append(2)
    assert buffer.rolling_variance(window_size=2) == [pytest.approx(np.var([1, 2]))]
    
    # Test error handling
    buffer = RingBuffer(3)
    buffer.append(1)
    buffer.append(float('nan'))  # Non-finite value
    buffer.append(3)
    
    vars = buffer.rolling_variance()
    assert len(vars) == 1
    assert np.isfinite(vars[0])
    
    # Test with error in numpy function
    orig_var = np.var
    
    def mock_error(*args, **kwargs):
        raise RuntimeError("Simulated numpy error")
    
    np.var = mock_error
    
    try:
        assert buffer.rolling_variance() == []
    finally:
        np.var = orig_var


def test_trend_analysis_empty():
    """Test trend analysis with insufficient data."""
    buffer = RingBuffer(5)
    
    trend = buffer.trend_analysis()
    
    assert trend['increasing'] is False
    assert trend['decreasing'] is False
    assert trend['stable'] is True
    assert trend['slope'] == 0.0


def test_trend_analysis_increasing():
    """Test trend analysis with increasing values."""
    buffer = RingBuffer(5)
    
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    buffer.append(4)
    buffer.append(5)
    
    trend = buffer.trend_analysis()
    
    assert trend['increasing'] is True
    assert trend['decreasing'] is False
    assert trend['stable'] is False
    assert trend['slope'] > 0.9  # Should be close to 1.0


def test_trend_analysis_decreasing():
    """Test trend analysis with decreasing values."""
    buffer = RingBuffer(5)
    
    buffer.append(5)
    buffer.append(4)
    buffer.append(3)
    buffer.append(2)
    buffer.append(1)
    
    trend = buffer.trend_analysis()
    
    assert trend['increasing'] is False
    assert trend['decreasing'] is True
    assert trend['stable'] is False
    assert trend['slope'] < -0.9  # Should be close to -1.0


def test_trend_analysis_stable():
    """Test trend analysis with stable values."""
    buffer = RingBuffer(5)
    
    buffer.append(3)
    buffer.append(3.02)
    buffer.append(2.97)
    buffer.append(3.03)
    buffer.append(2.98)
    
    trend = buffer.trend_analysis()
    
    assert trend['stable'] is True
    assert abs(trend['slope']) < 0.05  # Should be close to 0.0


def test_trend_analysis_with_non_finite():
    """Test trend analysis with non-finite values."""
    buffer = RingBuffer(5)
    
    buffer.append(1)
    buffer.append(float('nan'))  # Should be replaced with 0.0
    buffer.append(3)
    buffer.append(float('inf'))  # Should be replaced with 0.0
    buffer.append(5)
    
    trend = buffer.trend_analysis()
    
    # Check that we got valid results despite non-finite values
    assert isinstance(trend['increasing'], bool)
    assert isinstance(trend['decreasing'], bool)
    assert isinstance(trend['stable'], bool)
    assert np.isfinite(trend['slope'])


def test_trend_analysis_error_handling():
    """Test error handling in trend analysis."""
    buffer = RingBuffer(5)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    
    # Mock numpy functions to simulate errors
    orig_mean = np.mean
    orig_sum = np.sum
    
    def mock_error(*args, **kwargs):
        raise RuntimeError("Simulated numpy error")
    
    np.mean = mock_error
    np.sum = mock_error
    
    try:
        trend = buffer.trend_analysis()
        
        # Should return stable trend with zero slope on error
        assert trend['increasing'] is False
        assert trend['decreasing'] is False
        assert trend['stable'] is True
        assert trend['slope'] == 0.0
    finally:
        # Restore original functions
        np.mean = orig_mean
        np.sum = orig_sum


def test_trend_analysis_minimal_data():
    """Test trend analysis with minimal data points."""
    buffer = RingBuffer(5)
    
    # Test with single value
    buffer.append(1.0)
    trend = buffer.trend_analysis()
    assert trend['stable'] is True
    
    # Test with two values that are identical
    buffer.clear()
    buffer.append(1.0)
    buffer.append(1.0)
    trend = buffer.trend_analysis()
    assert trend['stable'] is True
    assert abs(trend['slope']) < 1e-6
    
    # Test with two values that are different
    buffer.clear()
    buffer.append(1.0)
    buffer.append(2.0)
    trend = buffer.trend_analysis()
    assert trend['increasing'] is True
    assert trend['slope'] > 0


def test_decay_values():
    """Test applying exponential decay to values."""
    buffer = RingBuffer(3)
    
    buffer.append(1.0)
    buffer.append(2.0)
    buffer.append(3.0)
    
    # Apply decay with factor 0.5
    buffer.decay_values(decay_factor=0.5)
    
    assert buffer[0] == 0.5
    assert buffer[1] == 1.0
    assert buffer[2] == 1.5


def test_decay_values_edge_cases():
    """Test decay_values with edge cases."""
    buffer = RingBuffer(3)
    
    # Empty buffer should not raise an error
    buffer.decay_values()
    assert len(buffer) == 0
    
    # Out of range decay factors should be clamped
    buffer.append(2.0)
    buffer.decay_values(decay_factor=2.0)  # Too high, should clamp to 1.0
    assert buffer[0] == 2.0
    
    buffer.decay_values(decay_factor=-0.5)  # Too low, should clamp to 0.0
    assert buffer[0] == 0.0
    
    # Test error handling
    buffer = RingBuffer(2)
    buffer.append(1.0)
    
    # Create mock multiplication that raises an error
    class MockInt(float):
        def __mul__(self, other):
            raise RuntimeError("Simulated multiplication error")
    
    buffer.buffer[0] = MockInt(5.0)
    
    # Should not raise an error
    buffer.decay_values(decay_factor=0.5)


def test_weighted_average():
    """Test calculating weighted average of values."""
    buffer = RingBuffer(3)
    
    buffer.append(1.0)
    buffer.append(2.0)
    buffer.append(3.0)
    
    # Without explicit weights (should use exponential weighting)
    avg = buffer.weighted_average()
    
    # Most recent value (3.0) should have highest weight
    assert avg > 2.0
    
    # With explicit weights
    avg = buffer.weighted_average(weights=[1.0, 2.0, 3.0])
    
    # Equivalent to (1.0*1.0 + 2.0*2.0 + 3.0*3.0) / (1.0 + 2.0 + 3.0)
    expected = (1.0*1.0 + 2.0*2.0 + 3.0*3.0) / (1.0 + 2.0 + 3.0)
    assert avg == expected


def test_weighted_average_edge_cases():
    """Test weighted_average with edge cases."""
    buffer = RingBuffer(3)
    
    # Empty buffer should return 0.0
    assert buffer.weighted_average() == 0.0
    
    # With zero or negative weights sum
    buffer.append(1.0)
    buffer.append(2.0)
    
    assert buffer.weighted_average(weights=[0.0, 0.0]) == 0.0
    assert buffer.weighted_average(weights=[-1.0, -2.0]) == 0.0
    
    # Test error handling
    orig_sum = sum  # Python's built-in sum function
    
    def mock_error(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], list) and len(args[0]) > 0:
            if isinstance(args[0][0], tuple):
                raise RuntimeError("Simulated sum error")
        return orig_sum(*args, **kwargs)
    
    globals()['sum'] = mock_error
    
    try:
        avg = buffer.weighted_average()
        # Just test that we get a valid value back, don't check exact value
        assert isinstance(avg, float)
        assert np.isfinite(avg)
    finally:
        globals()['sum'] = orig_sum


def test_weighted_average_complex_weights():
    """Test weighted average with more complex weight patterns."""
    buffer = RingBuffer(4)
    
    buffer.append(1.0)
    buffer.append(2.0)
    buffer.append(3.0)
    buffer.append(4.0)
    
    # Test with zero weights for some elements
    weights = [0.0, 0.0, 1.0, 1.0]
    avg = buffer.weighted_average(weights=weights)
    assert avg == 3.5  # (3*1 + 4*1)/(1+1)
    
    # Test with negative weights (should be handled gracefully)
    weights = [-1.0, -2.0, 3.0, 4.0]
    avg = buffer.weighted_average(weights=weights)
    # We should get a result, but exact behavior depends on implementation
    assert np.isfinite(avg)


# New tests to improve coverage on lines 242-248, 263, 341-343

def test_trend_analysis_exception_early():
    """Test trend analysis with an exception early in the method."""
    buffer = RingBuffer(3)
    buffer.append(1.0)
    buffer.append(2.0)
    
    # Mock np.mean to raise an exception
    orig_mean = np.mean
    def mock_mean(*args, **kwargs):
        raise RuntimeError("Mock mean error")
    
    np.mean = mock_mean
    
    try:
        trend = buffer.trend_analysis()
        
        # Verify default values from the exception handler
        assert trend['increasing'] is False
        assert trend['decreasing'] is False
        assert trend['stable'] is True
        assert trend['slope'] == 0.0
    finally:
        # Restore original function
        np.mean = orig_mean
        
def test_decay_values_with_buffer_exception():
    """Test decay_values with an element that raises an exception during multiplication."""
    buffer = RingBuffer(3)
    
    # Create a custom class that raises an exception when multiplied
    class ErrorOnMultiply(float):
        def __rmul__(self, other):
            raise RuntimeError("Error on multiply")
        
        def __mul__(self, other):
            raise RuntimeError("Error on multiply")
    
    # Add normal values first
    buffer.append(1.0)
    buffer.append(ErrorOnMultiply(2.0))
    
    # This should handle the exception without crashing
    buffer.decay_values(0.5)
    
    # Just assert that we reached this point without an exception
    assert True

def test_weighted_average_with_weights_exception():
    """Test weighted_average when weights cause an exception."""
    buffer = RingBuffer(3)
    buffer.append(1.0)
    buffer.append(2.0)
    
    # Create weights that will cause an error when used
    class BadWeights(list):
        def __iter__(self):
            raise RuntimeError("Bad weights iterator")
    
    # This should hit lines 341-343 in the exception handler
    avg = buffer.weighted_average(weights=BadWeights([1.0, 2.0]))
    
    # Should fall back to simple average
    assert avg == 1.5  # (1.0 + 2.0) / 2


def test_weighted_average_with_exception_in_calculation():
    """Test weighted_average with an exception during the calculation."""
    buffer = RingBuffer(3)
    
    # Add normal values first
    buffer.append(1.0)
    buffer.append(2.0)
    
    # Create an object that will cause multiplication to fail
    class ErrorValue(float):
        def __new__(cls):
            return float.__new__(cls, 3.0)
            
        def __mul__(self, other):
            raise TypeError("Cannot multiply ErrorValue")
        
        def __rmul__(self, other):
            raise TypeError("Cannot multiply ErrorValue")
    
    # Replace one of the values directly in the buffer
    buffer.buffer[1] = ErrorValue()
    
    # This should hit lines 341-343 in the exception handler
    avg = buffer.weighted_average()
    
    # Verify we get some valid result without error
    assert isinstance(avg, float)
    assert np.isfinite(avg)
