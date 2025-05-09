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
