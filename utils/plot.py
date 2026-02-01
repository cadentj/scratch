import numpy as np

# Smoothing window size (2 hours = 8 slots of 15-min each)
SMOOTHING_WINDOW = 8

def smooth_data(data, window_size=SMOOTHING_WINDOW):
    """Apply a centered rolling mean to smooth the data."""
    data = np.array(data, dtype=float)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='same')
    # Fix edge effects
    for i in range(window_size // 2):
        left_window = i + 1
        smoothed[i] = np.mean(data[:left_window + window_size // 2])
        right_idx = len(data) - 1 - i
        smoothed[right_idx] = np.mean(data[right_idx - window_size // 2:])
    return smoothed
