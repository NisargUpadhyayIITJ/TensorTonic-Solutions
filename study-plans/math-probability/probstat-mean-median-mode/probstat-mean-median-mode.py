import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Returns: dict with 'mean', 'median', 'mode' as floats.
    """
    x = np.array(x)
    mean = np.mean(x)
    median = np.median(x)
    values, counts = np.unique(x, return_counts=True)
    index = np.argmax(counts)
    mode_value = values[index]
    return {
        "mean" : mean,
        "median" : median,
        "mode" : mode_value
    }
    