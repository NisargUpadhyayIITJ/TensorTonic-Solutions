import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    x = np.asarray(data)
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)