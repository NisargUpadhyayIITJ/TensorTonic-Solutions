import numpy as np

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    x = np.array(data)
    a = [np.sort(x, axis=axis), np.argsort(x, axis=axis)]
    return np.array(a)