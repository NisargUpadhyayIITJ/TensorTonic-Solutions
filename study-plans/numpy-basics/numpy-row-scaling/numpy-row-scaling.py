import numpy as np

def scale_rows(data, weights):
    """Returns: np.ndarray of shape (m, n), each row scaled by corresponding weight"""
    a = np.array(data)
    weights = np.array(weights)
    return a * weights[:, np.newaxis]