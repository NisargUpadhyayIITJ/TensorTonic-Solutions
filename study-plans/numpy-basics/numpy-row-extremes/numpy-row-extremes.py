import numpy as np

def row_extremes(data):
    """Returns: np.ndarray of shape (4, m), rows are max_val, max_col, min_val, min_col"""
    X = np.asarray(data, dtype=np.float64)
    a = [np.max(X, axis=1), np.argmax(X, axis=1), np.min(X, axis=1), np.argmin(X, axis=1)]
    return np.array(a)