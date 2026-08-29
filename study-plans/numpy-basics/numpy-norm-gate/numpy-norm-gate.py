import numpy as np

def norm_gate(X, W, threshold):
    """Returns: np.ndarray of shape (n, k), gated projection where rows below threshold are zeroed"""
    X = np.array(X)
    W = np.array(W)
    T = X @ W
    norms = np.linalg.norm(T, axis=1)
    return np.where(norms[:, np.newaxis] >= threshold, T, 0.0)