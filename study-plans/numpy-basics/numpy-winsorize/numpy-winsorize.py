import numpy as np

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    X = np.array(data)
    lo = np.percentile(X, lo_q, axis=0)
    hi = np.percentile(X, hi_q, axis=0)
    res = np.clip(X, lo, hi)
    a = [res, X < lo, X > hi]
    return np.array(a)