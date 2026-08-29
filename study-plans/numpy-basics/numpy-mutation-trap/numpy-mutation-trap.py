import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    a = np.array(data, dtype=np.float64)[row_idx]
    b = np.where(a > hi, hi, a)
    b = np.where(b < lo, lo, b)
    return np.stack([a, b])
