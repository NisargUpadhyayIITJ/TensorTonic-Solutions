import numpy as np

def svd(A):
    """
    Returns: tuple (U, s, Vt) where A = U @ diag(s) @ Vt.
    """
    A = np.array(A)
    return np.linalg.svd(A, full_matrices=False)