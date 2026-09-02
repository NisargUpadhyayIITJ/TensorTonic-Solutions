import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.array(v)
    v1 = np.linalg.norm(v, ord=1)
    v2 = np.linalg.norm(v, ord=2)
    v_inf = np.linalg.norm(v, ord=np.inf)
    return np.stack([v1, v2, v_inf])