import numpy as np

def vector_projection(u, v):
    """
    Returns: float64 array, the projection of u onto v.
    """
    u = np.array(u)
    v = np.array(v)
    v = v / np.linalg.norm(v)
    return np.dot(u, v) * v