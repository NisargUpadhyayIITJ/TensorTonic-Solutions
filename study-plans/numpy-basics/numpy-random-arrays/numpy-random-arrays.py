import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    rng = np.random.default_rng(seed)
    if kind == 'uniform':
        return rng.random(tuple(shape), dtype=np.float64)
    return rng.standard_normal(tuple(shape), dtype=np.float64)
