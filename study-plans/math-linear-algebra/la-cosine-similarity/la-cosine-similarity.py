import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    x = np.array(a)
    y = np.array(b)
    mod_x = np.linalg.norm(x)
    mod_y = np.linalg.norm(y)
    if(mod_x == 0.0 or mod_y == 0.0):
        return np.dot(x, y)
    return np.dot(x, y) / (mod_x * mod_y)