import numpy as np

def percentiles(x, q):
    """
    Returns: numpy array of percentile values.
    """
    x = np.array(x)
    q = np.array(q)
    return np.percentile(x, q)
