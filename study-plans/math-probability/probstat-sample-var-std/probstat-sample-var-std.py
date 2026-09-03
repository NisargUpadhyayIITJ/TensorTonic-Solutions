import numpy as np

def sample_var_std(x):
    """
    Returns: dict with 'variance' and 'std_dev' as floats.
    """
    x = np.array(x)
    s2 = np.sum((x - np.mean(x)) ** 2) / (x.shape[0] - 1)
    s = np.sqrt(s2)
    return {
        "variance" : s2,
        "std_dev" : s
    }