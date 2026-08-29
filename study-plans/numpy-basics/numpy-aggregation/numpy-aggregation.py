import numpy as np

def summarize(data, axis):
    """Returns: np.ndarray of shape (4, k), rows are mean, std, min, max"""
    x = np.asarray(data)
    a = [np.mean(x, axis=axis), np.std(x, axis=axis), np.min(x, axis=axis), np.max(x, axis=axis)]
    return np.array(a)