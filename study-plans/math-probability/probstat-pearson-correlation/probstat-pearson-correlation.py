import numpy as np

def pearson_correlation(X):
    """
    Returns: ndarray, the Pearson correlation matrix.
    """
    X = np.array(X)
    return np.corrcoef([X[:, idx] for idx in range(X.shape[1])])