import numpy as np

def least_squares(A, b):
    """
    Returns: float64 array, the solution minimizing ||A @ x - b||^2.
    """
    X = np.array(A)
    y = np.array(b)
    xtx = np.linalg.pinv(X.T @ X)
    return xtx @ X.T @ y