import numpy as np

def rbf_kernel_matrix(X, gamma):
    """
    Returns: ndarray of shape (n, n), the RBF kernel matrix.
    """
    X = np.array(X)
    res = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            xi = X[i]
            xj = X[j]
            res[i][j] = np.exp(-1 * gamma * np.sum((xi - xj) ** 2))
    return res