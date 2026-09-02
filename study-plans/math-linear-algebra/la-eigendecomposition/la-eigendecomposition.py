import numpy as np

def eigendecompose(A):
    """
    Returns: tuple (eigenvalues, eigenvectors), sorted by descending magnitude.
    """
    A = np.array(A)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return (eigenvalues, eigenvectors)