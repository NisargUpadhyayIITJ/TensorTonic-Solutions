import numpy as np

def projection_matrix(A):
    """
    Returns: ndarray, the projection matrix onto the column space of A.
    """
    A = np.array(A)
    ata = np.linalg.pinv(A.T @ A)
    return A @ ata @ A.T