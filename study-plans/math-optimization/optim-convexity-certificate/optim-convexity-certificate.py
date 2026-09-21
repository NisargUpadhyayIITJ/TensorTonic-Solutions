import numpy as np

def convexity_certificate(H: list) -> dict:
    """
    Returns the PSD decision and smallest eigenvalue in a dictionary.
    """
    H = np.array(H)
    eigenvalues, _ = np.linalg.eig(H)
    eigenvalues = np.sort(eigenvalues)
    PSD = np.all(eigenvalues >= -10e-6)
    return {
        "is_convex": PSD,
        "min_eigenvalue": round(float(eigenvalues[0]), 6)
    }