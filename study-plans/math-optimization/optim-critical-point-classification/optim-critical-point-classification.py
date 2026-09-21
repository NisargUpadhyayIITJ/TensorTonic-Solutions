import numpy as np

def classify_critical_point(H: list) -> str:
    """
    Returns the second-order classification string.
    """
    H = np.array(H)
    eigenvalues, _ = np.linalg.eig(H)
    if(np.any(np.abs(eigenvalues) <= 1e-6)):
        return "degenerate"
    if(np.all(eigenvalues > 1e-6)):
        return "local_min"
    if(np.all(eigenvalues < -1e-6)):
        return "local_max"
    return "saddle"    