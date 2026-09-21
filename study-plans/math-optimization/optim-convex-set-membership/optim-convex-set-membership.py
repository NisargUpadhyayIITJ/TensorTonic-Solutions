import numpy as np

def convex_set_membership(A: list, b: list, x: list) -> dict:
    """
    Returns the membership decision and maximum violation in a dictionary.
    """
    A = np.array(A)
    x = np.array(x)
    b = np.array(b)
    residual = A @ x - b
    in_set = np.all(residual <= 10e-6)
    max_violation = np.max(residual)
    return {
        "in_set": in_set,
        "max_violation": round(float(max_violation), 6)
    }