import numpy as np

def create_sequence(start, stop, param, kind):
    """
    Returns: 1D ndarray of float64 values
    """
    if(kind == "arange"):
        return np.arange(start, stop, param, dtype="float64")
    return np.linspace(start, stop, param, dtype="float64")
