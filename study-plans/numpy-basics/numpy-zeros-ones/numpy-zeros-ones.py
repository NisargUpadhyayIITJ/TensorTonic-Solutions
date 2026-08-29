import numpy as np

def create_filled_array(shape, kind):
    """
    Returns: 2D numpy array of given shape with dtype float64
    """
    if(kind == "zeros"):
        return np.zeros((shape[0], shape[1]), dtype="float64")
    return np.ones((shape[0], shape[1]), dtype="float64")