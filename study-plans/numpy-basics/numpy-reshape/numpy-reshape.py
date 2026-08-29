import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    array = np.asarray(data, dtype=np.float64)
    if(operation == "flatten"):
        return np.reshape(array, array.shape[0] * array.shape[1])
    elif(operation == "transpose"):
        return np.transpose(array)
    return np.reshape(array, (1, array.shape[0], array.shape[1]))
