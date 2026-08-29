import numpy as np

def extract_subarray(arr, row_start, row_stop, col_start, col_stop):
    """
    Returns: 2D ndarray of float64
    """
    array = np.asarray(arr, dtype=np.float64)
    return array[row_start : row_stop, col_start : col_stop]
