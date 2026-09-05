import numpy as np

def activation_derivative(name, x):
    """
    Returns: list of floats (the derivative evaluated at each x)
    """
    x = np.asarray(x, dtype=np.float64)
    
    if name == 'sigmoid':
        s = 1.0 / (1.0 + np.exp(-x))
        return (s * (1.0 - s)).tolist()

    elif name == 'tanh':
        t = np.tanh(x)
        return (1.0 - t ** 2).tolist()

    elif name == 'relu':
        return np.where(x > 0, 1.0, 0.0).tolist()

    elif name == 'swish':
        s = 1.0 / (1.0 + np.exp(-x))
        return (s + x * s * (1.0 - s)).tolist()
