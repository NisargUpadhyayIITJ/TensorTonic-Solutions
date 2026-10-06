import numpy as np

def exp_power_backward(x: float, exponent: float, upstream_gradient: float) -> tuple:
    """
    Returns a tuple of exp output, exp gradient, power output, and power gradient.
    """
    exp_output = np.exp(x)
    exp_gradient = upstream_gradient * np.exp(x)
    power_output = 1
    power_gradient = 0
    if(exponent > 0):
        power_output = np.power(x, exponent)
        power_gradient = upstream_gradient * exponent * np.power(x, exponent - 1)
    elif (exponent < 0):
        power_output = 1 / np.power(x, abs(exponent))
        power_gradient = upstream_gradient * exponent * (1 / np.power(x, abs(exponent - 1)))
    return float(exp_output), float(exp_gradient), float(power_output), float(power_gradient)
