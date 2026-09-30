import numpy as np

def f(x, coefficients):
    ans = 0
    poly = 1
    for coefficient in coefficients:
        ans += coefficient * poly
        poly *= x
    return ans

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    fx = float(f(x, coefficients))
    fxh = float(f(x + h, coefficients))
    slope = float((fxh - fx) / h)
    return (fx, fxh, slope)