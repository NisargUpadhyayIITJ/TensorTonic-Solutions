def f(a, b, c):
    return a * b + c

def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    da = f(a + h, b, c)
    db = f(a, b + h, c)
    dc = f(a, b, c + h)
    d = f(a, b, c)
    return (
        float(d),
        float((da - d) / h),
        float((db - d) / h),
        float((dc - d) / h)
    )

