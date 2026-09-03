def independence_test(p_a, p_b, p_a_and_b):
    """
    Returns: dict with 'p_a_times_p_b' (float) and 'is_independent' (bool).
    """
    product = round(p_a * p_b, 4)
    is_independent = abs(p_a_and_b - product) < 1e-9
    return {"p_a_times_p_b": product, "is_independent": is_independent}