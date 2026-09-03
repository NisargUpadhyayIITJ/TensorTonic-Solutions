def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    """
    Returns: float, the posterior probability P(A|B).
    """
    p_not_a = 1 - p_a
    p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    return round((p_b_given_a * p_a) / p_b, 4)