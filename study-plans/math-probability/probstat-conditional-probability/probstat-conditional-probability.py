def conditional_probability(p_a, p_b, p_a_and_b):
    """
    Returns: [p_a_given_b, p_b_given_a] as a list.
    """
    p_a_given_b = round(p_a_and_b / p_b, 4)
    p_b_given_a = round(p_a_and_b / p_a, 4)
    return [p_a_given_b, p_b_given_a]