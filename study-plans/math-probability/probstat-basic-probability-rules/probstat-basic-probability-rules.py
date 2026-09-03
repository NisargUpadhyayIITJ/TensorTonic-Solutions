def basic_probability(p_a, p_b, p_a_and_b):
    """
    Returns: [p_union, p_a_complement, p_b_complement, p_a_and_not_b] as a list.
    """
    p_union = round(p_a + p_b - p_a_and_b, 4)
    p_a_comp = round(1 - p_a, 4)
    p_b_comp = round(1 - p_b, 4)
    p_a_and_b_comp = round(p_a - p_a_and_b, 4)
    return [p_union, p_a_comp, p_b_comp, p_a_and_b_comp]