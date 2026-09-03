from scipy import stats

def z_test_one_sample(x_bar, mu_0, sigma, n, alpha):
    """
    Returns: [z_stat, p_value, reject] as a list.
    """
    z = round((x_bar - mu_0) / (sigma / n**0.5), 4)
    p_val = round(2 * float(stats.norm.sf(abs(z))), 4)
    return [z, p_val, p_val < alpha]