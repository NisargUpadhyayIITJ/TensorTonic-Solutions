from scipy import stats

def sampling_distribution(mu, sigma, n, threshold):
    """
    Returns: dict with 'mean', 'std_error', 'tail_probability' as floats.
    """
    samp_mean = round(mu, 4)
    samp_std = round(sigma / n**0.5, 4)
    prob_below = round(float(stats.norm.cdf(threshold, mu, sigma / n**0.5)), 4)
    return {"sampling_mean": samp_mean, "sampling_std": samp_std, "prob_below_threshold": prob_below}