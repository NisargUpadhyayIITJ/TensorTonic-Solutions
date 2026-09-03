from scipy import stats

def p_value_analysis(test_stat, dist_type, df, tail):
    """
    Returns: dict with 'p_value' (float) and significance level classifications.
    """
    if dist_type == "z":
        if tail == "two": p = 2 * float(stats.norm.sf(abs(test_stat)))
        elif tail == "right": p = float(stats.norm.sf(test_stat))
        else: p = float(stats.norm.cdf(test_stat))
    else:
        if tail == "two": p = 2 * float(stats.t.sf(abs(test_stat), df))
        elif tail == "right": p = float(stats.t.sf(test_stat, df))
        else: p = float(stats.t.cdf(test_stat, df))
    p = round(p, 4)
    return {"p_value": p, "significant_at_01": p < 0.01, "significant_at_05": p < 0.05, "significant_at_10": p < 0.10}