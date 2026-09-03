import numpy as np
from scipy import stats

def clt_confidence_interval(data, confidence):
    """
    Returns: [mean, std_error, ci_lower, ci_upper] as a list.
    """
    n = len(data)
    x_bar = round(float(np.mean(data)), 4)
    s = float(np.std(data, ddof=1))
    se = round(s / n**0.5, 4)
    z = round(float(stats.norm.ppf((1 + confidence) / 2)), 4)
    ci_lower = round(x_bar - z * se, 4)
    ci_upper = round(x_bar + z * se, 4)
    return [x_bar, se, ci_lower, ci_upper]