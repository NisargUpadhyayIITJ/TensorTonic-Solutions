import numpy as np
from scipy import stats

def t_test_one_sample(data, mu_0, alpha):
    """
    Returns: dict with 't_stat', 'p_value', 'df' (floats), 'reject' (bool), 'ci_lower', 'ci_upper'.
    """
    arr = np.array(data, dtype=float)
    n = len(arr)
    x_bar = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    df = n - 1
    if s == 0.0:
        t_stat = 0.0 if x_bar == mu_0 else float('inf') if x_bar > mu_0 else float('-inf')
        p_val = 1.0 if x_bar == mu_0 else 0.0
    else:
        t_stat = round((x_bar - mu_0) / (s / n**0.5), 4)
        p_val = round(2 * float(stats.t.sf(abs(t_stat), df)), 4)
    return {"t_statistic": t_stat, "degrees_of_freedom": df, "p_value": p_val, "reject_null": p_val < alpha}