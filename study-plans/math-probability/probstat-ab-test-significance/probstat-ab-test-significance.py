from scipy import stats

def ab_test(control_visitors, control_conversions, treatment_visitors, treatment_conversions, alpha):
    """
    Returns: [p_control, p_treatment, z_stat, p_value, reject] as a list.
    """
    p_c = control_conversions / control_visitors
    p_t = treatment_conversions / treatment_visitors
    p_pool = (control_conversions + treatment_conversions) / (control_visitors + treatment_visitors)
    se = (p_pool * (1 - p_pool) * (1/control_visitors + 1/treatment_visitors)) ** 0.5
    z = round((p_t - p_c) / se, 4)
    p_val = round(2 * float(stats.norm.sf(abs(z))), 4)
    return [p_c, p_t, z, p_val, p_val < alpha]