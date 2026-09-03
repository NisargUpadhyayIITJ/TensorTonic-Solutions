import numpy as np

def mle_normal(data):
    """
    Returns: dict with 'mu_hat' and 'sigma_hat' as floats (MLE estimates).
    """
    arr = np.array(data, dtype=float)
    mu = round(float(np.mean(arr)), 4)
    sigma = round(float(np.std(arr, ddof=0)), 4)
    return {"mu_mle": mu, "sigma_mle": sigma}