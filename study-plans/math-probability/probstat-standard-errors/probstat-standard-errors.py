import numpy as np

def standard_errors(samples):
    """
    Returns: dict with 'standard_errors' (list of floats) and 'comparison'.
    """
    ses = []
    for s in samples:
        arr = np.array(s, dtype=float)
        se = float(np.std(arr, ddof=1) / len(arr)**0.5)
        ses.append(round(se, 4))
    mean_se = round(float(np.mean(ses)), 4)
    return {"standard_errors": ses, "mean_se": mean_se}

