import numpy as np

def impute(X, method="mean"):
    """
    Returns: 2D list with NaN values replaced using the specified method
    """
    X = np.array(X, dtype=float)
    result = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.all():
            result[mask, j] = 0.0
        else:
            if method == "mean":
                fill = np.nanmean(col)
            else:
                fill = np.nanmedian(col)
            result[mask, j] = fill
    return np.round(result, 4).tolist()