import numpy as np

def nesterov_momentum(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    """
    Returns classical and Nesterov MSE loss curves in a dictionary.
    """
    X = np.array(X)
    y = np.array(y)

    w_n = np.zeros(X.shape[1])
    w_m = np.zeros(X.shape[1])
    vt_n = np.zeros(X.shape[1])
    vt_m = np.zeros(X.shape[1])

    mse_nesterov = []
    mse_momentum = []

    for idx in range(n_epochs):
        
        mse_n = np.mean((X @ w_n - y) ** 2)
        mse_m = np.mean((X @ w_m - y) ** 2)

        mse_nesterov.append(mse_n)
        mse_momentum.append(mse_m)

        wlook = w_n - lr * beta * vt_n
        gt_n = (2 / X.shape[0]) * X.T @ (X @ wlook - y)
        vt_n = beta * vt_n + gt_n
        w_n = w_n - lr * vt_n

        gt_m = (2 / X.shape[0]) * X.T @ (X @ w_m - y)
        vt_m = beta * vt_m + gt_m
        w_m = w_m - lr * vt_m

    return {
        "classical_losses": mse_momentum,
        "nesterov_losses": mse_nesterov
    }