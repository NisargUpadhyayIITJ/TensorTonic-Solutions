import numpy as np

def momentum_gd(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    """
    Returns vanilla and momentum MSE loss curves in a dictionary.
    """
    X = np.array(X)
    y = np.array(y)

    w_v = np.zeros(X.shape[1])
    w_m = np.zeros(X.shape[1])
    vt = np.zeros(X.shape[1])

    mse_vanilla = []
    mse_momentum = []

    for idx in range(n_epochs):
        
        mse_v = np.mean((X @ w_v - y) ** 2)
        mse_m = np.mean((X @ w_m - y) ** 2)

        mse_vanilla.append(mse_v)
        mse_momentum.append(mse_m)
        
        gt_v = (2 / X.shape[0]) * X.T @ (X @ w_v - y)
        gt_m = (2 / X.shape[0]) * X.T @ (X @ w_m - y)
        
        w_v = w_v - lr * gt_v

        vt = beta * vt + gt_m
        w_m = w_m - lr * vt

    return {
        "vanilla_losses": mse_vanilla,
        "momentum_losses": mse_momentum
    }