import numpy as np

def rmsprop(X: list, y: list, lr: float, decay: float, n_epochs: int) -> dict:
    """
    Returns MSE losses and effective learning rates in a dictionary.
    """
    X = np.array(X)
    y = np.array(y)

    w = np.zeros(X.shape[1])
    Et = np.zeros(X.shape[1])

    mse = []
    elr = []

    for idx in range(n_epochs):
        
        mse_v = np.mean((X @ w - y) ** 2)

        mse.append(mse_v)
        
        gt = (2 / X.shape[0]) * X.T @ (X @ w - y)
        
        Et = decay * Et + (1 - decay) * gt * gt
        w = w - (lr / np.sqrt(Et + 1e-8)) * gt

        elr.append(lr / np.sqrt(Et + 1e-8))

    return {
        "losses": mse,
        "effective_lrs": elr
    }