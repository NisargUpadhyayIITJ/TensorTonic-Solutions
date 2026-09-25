import numpy as np

def learning_rate_sweep(X: list, y: list, learning_rates: list, n_epochs: int) -> list:
    """
    Returns one pre-update MSE loss curve per learning rate.
    """
    X = np.array(X)
    y = np.array(y)

    mse_loss = []

    for lr in learning_rates:
        mse_lr = []
        w = np.zeros(X.shape[1])
        
        for idx in range(n_epochs):
            mse = np.mean((X @ w - y) ** 2)
            mse_lr.append(mse)
            
            gt = (2 / X.shape[0]) * X.T @ (X @ w - y)
            w = w - lr * gt

        mse_loss.append(mse_lr)


    return mse_loss