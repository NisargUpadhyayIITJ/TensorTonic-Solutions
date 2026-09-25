import numpy as np

def linear_warmup(X: list, y: list, base_lr: float, warmup_epochs: int, total_epochs: int) -> dict:
    """
    Returns the learning-rate schedule and pre-update losses.
    """
    X = np.array(X)
    y = np.array(y)

    mse_loss = []
    lr_schedule = []
    lr = base_lr

    w = np.zeros(X.shape[1])
        
    for idx in range(total_epochs):
        mse = np.mean((X @ w - y) ** 2)
        
        gt = (2 / X.shape[0]) * X.T @ (X @ w - y)
        lr = base_lr * min((idx + 1) / warmup_epochs, 1)

        mse_loss.append(mse)
        lr_schedule.append(lr)
        w = w - lr * gt
        
    return {
        "lr_schedule": lr_schedule,
        "losses": mse_loss
    }
    