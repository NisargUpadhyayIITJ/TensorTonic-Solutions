import numpy as np

def cosine_restarts(X: list, y: list, eta_max: float, eta_min: float, T_0: int, T_mult: int, total_epochs: int) -> dict:
    """
    Returns the cosine-restart schedule and pre-update losses.
    """
    X = np.array(X)
    y = np.array(y)

    mse_loss = []
    lr_schedule = []
    lr = eta_min + ((eta_max - eta_min) / 2) * (1 + np.cos(np.pi * 0 / T_0))
    Ti = T_0
    T_cur = 0

    w = np.zeros(X.shape[1])
        
    for idx in range(total_epochs):
        mse = np.mean((X @ w - y) ** 2)
        
        gt = (2 / X.shape[0]) * X.T @ (X @ w - y)
        if(T_cur == Ti):
            T_cur = 0
            Ti = Ti * T_mult
        lr = eta_min + ((eta_max - eta_min) / 2) * (1 + np.cos(np.pi * T_cur / Ti))
        T_cur = T_cur + 1
        
        mse_loss.append(mse)
        lr_schedule.append(lr)
        w = w - lr * gt
        
    return {
        "lr_schedule": lr_schedule,
        "losses": mse_loss
    }