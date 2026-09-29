import numpy as np

def newtons_method(X: list, y: list, lr_gd: float, n_iters: int) -> dict:
    """
    Returns gradient-descent and Newton MSE curves in a dictionary.
    """
    X = np.array(X)
    y = np.array(y)

    X = np.column_stack((np.ones(X.shape[0]), X))

    w_v = np.zeros(X.shape[1])
    w_n = np.zeros(X.shape[1])

    gd_losses = [np.mean((X @ w_v - y) ** 2)]
    newton_losses = [np.mean((X @ w_n - y) ** 2)]

    H = (2 / X.shape[0]) * X.T @ X 
    
    for epoch in range(n_iters):
        gv = (2 / X.shape[0]) * X.T @ (X @ w_v - y)
        w_v = w_v - lr_gd * gv

        gn = (2 / X.shape[0]) * X.T @ (X @ w_n - y)
        w_n = w_n - np.linalg.pinv(H) @ gn

        gd_losses.append(np.mean((X @ w_v - y) ** 2))
        newton_losses.append(np.mean((X @ w_n - y) ** 2))      

    return {
        "gd_losses": gd_losses,
        "newton_losses": newton_losses
    }
        