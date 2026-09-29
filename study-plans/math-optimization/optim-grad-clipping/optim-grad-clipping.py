import numpy as np

def gradient_clipping(X: list, y: list, lr: float, clip_value: float, clip_norm: float, n_epochs: int) -> dict:
    """
    Returns unclipped, value-clipped, and norm-clipped MSE curves.
    """
    X = np.array(X)
    y = np.array(y)

    X = np.column_stack((np.ones(X.shape[0]), X))
    w = np.zeros(X.shape[1])
    w_clip_value = np.zeros(X.shape[1])
    w_global_norm = np.zeros(X.shape[1])

    unclipped_losses = []
    clip_value_losses = []
    clip_norm_losses = []
    
    for epoch in range(n_epochs):
        gw = (2 / X.shape[0]) * X.T @ (X @ w - y)
        w = w - lr * gw

        gv = (2 / X.shape[0]) * X.T @ (X @ w_clip_value - y)
        gv = np.clip(gv, -clip_value, clip_value)
        w_clip_value = w_clip_value - lr * gv

        gg = (2 / X.shape[0]) * X.T @ (X @ w_global_norm - y)
        euc_norm = np.linalg.norm(gg)
        if(euc_norm > clip_norm):
            gg = gg * clip_norm / euc_norm
        w_global_norm = w_global_norm - lr * gg

        unclipped_losses.append(np.mean((X @ w - y) ** 2))
        clip_value_losses.append(np.mean((X @ w_clip_value - y) ** 2))
        clip_norm_losses.append(np.mean((X @ w_global_norm - y) ** 2))

    return {
        "unclipped_losses": unclipped_losses,
        "clip_value_losses": clip_value_losses,
        "clip_norm_losses": clip_norm_losses
    }