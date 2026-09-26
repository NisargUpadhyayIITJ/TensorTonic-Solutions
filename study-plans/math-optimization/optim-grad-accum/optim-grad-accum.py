import numpy as np

def gradient_accumulation(X: list, y: list, lr: float, micro_batch: int, accum_steps: int, n_epochs: int) -> dict:
    """
    Returns both loss curves and final weight vectors in a dictionary.
    """
    X = np.array(X)
    y = np.array(y)

    w = np.zeros(X.shape[1])
    w_ga = np.zeros(X.shape[1])

    no_accum_losses = []
    accum_losses = []

    for epoch in range(n_epochs):
        for idx in range(0, X.shape[0], micro_batch):
            batch_len = min(micro_batch, X.shape[0] - idx)
            Xt = X[idx: idx + batch_len]
            yt = y[idx: idx + batch_len]

            gt = (2 / batch_len) * Xt.T @ (Xt @ w - yt)

            w = w - lr * gt

        mse = np.mean((X @ w - y) ** 2)

        no_accum_losses.append(mse)

    for epoch in range(n_epochs):
        for idx in range(0, X.shape[0], micro_batch * accum_steps):
            batch_len = min(micro_batch * accum_steps, X.shape[0] - idx)
            gt = 0
            
            for step_idx in range(idx, idx + batch_len, micro_batch):
                micro_len = min(micro_batch, idx + batch_len - step_idx)
                Xt = X[step_idx: step_idx + micro_len]
                yt = y[step_idx: step_idx + micro_len]
    
                gt = gt + (2) * Xt.T @ (Xt @ w_ga - yt)

            w_ga = w_ga - lr * (gt / batch_len)

        mse_ga = np.mean((X @ w_ga - y) ** 2)

        accum_losses.append(mse_ga)

    return {
        "no_accum_losses": no_accum_losses,
        "accum_losses": accum_losses,
        "no_accum_weights": w.tolist(),
        "accum_weights": w_ga.tolist()
    }

            
        