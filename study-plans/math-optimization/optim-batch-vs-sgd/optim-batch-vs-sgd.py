import numpy as np

def batch_gd_compare(X: list, y: list, batch_sizes: list, n_epochs: int, lr: float, seed: int) -> list:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n, d = X.shape
    mse_loss = []

    for batch_size in batch_sizes:
        w = np.zeros(d, dtype=np.float64)
        rng = np.random.RandomState(seed)
        losses = []

        for _ in range(n_epochs):
            indices = rng.permutation(n)

            for start in range(0, n, batch_size):
                batch = indices[start:start + batch_size]

                xt = X[batch]
                yt = y[batch]

                grad = (2.0 / len(batch)) * xt.T @ (xt @ w - yt)
                w = w - lr * grad

            losses.append(float(np.mean((X @ w - y) ** 2)))

        mse_loss.append(losses)

    return mse_loss