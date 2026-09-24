import numpy as np

def adam(X: list, y: list, lr: float, beta1: float, beta2: float, n_epochs: int) -> dict:
    """
    Returns pre-update MSE losses and final weights in a dictionary.
    """
    features = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    n, d = features.shape
    weights = np.zeros(d, dtype=np.float64)
    first = np.zeros(d, dtype=np.float64)
    second = np.zeros(d, dtype=np.float64)
    losses = []
    for timestep in range(1, n_epochs + 1):
        residual = features @ weights - targets
        losses.append(float(np.mean(residual ** 2)))
        gradient = 2.0 / n * features.T @ residual
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient ** 2
        corrected_first = first / (1.0 - beta1 ** timestep)
        corrected_second = second / (1.0 - beta2 ** timestep)
        weights -= lr * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return {"losses": losses, "final_weights": weights.tolist()}