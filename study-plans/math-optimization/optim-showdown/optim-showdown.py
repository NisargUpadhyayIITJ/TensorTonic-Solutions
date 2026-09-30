import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def optimizer_showdown(X: list, y: list, lr: float, beta1: float, beta2: float, epsilon: float, n_epochs: int, batch_size: int) -> dict:
    """
    Returns SGD, momentum, and Adam BCE curves in a dictionary.
    """
    X = np.array(X)
    y = np.array(y)

    X = np.column_stack((np.ones(X.shape[0]), X))

    w_sgd = np.zeros(X.shape[1])
    w_mom = np.zeros(X.shape[1])
    w_adam = np.zeros(X.shape[1])

    sgd_losses = []
    momentum_losses = []
    adam_losses = []

    v = np.zeros(X.shape[1])
    first = np.zeros(X.shape[1])
    second = np.zeros(X.shape[1])
    t = 1

    for epoch in range(1, n_epochs + 1):
        # sgd
        for idx in range(0, X.shape[0], batch_size):
            batch_len = min(batch_size, X.shape[0] - idx)
        
            x_sample = X[idx: idx + batch_len]
            y_sample = y[idx: idx + batch_len]
        
            g = (1 / batch_len) * x_sample.T @ (sigmoid(x_sample @ w_sgd) - y_sample)
            w_sgd -= lr * g

        y_pred = sigmoid(X @ w_sgd)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        sgd_losses.append(loss)

        # momentum
        for idx in range(0, X.shape[0], batch_size):
            batch_len = min(batch_size, X.shape[0] - idx)

            x_sample = X[idx: idx + batch_len]
            y_sample = y[idx: idx + batch_len]

            g = (1 / batch_len) * x_sample.T @ (sigmoid(x_sample @ w_mom) - y_sample)
            v = beta1 * v + g
            w_mom -= lr * v

        y_pred = sigmoid(X @ w_mom)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        momentum_losses.append(loss)
        
        # adam
        for idx in range(0, X.shape[0], batch_size):
            batch_len = min(batch_size, X.shape[0] - idx)

            x_sample = X[idx: idx + batch_len]
            y_sample = y[idx: idx + batch_len]

            g = (1 / batch_len) * x_sample.T @ (sigmoid(x_sample @ w_adam) - y_sample)
            first = beta1 * first + (1 - beta1) * g
            second = beta2 * second + (1 - beta2) * g ** 2
            corrected_first = first / (1 - beta1 ** t)
            corrected_second = second / (1 - beta2 ** t)
            t += 1
            w_adam -= lr * corrected_first / (np.sqrt(corrected_second) + epsilon)

        y_pred = sigmoid(X @ w_adam)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        adam_losses.append(loss)

    return {
        "sgd_losses": sgd_losses,
        "momentum_losses": momentum_losses,
        "adam_losses": adam_losses
    }
        