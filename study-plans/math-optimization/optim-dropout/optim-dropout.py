import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dropout_mlp(X_train: list, y_train: list, X_test: list, y_test: list, hidden_size: int, lr: float, dropout_rate: float, n_epochs: int, seed: int) -> list:
    """
    Returns one post-update test accuracy per epoch.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test)
    y_test = np.array(y_test).reshape(-1, 1)

    np.random.seed(seed)

    w1 = np.random.randn(X_train.shape[1], hidden_size) * np.sqrt(2 / X_train.shape[1])
    b1 = np.zeros(hidden_size)

    w2 = np.random.randn(hidden_size, 1) * np.sqrt(2 / hidden_size)
    b2 = np.zeros(1)

    test_acc = []

    for epoch in range(n_epochs):
        # forward
        # mask1 = rng.random(X_tr.shape) > dropout_rate 
        # X_train = X_tr * mask1 / (1 - dropout_rate)
        z1 = X_train @ w1 + b1
        a1 = np.maximum(0, z1)

        mask2 = np.random.rand(*a1.shape) > dropout_rate
        a1 = a1 * mask2 / (1 - dropout_rate)
        z2 = a1 @ w2 + b2  
        a2 = sigmoid(z2)

        # backprop
        db2 = np.mean((a2 - y_train))
        dw2 = (1 / X_train.shape[0]) * a1.T @ (a2 - y_train)
        relu_z1 = (np.where(z1 > 0.0, 1.0, 0.0) * mask2) / (1 - dropout_rate)
        db1 =  np.mean((a2 - y_train) @ w2.T * relu_z1 , axis=0)
        dw1 = (1 / X_train.shape[0]) * X_train.T @ ((a2 - y_train) @ w2.T * relu_z1)

        # Update
        w1 = w1 - lr * dw1
        b1 = b1 - lr * db1
        w2 = w2 - lr * dw2
        b2 = b2 - lr * db2

        y_pred = sigmoid(np.maximum(0, X_test @ w1 + b1) @ w2 + b2) > 0.5
        acc = np.mean(y_pred == y_test)
        test_acc.append(acc)

    return test_acc
        