import numpy as np

class LogisticRegression():

    def __init__(self):
        self.w = None

    def fit(self, X, y, lr, epochs):
        X_train = np.column_stack((np.ones(X.shape[0]), X))
        self.w = np.zeros(X_train.shape[1])

        for i in range(epochs):
            z = (X_train @ self.w)
            y_pred = 1 / (1 + np.exp(-z))
            gradients = (1 / X_train.shape[0]) * (X_train.T @ (y_pred - y))
            self.w = self.w - lr * gradients


def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.array(X)
    y = np.array(y)
    lreg = LogisticRegression()
    lreg.fit(X, y, lr, n_iters)
    return (lreg.w[1:], lreg.w[0])
