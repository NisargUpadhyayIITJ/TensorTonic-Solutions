import numpy as np

class RidgeRegression():
    def __init__(self):
        self.w = None

    def fit(self, X, y, lr, epochs, alpha):
        X_train = np.column_stack((np.ones(X.shape[0]), X))
        self.w = np.zeros(X_train.shape[1])

        for i in range(epochs):
            y_pred = X_train @ self.w
            gradients = (2 / X_train.shape[0]) * (X_train.T @ (y_pred - y)) + 2 * alpha * self.w
            gradients[0] = (2 / X_train.shape[0]) * (X_train.T @ (y_pred - y))[0]
            self.w = self.w - lr * gradients

def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.array(X)
    y = np.array(y)
    lreg = RidgeRegression()
    lreg.fit(X, y, lr, epochs, alpha)
    return (lreg.w[1:], lreg.w[0])
    