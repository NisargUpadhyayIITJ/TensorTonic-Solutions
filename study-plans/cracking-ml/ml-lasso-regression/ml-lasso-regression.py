import numpy as np

class LassoRegression():
    def __init__(self):
        self.w = None

    def fit(self, X, y, lr, epochs, alpha):
        X_train = np.column_stack((np.ones(X.shape[0]), X))
        self.w = np.zeros(X_train.shape[1])

        for i in range(epochs):
            y_pred = X_train @ self.w
            gradients = (2 / X_train.shape[0]) * (X_train.T @ (y_pred - y)) + alpha * np.sign(self.w)
            gradients[0] = (2 / X_train.shape[0]) * (X_train.T @ (y_pred - y))[0]
            self.w = self.w - lr * gradients

def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    X = np.array(X)
    y = np.array(y)
    lreg = LassoRegression()
    lreg.fit(X, y, lr, epochs, alpha)
    return (lreg.w[1:], lreg.w[0])