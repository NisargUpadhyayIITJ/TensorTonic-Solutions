import numpy as np

class LinearRegression():

    def __init__(self):
        self.w = None

    def fit(self, X, y, lr, epochs):
        X_train = np.column_stack((np.ones(X.shape[0]), X))
        self.w = np.zeros(X_train.shape[1])

        for i in range(epochs):
            y_pred = X_train @ self.w
            gradients = (2 / X_train.shape[0]) * (X_train.T @ (y_pred - y))
            self.w = self.w - lr * gradients

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.array(X)
    y = np.array(y)
    lreg = LinearRegression()
    lreg.fit(X, y, lr, epochs)
    return (lreg.w[1:], lreg.w[0])
    
