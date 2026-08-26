import numpy as np

def softmax(z):
    sum = np.sum(np.exp(z), axis=1, keepdims=True)
    return np.exp(z) / sum

class SoftmaxRegression():

    def __init__(self):
        self.w = None

    def fit(self, X, y, n_classes, lr, epochs):
        X_train = np.column_stack((np.ones(X.shape[0]), X))
        self.w = np.zeros((X_train.shape[1], n_classes))

        for i in range(epochs):
            z = (X_train @ self.w)
            z = z - np.max(z, axis=1, keepdims=True)
            y_pred = softmax(z)
            gradients = (1 / X_train.shape[0]) * (X_train.T @ (y_pred - y))
            self.w = self.w - lr * gradients


def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X)
    # y = np.array(y)
    y = np.eye(n_classes)[y]
    softreg = SoftmaxRegression()
    softreg.fit(X, y, n_classes, lr, n_iters)
    return (softreg.w[1:, :], softreg.w[0, :])
