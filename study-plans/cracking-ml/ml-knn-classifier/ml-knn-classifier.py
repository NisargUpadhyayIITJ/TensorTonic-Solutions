import numpy as np
from collections import Counter

class KNNClassifier():
    def __init__(self):
        self.k = None

    def predict(self, X_train, y_train, X_test, k):
        self.k = k
        labels = []
        for i in range(X_test.shape[0]):
            dist = np.sqrt(np.sum((X_test[i] - X_train) ** 2, axis=1))
            top_k = np.argsort(np.array(dist))[:k]
            counts = Counter(y_train[top_k])
            max_count = max(counts.values())
            best_label = min(label for label, c in counts.items() if c == max_count)
            labels.append(best_label)
        return labels
            
def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    knn = KNNClassifier()
    labels = knn.predict(X_train, y_train, X_test, k)
    return labels
