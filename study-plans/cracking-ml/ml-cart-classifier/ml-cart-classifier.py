import numpy as np
from collections import Counter

class DecisionTreeClassifier():
    def __init__(self):
        self.max_depth = None
        self.min_sample_split = None
        self.tree = None

    def gini(self, y):
        values, counts = np.unique(y, return_counts=True)
        np_counts = np.array(counts)
        probas = np_counts / y.shape[0]
        return 1.0 - np.sum(probas ** 2)

    def split_gini(self, y_left, y_right):
        left_gini = self.gini(y_left)
        right_gini = self.gini(y_right)
        l = y_left.shape[0]
        r = y_right.shape[0]
        return (l * left_gini + r * right_gini) / (l + r)

    def best_split(self, X, y):
        best_gini = float("inf")
        best_threshold = None
        best_feature = None
        
        for feature in range(0, X.shape[1]):
            unique_thresholds = np.unique(X[:, feature])
            for threshold in unique_thresholds:
                threshold = np.float64(threshold)
                y_left = y[X[:, feature] <= threshold]
                y_right = y[X[:, feature] > threshold]
                if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                    continue
                
                current_gini = self.split_gini(y_left, y_right)
                if(current_gini < best_gini):
                    best_gini = current_gini
                    best_threshold = threshold
                    best_feature = feature

        if best_feature is None:
            return None
            
        return (best_threshold, best_gini, best_feature)

    def majority_class(self, y):
        if(y.shape[0] == 0):
          return 0
        counts = Counter(y)
        max_count = max(counts.values())
        best_label = min(label for label, c in counts.items() if c == max_count)
        return best_label

    def build_tree(self, X, y, depth):
        predictions = self.majority_class(y)
        
        if(np.unique(y).shape[0] == 1):
            return {
                "leaf" : True,
                "predictions" : predictions
            }

        if(depth >= self.max_depth):
            return {
                "leaf" : True,
                "predictions" : predictions
            }

        if(X.shape[0] < self.min_samples_split):
            return {
                "leaf" : True,
                "predictions" : predictions
            }

        parent_gini = self.gini(y)

        split = self.best_split(X, y)
        
        if split is None:
            return {
                "leaf": True,
                "predictions": predictions
            }
        
        threshold, gini_impurity, feature = split
        
        parent_gini = self.gini(y)
        
        if gini_impurity >= parent_gini:
            return {
                "leaf": True,
                "predictions": predictions
            }

        X_left = X[X[:, feature] <= threshold]
        y_left = y[X[:, feature] <= threshold]

        X_right = X[X[:, feature] > threshold]
        y_right = y[X[:, feature] > threshold]

        left = self.build_tree(X_left, y_left, depth + 1)
        right = self.build_tree(X_right, y_right, depth + 1)

        return {
            "leaf" : False,
            "gini" : gini_impurity,
            "feature" : feature,
            "threshold" : threshold,
            "left" : left,
            "right" : right
        }

    def fit(self, X_train, y_train, max_depth, min_samples):
        self.max_depth = max_depth
        self.min_samples_split = min_samples
        self.tree = self.build_tree(X_train, y_train, 0)

    def predict(self, X_test):
        preds = []
        for X in X_test:
            tree = self.tree
            while(tree["leaf"] == False):
                feature = tree["feature"]
                threshold = tree["threshold"]
                if(X[feature] <= threshold):
                    tree = tree["left"]
                else: 
                    tree = tree["right"]
                    
            preds.append(tree["predictions"])
        return preds

def cart_classify(X_train, y_train, X_test, max_depth=5, min_samples=2):
    """
    Returns: list of predicted class labels for each test point
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train, max_depth, min_samples)
    # print(dt.tree)
    return dt.predict(X_test)
    
