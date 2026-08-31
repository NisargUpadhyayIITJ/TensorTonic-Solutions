import numpy as np

class Kmeans():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = None
        self.labels = None

    def assign_clusters(self, X):
        labels = []
        for row in X:
            distances = []
            for centroid in self.centroids:
                dist = np.sqrt(np.sum((row - centroid) ** 2))
                distances.append(dist)
            np_dist = np.array(distances)
            idx = np.argmin(np_dist)
            labels.append(idx)
        return np.array(labels)

    def move_centroids(self, X):
        np_labels = np.array(self.labels)
        new_centroids = []

        for i in range(self.n_clusters):
            points = X[np_labels == i]
        
            if len(points) > 0:
                centroid = np.mean(points, axis=0)
            else:
                # Keep the old centroid if the cluster is empty
                centroid = self.centroids[i]
        
            new_centroids.append(centroid)
        
        return np.array(new_centroids)

    def fit(self, X, k, max_iters):
        # self.centroids = X[np.random.randint(0, high = X.shape[0], size = k)]
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[indices]
        # print(self.n_clusters)
        for i in range(max_iters):
            self.labels = self.assign_clusters(X)
            self.centroids = self.move_centroids(X)
        # return self.labels.to_list(), self.n_centroids
        # pass

def kmeans(X, k, max_iters=100, seed=42):
    """
    Returns: tuple of (labels as list[int], centroids as list[list[float]])
    """
    np.random.seed(seed)
    X = np.array(X)
    kmeans = Kmeans(k)
    kmeans.fit(X, k, max_iters)
    return list(kmeans.labels), np.round(kmeans.centroids, decimals=4)
    
    
    
    
