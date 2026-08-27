import numpy as np

def pca(X, n_components=2):
    """
    Returns: tuple of (transformed_data, explained_variance_ratios)
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    mean = X.mean(axis=0)
    X_centered = X - mean

    cov = (X_centered.T @ X_centered) / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    components = eigenvectors[:, :n_components]
    X_transformed = X_centered @ components

    total_var = eigenvalues.sum()
    explained = eigenvalues[:n_components] / total_var

    return ([[round(float(v), 4) for v in row] for row in X_transformed],
            [round(float(v), 4) for v in explained])
