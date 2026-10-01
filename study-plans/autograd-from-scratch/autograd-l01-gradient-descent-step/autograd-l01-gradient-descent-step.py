import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    values = np.array(values)
    gradients = np.array(gradients)
    new_values = values - learning_rate * gradients
    predicted_change = np.sum(gradients * (new_values - values))
    return (new_values.tolist(), float(predicted_change))