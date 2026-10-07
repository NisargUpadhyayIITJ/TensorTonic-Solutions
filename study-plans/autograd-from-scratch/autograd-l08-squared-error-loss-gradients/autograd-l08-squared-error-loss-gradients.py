import torch

def squared_error_loss_gradients(predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
    """
    Returns a scalar loss tensor and a prediction-shaped gradient tensor.
    """
    se_loss = torch.sum((predictions - targets) ** 2)
    grads = 2 * (predictions - targets)
    return se_loss, grads
