import torch

def tanh_forward_backward(x: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of scalar tensors: tanh output and input gradient.
    """
    out = torch.tanh(x)
    return out, upstream_gradient * (1 - out ** 2)
