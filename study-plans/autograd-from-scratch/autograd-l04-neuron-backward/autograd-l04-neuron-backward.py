import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    a = torch.dot(inputs, weights) + bias
    output = torch.tanh(a)
    input_gradients = upstream_gradient * (1 - output ** 2) * weights
    weight_gradients = upstream_gradient * (1 - output ** 2) * inputs
    return output, input_gradients, weight_gradients, upstream_gradient * (1 - output ** 2)
