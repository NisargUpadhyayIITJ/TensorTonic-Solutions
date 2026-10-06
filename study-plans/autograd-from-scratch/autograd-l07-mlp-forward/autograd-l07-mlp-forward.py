import torch

def mlp_forward(inputs: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> tuple:
    """
    Returns the final output tensor and a list of layer-output tensors.
    """
    h = []
    a = inputs
    for weight, bias in zip(weights, biases):
        a = weight @ a + bias
        a = torch.tanh(a)
        h.append(a)
    return a, h