import torch

def scalar_neuron_module(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns a scalar tensor preserving the input dtype and device.
    """
    a = (inputs * weights).sum() + bias
    if(nonlinear):
        a = torch.tanh(a)
    return a
