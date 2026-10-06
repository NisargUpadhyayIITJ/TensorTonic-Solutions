import torch

def recursive_parameter_collection(weights: list[torch.Tensor], biases: list[torch.Tensor]) -> tuple:
    """
    Returns a list of scalar parameter views and its integer count.
    """
    parameters = []
    for weight, bias in zip(weights, biases):
        for neuron_index in range(weight.shape[0]):
            for input_index in range(weight.shape[1]):
                parameters.append(weight[neuron_index, input_index])
            parameters.append(bias[neuron_index])
    return (parameters, len(parameters))
