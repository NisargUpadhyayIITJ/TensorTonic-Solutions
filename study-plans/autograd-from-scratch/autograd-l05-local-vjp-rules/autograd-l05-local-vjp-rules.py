def local_vjp(operation: str, inputs: list, output: float, upstream_gradient: float) -> list:
    """
    Returns a list of float gradient contributions in input order.
    """
    if operation == "add":
        return [float(upstream_gradient), float(upstream_gradient)]
    elif(operation == "mul"):
        return [float(inputs[1] * upstream_gradient), float(inputs[0] * upstream_gradient)]
    else:
        return [float(upstream_gradient * (1 - np.tanh(inputs[0]) ** 2))]