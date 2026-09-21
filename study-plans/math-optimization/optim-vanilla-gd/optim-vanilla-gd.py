def vanilla_gradient_descent(x0: float, y0: float, lr: float, n_iters: int) -> dict:
    """
    Returns the trajectory, final point, and final function value.
    """
    trajectory = [[x0, y0]]
    for idx in range(n_iters):
        x0 = x0 - lr * 2 * x0
        y0 = y0 - lr * 6 * y0
        trajectory.append([x0, y0])

    final_point = [x0, y0]
    final_value = x0 ** 2 + 3 * y0 ** 2

    return {
        "trajectory": trajectory,
        "final_point": final_point,
        "final_value": final_value
    }