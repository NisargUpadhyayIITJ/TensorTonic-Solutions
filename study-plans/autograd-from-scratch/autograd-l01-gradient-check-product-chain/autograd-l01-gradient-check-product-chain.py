import numpy as np

def gradient_check_product_chain(
    a: float,
    b: float,
    c: float,
    f: float,
    h: float,
) -> tuple[float, list, list, float]:
    """
    Returns loss, analytic gradients, numerical gradients, and maximum error.
    """
    values = (a, b, c, f, h)
    a_value, b_value, c_value, f_value, h_value = (float(value) for value in values)

    def loss(a_input, b_input, c_input, f_input):
        result = (a_input * b_input + c_input) * f_input
        return float(result)
    base_loss = loss(a_value, b_value, c_value, f_value)
    intermediate = a_value * b_value + c_value
    analytic = [b_value * f_value, a_value * f_value, f_value, intermediate]
    numerical = [(loss(a_value + h_value, b_value, c_value, f_value) - base_loss) / h_value, (loss(a_value, b_value + h_value, c_value, f_value) - base_loss) / h_value, (loss(a_value, b_value, c_value + h_value, f_value) - base_loss) / h_value, (loss(a_value, b_value, c_value, f_value + h_value) - base_loss) / h_value]
    max_error = float(np.max(np.abs(np.asarray(analytic) - np.asarray(numerical))))
    return (base_loss, [float(value) for value in analytic], [float(value) for value in numerical], max_error)
