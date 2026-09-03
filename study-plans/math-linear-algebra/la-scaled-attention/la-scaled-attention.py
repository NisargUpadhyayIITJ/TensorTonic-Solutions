import numpy as np

def softmax(temp):
    temp = temp - np.max(temp, axis=1, keepdims=True)
    exp_temp = np.exp(temp)
    return exp_temp / np.sum(exp_temp, axis=1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: ndarray, the attention output softmax(Q @ K.T / sqrt(d_k)) @ V.
    """
    Q = np.array(Q)
    K = np.array(K)
    V = np.array(V)
    temp = (Q @ K.T) / np.sqrt(Q.shape[1])
    temp = softmax(temp)
    return temp @ V