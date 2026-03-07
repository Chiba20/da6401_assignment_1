import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative_from_activation(a):
    return a * (1.0 - a)


def tanh(x):
    return np.tanh(x)


def tanh_derivative_from_activation(a):
    return 1.0 - a ** 2


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative_from_preactivation(z):
    return (z > 0).astype(np.float64)


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def get_activation(name):
    if name == "sigmoid":
        return sigmoid
    if name == "tanh":
        return tanh
    if name == "relu":
        return relu
    if name is None:
        return None
    raise ValueError(f"Unsupported activation: {name}")


def get_activation_derivative(name):
    if name == "sigmoid":
        return sigmoid_derivative_from_activation
    if name == "tanh":
        return tanh_derivative_from_activation
    if name == "relu":
        return relu_derivative_from_preactivation
    if name is None:
        return None
    raise ValueError(f"Unsupported activation: {name}")
