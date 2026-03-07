import numpy as np
from ann.activations import get_activation, get_activation_derivative


class NeuralLayer:
    def __init__(self, in_features, out_features, activation=None, weight_init="xavier"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.activation = get_activation(activation)
        self.activation_derivative = get_activation_derivative(activation)

        if weight_init == "xavier":
            scale = np.sqrt(1.0 / in_features)
            self.W = np.random.randn(in_features, out_features) * scale
        elif weight_init == "random":
            self.W = np.random.randn(in_features, out_features) * 0.01
        elif weight_init == "zeros":
            self.W = np.zeros((in_features, out_features), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported weight_init: {weight_init}")

        self.b = np.zeros((1, out_features), dtype=np.float64)

        self.input_cache = None
        self.z_cache = None
        self.a_cache = None

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        self.input_cache = x
        self.z_cache = x @ self.W + self.b

        if self.activation is None:
            self.a_cache = self.z_cache
        else:
            self.a_cache = self.activation(self.z_cache)

        return self.a_cache

    def backward(self, upstream_gradient):
        if self.activation_name is None:
            dz = upstream_gradient
        elif self.activation_name == "relu":
            dz = upstream_gradient * self.activation_derivative(self.z_cache)
        else:
            dz = upstream_gradient * self.activation_derivative(self.a_cache)

        self.grad_W = self.input_cache.T @ dz
        self.grad_b = np.sum(dz, axis=0, keepdims=True)
        grad_input = dz @ self.W.T
        return grad_input