import numpy as np
from ann.activations import softmax
from ann.neural_layer import NeuralLayer
from ann.objective_functions import cross_entropy_loss, cross_entropy_gradient, mse_loss, mse_gradient
from ann.optimizers import Optimizer


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes,
        activation="relu",
        loss_name="cross_entropy",
        optimizer_name="sgd",
        learning_rate=0.001,
        weight_decay=0.0,
        weight_init="xavier",
    ):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.loss_name = loss_name
        self.optimizer = Optimizer(
            name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.layers = []
        for i in range(len(layer_sizes) - 2):
            self.layers.append(
                NeuralLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    activation=activation,
                    weight_init=weight_init,
                )
            )
        self.layers.append(
            NeuralLayer(
                layer_sizes[-2],
                layer_sizes[-1],
                activation=None,
                weight_init=weight_init,
            )
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def compute_loss(self, y_true, logits):
        if self.loss_name == "cross_entropy":
            return cross_entropy_loss(y_true, logits)
        if self.loss_name == "mean_squared_error":
            probs = softmax(logits)
            return mse_loss(y_true, probs)
        raise ValueError(f"Unsupported loss: {self.loss_name}")

    def backward(self, y_true, logits):
        if self.loss_name == "cross_entropy":
            grad = cross_entropy_gradient(y_true, logits)
        elif self.loss_name == "mean_squared_error":
            probs = softmax(logits)
            grad = mse_gradient(y_true, probs)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self):
        self.optimizer.update(self.layers)

    def predict(self, x):
        logits = self.forward(x)
        probs = softmax(logits)
        return np.argmax(probs, axis=1), probs

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append({"W": layer.W, "b": layer.b})
        return np.array(weights, dtype=object)

    def set_weights(self, weights):
        for layer, saved in zip(self.layers, weights):
            layer.W = saved["W"]
            layer.b = saved["b"]
