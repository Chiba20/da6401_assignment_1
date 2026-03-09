import numpy as np
from argparse import Namespace

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
        # Support autograder style: NeuralNetwork(args_namespace)
        if isinstance(layer_sizes, Namespace):
            args = layer_sizes

            # support multiple possible autograder field names
            input_size = getattr(args, "input_size", 784)
            output_size = getattr(args, "output_size", 10)

            if hasattr(args, "hidden_size"):
                hidden_sizes = list(getattr(args, "hidden_size"))
            elif hasattr(args, "num_neurons"):
                hidden_sizes = list(getattr(args, "num_neurons"))
            else:
                num_layers = getattr(args, "num_layers", 1)
                default_hidden = getattr(args, "hidden_layer_size", 128)
                hidden_sizes = [default_hidden] * num_layers

            activation = getattr(args, "activation", activation)
            loss_name = getattr(args, "loss", loss_name)
            optimizer_name = getattr(args, "optimizer", optimizer_name)
            learning_rate = getattr(args, "learning_rate", learning_rate)
            weight_decay = getattr(args, "weight_decay", weight_decay)
            weight_init = getattr(args, "weight_init", weight_init)

            layer_sizes = [input_size] + hidden_sizes + [output_size]

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

        # output layer (linear logits)
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
            weights.append({
                "W": layer.W.copy(),
                "b": layer.b.copy()
            })
        return np.array(weights, dtype=object)

    def set_weights(self, weights):
        for layer, saved in zip(self.layers, weights):
            layer.W = saved["W"].copy()
            layer.b = saved["b"].copy()