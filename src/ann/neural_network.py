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

        # output layer
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

        return {
            "weights": [layer.W.copy() for layer in self.layers],
            "biases": [layer.b.copy() for layer in self.layers]
        }

    def set_weights(self, weights):
        # Case 1: np.load(..., allow_pickle=True) may return 0-d object array
        if isinstance(weights, np.ndarray) and weights.shape == ():
            weights = weights.item()

        # Case 2: dict format {"weights":[...], "biases":[...]}
        if isinstance(weights, dict):
            if "weights" in weights and "biases" in weights:
                W_list = weights["weights"]
                b_list = weights["biases"]

            # Support dict like {"layers":[{"W":...,"b":...}, ...]}
            elif "layers" in weights:
                W_list = [entry["W"] for entry in weights["layers"]]
                b_list = [entry["b"] for entry in weights["layers"]]

            # Support dict like {"W1":..., "b1":..., "W2":..., "b2":...}
            else:
                W_list = []
                b_list = []
                i = 1
                while f"W{i}" in weights and f"b{i}" in weights:
                    W_list.append(weights[f"W{i}"])
                    b_list.append(weights[f"b{i}"])
                    i += 1

                if len(W_list) == 0:
                    raise ValueError("Unsupported dict weight format")

        # Case 3: list/tuple of dicts [{"W":...,"b":...}, ...]
        elif isinstance(weights, (list, tuple)) and len(weights) > 0 and isinstance(weights[0], dict):
            W_list = [entry["W"] for entry in weights]
            b_list = [entry["b"] for entry in weights]

        # Case 4: object array of dicts
        elif isinstance(weights, np.ndarray) and weights.dtype == object:
            weights_list = list(weights)
            if len(weights_list) > 0 and isinstance(weights_list[0], dict):
                W_list = [entry["W"] for entry in weights_list]
                b_list = [entry["b"] for entry in weights_list]
            else:
                raise ValueError("Unsupported ndarray weight format")

        else:
            raise ValueError("Unsupported weight format passed to set_weights")

        if len(W_list) != len(self.layers) or len(b_list) != len(self.layers):
            raise ValueError("Number of weight matrices / bias vectors does not match model layers")

        for layer, W, b in zip(self.layers, W_list, b_list):
            layer.W = np.array(W, dtype=np.float64).copy()
            layer.b = np.array(b, dtype=np.float64).copy()
            if layer.b.ndim == 1:
                layer.b = layer.b.reshape(1, -1)