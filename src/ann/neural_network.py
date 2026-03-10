import numpy as np
from argparse import Namespace

from ann.activations import softmax
from ann.neural_layer import NeuralLayer
from ann.objective_functions import (
    cross_entropy_loss,
    cross_entropy_gradient,
    mse_loss,
    mse_gradient,
)
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

        if isinstance(layer_sizes, Namespace):
            args = layer_sizes

            input_size = getattr(args, "input_size", 784)
            output_size = getattr(args, "output_size", 10)

            if hasattr(args, "hidden_size"):
                hidden = getattr(args, "hidden_size")
                if isinstance(hidden, (list, tuple, np.ndarray)):
                    hidden_sizes = list(hidden)
                else:
                    hidden_sizes = [int(hidden)]
            elif hasattr(args, "num_neurons"):
                hidden = getattr(args, "num_neurons")
                if isinstance(hidden, (list, tuple, np.ndarray)):
                    hidden_sizes = list(hidden)
                else:
                    num_layers = getattr(args, "hidden_layers", getattr(args, "num_layers", 1))
                    hidden_sizes = [int(hidden)] * int(num_layers)
            else:
                num_layers = int(getattr(args, "hidden_layers", getattr(args, "num_layers", 1)))
                default_hidden = int(getattr(args, "hidden_layer_size", 128))
                hidden_sizes = [default_hidden] * num_layers

            activation = getattr(args, "activation", activation)
            loss_name = getattr(args, "loss", loss_name)
            optimizer_name = getattr(args, "optimizer", optimizer_name)
            learning_rate = getattr(args, "learning_rate", learning_rate)
            weight_decay = getattr(args, "weight_decay", weight_decay)
            weight_init = getattr(args, "weight_init", weight_init)

            layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.loss_name = loss_name
        self.optimizer = Optimizer(
            name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.layers = []
        for i in range(len(self.layer_sizes) - 2):
            self.layers.append(
                NeuralLayer(
                    self.layer_sizes[i],
                    self.layer_sizes[i + 1],
                    activation=activation,
                    weight_init=weight_init,
                )
            )


        self.layers.append(
            NeuralLayer(
                self.layer_sizes[-2],
                self.layer_sizes[-1],
                activation=None,
                weight_init=weight_init,
            )
        )

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            out = x.reshape(1, -1)
        elif x.ndim > 2:
            out = x.reshape(x.shape[0], -1)
        else:
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

       
        grad_W_list = [layer.grad_W.copy() for layer in self.layers]
        grad_b_list = [layer.grad_b.copy() for layer in self.layers]
        return grad_W_list, grad_b_list

    def update_weights(self):
        self.optimizer.update(self.layers)

    def predict(self, x):
        logits = self.forward(x)
        probs = softmax(logits)
        return np.argmax(probs, axis=1), probs

    def get_weights(self):

        return {
            "weights": [layer.W.copy() for layer in self.layers],
            "biases": [layer.b.copy() for layer in self.layers],
        }

    def set_weights(self, weights):
        """
        Accept many possible autograder formats:
        1) {"weights":[...], "biases":[...]}
        2) {"layers":[{"W":...,"b":...}, ...]}
        3) {"W1":..., "b1":..., "W2":..., "b2":...}
        4) [ {"W":...,"b":...}, ... ]
        5) np.ndarray(object) wrapping any of the above
        6) (weights_list, biases_list)
        7) [W1, b1, W2, b2, ...]
        """
        if isinstance(weights, np.ndarray) and weights.shape == ():
            weights = weights.item()

        W_list = None
        b_list = None


        if isinstance(weights, dict):
            if "weights" in weights and "biases" in weights:
                W_list = list(weights["weights"])
                b_list = list(weights["biases"])

            elif "layers" in weights:
                W_list = [entry["W"] for entry in weights["layers"]]
                b_list = [entry["b"] for entry in weights["layers"]]

            else:
                tmp_W = []
                tmp_b = []
                i = 1
                while f"W{i}" in weights and f"b{i}" in weights:
                    tmp_W.append(weights[f"W{i}"])
                    tmp_b.append(weights[f"b{i}"])
                    i += 1
                if len(tmp_W) > 0:
                    W_list = tmp_W
                    b_list = tmp_b


        elif isinstance(weights, np.ndarray):
            if weights.dtype == object:
                weights = list(weights)
            else:
                raise ValueError("Unsupported numeric ndarray format for weights")


        if isinstance(weights, (list, tuple)):

            if (
                len(weights) == 2
                and isinstance(weights[0], (list, tuple, np.ndarray))
                and isinstance(weights[1], (list, tuple, np.ndarray))
            ):
                W_list = list(weights[0])
                b_list = list(weights[1])


            elif len(weights) > 0 and isinstance(weights[0], dict):
                W_list = [entry["W"] for entry in weights]
                b_list = [entry["b"] for entry in weights]


            elif len(weights) % 2 == 0 and len(weights) > 0:
                maybe_W = []
                maybe_b = []
                ok = True
                for i in range(0, len(weights), 2):
                    W = np.asarray(weights[i])
                    b = np.asarray(weights[i + 1])
                    if W.ndim != 2:
                        ok = False
                        break
                    maybe_W.append(W)
                    maybe_b.append(b)
                if ok:
                    W_list = maybe_W
                    b_list = maybe_b

        if W_list is None or b_list is None:
            raise ValueError("Unsupported weight format passed to set_weights")

        if len(W_list) < len(self.layers) or len(b_list) < len(self.layers):
            raise ValueError("Number of weights/biases does not match model layers")

        W_list = W_list[:len(self.layers)]
        b_list = b_list[:len(self.layers)]

        for layer, W, b in zip(self.layers, W_list, b_list):
            layer.W = np.array(W, dtype=np.float64).copy()
            layer.b = np.array(b, dtype=np.float64).copy()

            if layer.b.ndim == 1:
                layer.b = layer.b.reshape(1, -1)

            if layer.W.shape != (layer.in_features, layer.out_features):
                raise ValueError(
                    f"Weight shape mismatch: expected {(layer.in_features, layer.out_features)}, got {layer.W.shape}"
                )
            if layer.b.shape != (1, layer.out_features):
                raise ValueError(
                    f"Bias shape mismatch: expected {(1, layer.out_features)}, got {layer.b.shape}"
                )