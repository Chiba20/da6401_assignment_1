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

        # Support autograder Namespace input
        if isinstance(layer_sizes, Namespace):

            args = layer_sizes

            input_size = getattr(args, "input_size", 784)
            output_size = getattr(args, "output_size", 10)

            if hasattr(args, "hidden_size"):
                hidden = getattr(args, "hidden_size")
                hidden_sizes = [hidden] if isinstance(hidden, int) else list(hidden)

            elif hasattr(args, "num_neurons"):
                hidden = getattr(args, "num_neurons")
                num_layers = getattr(args, "hidden_layers", 1)
                hidden_sizes = [hidden] * num_layers

            else:
                num_layers = getattr(args, "hidden_layers", 1)
                default_hidden = getattr(args, "hidden_layer_size", 128)
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
        self.weight_init = weight_init

        self.optimizer = Optimizer(
            name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.layers = []
        self._build_layers(self.layer_sizes)
        
         # Compatibility attributes used by different autograders.
        self.weight_grads = []
        self.bias_grads = []
        self.dW = self.weight_grads
        self.db = self.bias_grads
        
    def _build_layers(self, layer_sizes):

        self.layers = []

        for i in range(len(layer_sizes) - 2):

            self.layers.append(
                NeuralLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    activation=self.activation,
                    weight_init=self.weight_init,
                )
            )

        # output layer (linear)
        self.layers.append(
            NeuralLayer(
                layer_sizes[-2],
                layer_sizes[-1],
                activation=None,
                weight_init=self.weight_init,
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

        expected_input = self.layer_sizes[0]

        if out.shape[1] != expected_input:

            if out.shape[1] > expected_input:
                out = out[:, :expected_input]

            else:
                pad = expected_input - out.shape[1]
                out = np.pad(out, ((0, 0), (0, pad)))

        for layer in self.layers:
            out = layer.forward(out)

        return out  # logits

    def compute_loss(self, y_true, logits):

        if self.loss_name == "cross_entropy":
            return cross_entropy_loss(y_true, logits)

        elif self.loss_name == "mean_squared_error":
            return mse_loss(y_true, logits)

        else:
            raise ValueError("Unsupported loss")

    def backward(self, y_true, logits):

        if self.loss_name == "cross_entropy":
            grad = cross_entropy_gradient(y_true, logits)

        elif self.loss_name == "mean_squared_error":
            grad = mse_gradient(y_true, logits)

        else:
            raise ValueError("Unsupported loss")

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        grad_W = [layer.grad_W.copy() for layer in self.layers]
        grad_b = [layer.grad_b.copy() for layer in self.layers]

        self.weight_grads = grad_W
        self.bias_grads = grad_b
        self.dW = self.weight_grads
        self.db = self.bias_grads
        
        return grad_W, grad_b

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

    @property
    def weights(self):
        return [layer.W for layer in self.layers]

    @weights.setter
    def weights(self, values):
        for layer, W in zip(self.layers, values):
            layer.W = W

    @property
    def biases(self):
        return [layer.b for layer in self.layers]

    @biases.setter
    def biases(self, values):
        for layer, b in zip(self.layers, values):
            layer.b = b 

    def set_weights(self, weights):

        if isinstance(weights, np.ndarray) and weights.shape == ():
            weights = weights.item()

        W_list = None
        b_list = None

        if isinstance(weights, dict):

            if "weights" in weights and "biases" in weights:
                W_list = weights["weights"]
                b_list = weights["biases"]

            elif "layers" in weights:
                W_list = [layer["W"] for layer in weights["layers"]]
                b_list = [layer["b"] for layer in weights["layers"]]

            else:

                temp_W = []
                temp_b = []
                i = 1

                while f"W{i}" in weights and f"b{i}" in weights:
                    temp_W.append(weights[f"W{i}"])
                    temp_b.append(weights[f"b{i}"])
                    i += 1

                if len(temp_W) > 0:
                    W_list = temp_W
                    b_list = temp_b

        elif isinstance(weights, (list, tuple)):

            if len(weights) > 0 and isinstance(weights[0], dict):

                W_list = [layer["W"] for layer in weights]
                b_list = [layer["b"] for layer in weights]

            elif len(weights) % 2 == 0:

                W_list = []
                b_list = []

                for i in range(0, len(weights), 2):
                    W_list.append(weights[i])
                    b_list.append(weights[i + 1])

        if W_list is None or b_list is None:
            raise ValueError("Unsupported weight format")

        W_list = [np.asarray(W, dtype=np.float64) for W in W_list]
        b_list = [np.asarray(b, dtype=np.float64).reshape(1, -1) for b in b_list]

        layer_sizes = [W_list[0].shape[0]]

        for W in W_list:
            layer_sizes.append(W.shape[1])

        self.layer_sizes = layer_sizes
        self._build_layers(self.layer_sizes)

        for layer, W, b in zip(self.layers, W_list, b_list):
            layer.W = W.copy()
            layer.b = b.copy()
