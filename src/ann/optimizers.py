import numpy as np


class Optimizer:
    def __init__(self, name, learning_rate=0.001, weight_decay=0.0, beta=0.9, epsilon=1e-8, beta2=0.999):
        self.name = name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        self.velocity = {}
        self.cache = {}

    def _grad_with_decay(self, layer):
        grad_w = layer.grad_W + self.weight_decay * layer.W
        grad_b = layer.grad_b
        return grad_w, grad_b

    def update(self, layers):
        for idx, layer in enumerate(layers):
            grad_w, grad_b = self._grad_with_decay(layer)

            if self.name == "sgd":
                layer.W -= self.learning_rate * grad_w
                layer.b -= self.learning_rate * grad_b

            elif self.name == "momentum":
                if idx not in self.velocity:
                    self.velocity[idx] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

                self.velocity[idx]["W"] = self.beta * self.velocity[idx]["W"] - self.learning_rate * grad_w
                self.velocity[idx]["b"] = self.beta * self.velocity[idx]["b"] - self.learning_rate * grad_b

                layer.W += self.velocity[idx]["W"]
                layer.b += self.velocity[idx]["b"]

            elif self.name == "nag":
                if idx not in self.velocity:
                    self.velocity[idx] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

                prev_vw = self.velocity[idx]["W"].copy()
                prev_vb = self.velocity[idx]["b"].copy()

                self.velocity[idx]["W"] = self.beta * self.velocity[idx]["W"] - self.learning_rate * grad_w
                self.velocity[idx]["b"] = self.beta * self.velocity[idx]["b"] - self.learning_rate * grad_b

                layer.W += -self.beta * prev_vw + (1.0 + self.beta) * self.velocity[idx]["W"]
                layer.b += -self.beta * prev_vb + (1.0 + self.beta) * self.velocity[idx]["b"]

            elif self.name == "rmsprop":
                if idx not in self.cache:
                    self.cache[idx] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

                self.cache[idx]["W"] = self.beta * self.cache[idx]["W"] + (1.0 - self.beta) * (grad_w ** 2)
                self.cache[idx]["b"] = self.beta * self.cache[idx]["b"] + (1.0 - self.beta) * (grad_b ** 2)

                layer.W -= self.learning_rate * grad_w / (np.sqrt(self.cache[idx]["W"]) + self.epsilon)
                layer.b -= self.learning_rate * grad_b / (np.sqrt(self.cache[idx]["b"]) + self.epsilon)

            elif self.name == "adam":
                if idx not in self.velocity:
                    self.velocity[idx] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
                if idx not in self.cache:
                    self.cache[idx] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

                self.velocity[idx]["W"] = self.beta * self.velocity[idx]["W"] + (1.0 - self.beta) * grad_w
                self.velocity[idx]["b"] = self.beta * self.velocity[idx]["b"] + (1.0 - self.beta) * grad_b

                self.cache[idx]["W"] = self.beta2 * self.cache[idx]["W"] + (1.0 - self.beta2) * (grad_w ** 2)
                self.cache[idx]["b"] = self.beta2 * self.cache[idx]["b"] + (1.0 - self.beta2) * (grad_b ** 2)

                layer.W -= self.learning_rate * self.velocity[idx]["W"] / (np.sqrt(self.cache[idx]["W"]) + self.epsilon)
                layer.b -= self.learning_rate * self.velocity[idx]["b"] / (np.sqrt(self.cache[idx]["b"]) + self.epsilon)

            elif self.name == "nadam":
                if idx not in self.velocity:
                    self.velocity[idx] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
                if idx not in self.cache:
                    self.cache[idx] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

                self.velocity[idx]["W"] = self.beta * self.velocity[idx]["W"] + (1.0 - self.beta) * grad_w
                self.velocity[idx]["b"] = self.beta * self.velocity[idx]["b"] + (1.0 - self.beta) * grad_b

                self.cache[idx]["W"] = self.beta2 * self.cache[idx]["W"] + (1.0 - self.beta2) * (grad_w ** 2)
                self.cache[idx]["b"] = self.beta2 * self.cache[idx]["b"] + (1.0 - self.beta2) * (grad_b ** 2)

                nadam_vw = self.beta * self.velocity[idx]["W"] + (1.0 - self.beta) * grad_w
                nadam_vb = self.beta * self.velocity[idx]["b"] + (1.0 - self.beta) * grad_b

                layer.W -= self.learning_rate * nadam_vw / (np.sqrt(self.cache[idx]["W"]) + self.epsilon)
                layer.b -= self.learning_rate * nadam_vb / (np.sqrt(self.cache[idx]["b"]) + self.epsilon)

            else:
                raise ValueError(f"Unsupported optimizer: {self.name}")
