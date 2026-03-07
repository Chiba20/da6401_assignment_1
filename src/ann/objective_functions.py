import numpy as np
from ann.activations import softmax


def one_hot_encode(y, num_classes=10):
    y = np.asarray(y).astype(int)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def cross_entropy_loss(y_true, logits):
    probs = softmax(logits)
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(probs + eps), axis=1))


def cross_entropy_gradient(y_true, logits):
    probs = softmax(logits)
    batch_size = y_true.shape[0]
    return (probs - y_true) / batch_size


def mse_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def mse_gradient(y_true, y_pred):
    batch_size = y_true.shape[0]
    return 2.0 * (y_pred - y_true) / batch_size
