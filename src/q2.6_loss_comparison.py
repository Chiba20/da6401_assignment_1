import os
import sys
import numpy as np
import wandb

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from ann.objective_functions import one_hot_encode
from utils.data_loader import load_data


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def train_with_loss(loss_name):
    wandb.init(
        project="da6401-q26-loss-comparison",
        name=f"q26_{loss_name}",
        config={
            "dataset": "mnist",
            "epochs": 10,
            "batch_size": 64,
            "loss": loss_name,
            "optimizer": "rmsprop",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "num_layers": 3,
            "hidden_size": [128, 128, 128],
            "activation": "relu",
            "weight_init": "xavier"
        }
    )

    x_train, y_train, x_val, y_val, x_test, y_test = load_data("mnist")
    y_train_oh = one_hot_encode(y_train, 10)
    y_val_oh = one_hot_encode(y_val, 10)

    model = NeuralNetwork(
        layer_sizes=[784, 128, 128, 128, 10],
        activation="relu",
        loss_name=loss_name,
        optimizer_name="rmsprop",
        learning_rate=0.001,
        weight_decay=0.0,
        weight_init="xavier"
    )

    batch_size = 64
    epochs = 10

    for epoch in range(epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train = x_train[indices]
        y_train = y_train[indices]
        y_train_oh = y_train_oh[indices]

        batch_losses = []

        for start in range(0, x_train.shape[0], batch_size):
            end = start + batch_size
            xb = x_train[start:end]
            yb = y_train_oh[start:end]

            logits = model.forward(xb)
            batch_loss = model.compute_loss(yb, logits)
            batch_losses.append(batch_loss)

            model.backward(yb, logits)
            model.update_weights()

        train_pred, _ = model.predict(x_train)
        train_acc = accuracy_score(y_train, train_pred)

        val_logits = model.forward(x_val)
        val_loss = model.compute_loss(y_val_oh, val_logits)
        val_pred, _ = model.predict(x_val)
        val_acc = accuracy_score(y_val, val_pred)

        print(
            f"[{loss_name}] Epoch {epoch + 1}/{epochs} | "
            f"train_loss={np.mean(batch_losses):.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"train_acc={train_acc:.6f} | "
            f"val_acc={val_acc:.6f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(batch_losses)),
            "val_loss": float(val_loss),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "loss_type": loss_name
        })

    test_pred, _ = model.predict(x_test)
    test_acc = accuracy_score(y_test, test_pred)
    wandb.summary["final_test_accuracy"] = float(test_acc)
    print(f"[{loss_name}] Final test accuracy: {test_acc:.6f}")

    wandb.finish()


if __name__ == "__main__":
    train_with_loss("cross_entropy")
    train_with_loss("mean_squared_error")