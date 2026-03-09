import os
import wandb
import argparse
import json
import numpy as np

from ann.neural_network import NeuralNetwork
from ann.objective_functions import one_hot_encode
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion"], required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], required=True)
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop"], required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, required=True)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, required=True)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, required=True)
    parser.add_argument("-a", "--activation", choices=["sigmoid", "tanh", "relu"], required=True)
    parser.add_argument("-w_i", "--weight_init", choices=["random", "xavier"], required=True)
    return parser.parse_args()


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def main():
    args = parse_arguments()

    if args.num_layers != len(args.hidden_size):
        raise ValueError("num_layers must match the count of hidden_size values")

    # Safe for local use and autograder
    wandb_mode = "online" if os.path.exists(os.path.expanduser("~/.netrc")) else "disabled"
    wandb.init(
        project="da6401-mlp",
        config=vars(args),
        mode=wandb_mode
    )

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)

    y_train_oh = one_hot_encode(y_train, 10)
    y_val_oh = one_hot_encode(y_val, 10)

    layer_sizes = [x_train.shape[1]] + args.hidden_size + [10]

    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation=args.activation,
        loss_name=args.loss,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        weight_init=args.weight_init,
    )

    best_val_acc = -1.0
    best_weights = None

    for epoch in range(args.epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train_epoch = x_train[indices]
        y_train_epoch = y_train[indices]
        y_train_oh_epoch = y_train_oh[indices]

        batch_losses = []

        for start in range(0, x_train_epoch.shape[0], args.batch_size):
            end = start + args.batch_size

            xb = x_train_epoch[start:end]
            yb = y_train_oh_epoch[start:end]

            logits = model.forward(xb)
            loss = model.compute_loss(yb, logits)
            batch_losses.append(loss)

            model.backward(yb, logits)
            model.update_weights()

        train_pred, _ = model.predict(x_train)
        train_acc = accuracy_score(y_train, train_pred)

        val_logits = model.forward(x_val)
        val_loss = model.compute_loss(y_val_oh, val_logits)
        val_pred, _ = model.predict(x_val)
        val_acc = accuracy_score(y_val, val_pred)

        test_pred, _ = model.predict(x_test)
        test_acc = accuracy_score(y_test, test_pred)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_acc={train_acc:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc={val_acc:.6f} | "
            f"test_acc={test_acc:.6f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(batch_losses)),
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

    if best_weights is None:
        best_weights = model.get_weights()

    np.save("src/best_model.npy", best_weights, allow_pickle=True)

    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "weight_init": args.weight_init,
    }

    with open("src/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    final_test_pred, _ = model.predict(x_test)
    final_test_acc = accuracy_score(y_test, final_test_pred)
    print(f"Final test accuracy: {final_test_acc:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()