import wandb
import argparse
import json
import numpy as np

from keras.datasets import mnist, fashion_mnist
from ann.neural_network import NeuralNetwork
from ann.objective_functions import one_hot_encode
from utils.data_loader import load_data


def parse_args():
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
    parser.add_argument("-w_i", "--weight_init", choices=["random", "xavier", "zeros"], required=True)
    parser.add_argument("--log_samples", action="store_true",
                        help="Log 5 sample images from each class to W&B")
    return parser.parse_args()


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def log_sample_images_to_wandb(dataset_name):
    if dataset_name == "mnist":
        (images, labels), _ = mnist.load_data()
        class_names = [str(i) for i in range(10)]
    else:
        (images, labels), _ = fashion_mnist.load_data()
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

    table = wandb.Table(columns=["class_id", "class_name", "image_index", "image"])
    counts = {i: 0 for i in range(10)}

    for idx in range(len(images)):
        label = int(labels[idx])
        if counts[label] < 5:
            table.add_data(
                label,
                class_names[label],
                idx,
                wandb.Image(images[idx], caption=f"class={class_names[label]}")
            )
            counts[label] += 1

        if all(counts[i] == 5 for i in range(10)):
            break

    print("Logging sample_images_table to W&B with 50 images")
    wandb.log({"sample_images_table": table})


def collect_activation_stats(model, x_sample):
    stats = {}
    current = x_sample

    # hidden layers only
    for i, layer in enumerate(model.layers[:-1]):
        current = layer.forward(current)

        zero_fraction = float(np.mean(current == 0.0))
        mean_activation = float(np.mean(current))
        std_activation = float(np.std(current))

        stats[f"layer_{i+1}_zero_fraction"] = zero_fraction
        stats[f"layer_{i+1}_mean_activation"] = mean_activation
        stats[f"layer_{i+1}_std_activation"] = std_activation

    return stats


def main():
    args = parse_args()

    # Simple clean W&B init
    wandb.init(
        project="da6401-mlp",
        config=vars(args)
    )

    if args.log_samples:
        log_sample_images_to_wandb(args.dataset)

    if args.num_layers != len(args.hidden_size):
        raise ValueError("num_layers must match the count of hidden_size values")

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

    # fixed sample for activation monitoring
    x_monitor = x_val[:512]
    global_iteration = 0

    for epoch in range(args.epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train = x_train[indices]
        y_train = y_train[indices]
        y_train_oh = y_train_oh[indices]

        first_layer_grad_norm_sum = 0.0
        batch_count = 0

        for start in range(0, x_train.shape[0], args.batch_size):
            end = start + args.batch_size
            xb = x_train[start:end]
            yb = y_train_oh[start:end]

            logits = model.forward(xb)
            model.backward(yb, logits)

            # Gradient norm of first hidden layer
            first_layer_grad_norm = np.linalg.norm(model.layers[0].grad_W)
            first_layer_grad_norm_sum += first_layer_grad_norm
            batch_count += 1

            # Log gradients of 5 neurons for first 50 iterations (needed for Q2.9)
            if global_iteration < 50:
                grad_log = {
                    "iteration": global_iteration + 1,
                    "grad_neuron_1": float(model.layers[0].grad_b[0, 0]),
                    "grad_neuron_2": float(model.layers[0].grad_b[0, 1]),
                    "grad_neuron_3": float(model.layers[0].grad_b[0, 2]),
                    "grad_neuron_4": float(model.layers[0].grad_b[0, 3]),
                    "grad_neuron_5": float(model.layers[0].grad_b[0, 4]),
                }
                wandb.log(grad_log)

            global_iteration += 1
            model.update_weights()

        avg_first_layer_grad_norm = first_layer_grad_norm_sum / batch_count

        val_logits = model.forward(x_val)
        val_loss = model.compute_loss(y_val_oh, val_logits)
        val_pred, _ = model.predict(x_val)
        val_acc = accuracy_score(y_val, val_pred)

        activation_stats = collect_activation_stats(model, x_monitor)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc={val_acc:.6f} | "
            f"first_layer_grad_norm={avg_first_layer_grad_norm:.6f}"
        )

        log_data = {
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "first_layer_grad_norm": avg_first_layer_grad_norm
        }
        log_data.update(activation_stats)

        wandb.log(log_data)

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
        "log_samples": args.log_samples
    }

    with open("src/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    test_pred, _ = model.predict(x_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Final test accuracy: {test_acc:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()