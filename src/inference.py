import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="src/best_model.npy")
    parser.add_argument("--config_path", default="src/config.json")
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion"], default="mnist")
    return parser.parse_args()


def save_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    classes = np.arange(cm.shape[0])
    plt.xticks(classes)
    plt.yticks(classes)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_misclassified_examples(x_test, y_test, predictions, save_path, max_images=25):
    misclassified_idx = np.where(y_test != predictions)[0]
    num_images = min(len(misclassified_idx), max_images)

    if num_images == 0:
        return

    cols = 5
    rows = int(np.ceil(num_images / cols))

    plt.figure(figsize=(12, 10))
    for plot_idx in range(num_images):
        idx = misclassified_idx[plot_idx]
        image = x_test[idx].reshape(28, 28)

        plt.subplot(rows, cols, plot_idx + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"T:{y_test[idx]} P:{predictions[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = parse_arguments()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    _, _, _, _, x_test, y_test = load_data(args.dataset)

    layer_sizes = [x_test.shape[1]] + config["hidden_size"] + [10]

    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation=config["activation"],
        loss_name=config["loss"],
        optimizer_name=config["optimizer"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        weight_init=config["weight_init"],
    )

    weights = np.load(args.model_path, allow_pickle=True)
    if isinstance(weights, np.ndarray) and weights.shape == ():
        weights = weights.item()

    model.set_weights(weights)

    predictions, _ = model.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average="macro", zero_division=0)
    rec = recall_score(y_test, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

    print(f"Accuracy: {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall: {rec:.6f}")
    print(f"F1-score: {f1:.6f}")

    cm = confusion_matrix(y_test, predictions)
    save_confusion_matrix(cm, "src/confusion_matrix.png")
    save_misclassified_examples(x_test, y_test, predictions, "src/misclassified_examples.png")

    print("Saved: src/confusion_matrix.png")
    print("Saved: src/misclassified_examples.png")


if __name__ == "__main__":
    main()