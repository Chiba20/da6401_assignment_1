import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def load_data(dataset_name="mnist", validation_size=0.1, random_state=42):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("dataset must be either 'mnist' or 'fashion'")

    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    return x_train, y_train, x_val, y_val, x_test, y_test
