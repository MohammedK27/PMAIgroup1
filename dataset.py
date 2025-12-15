import pickle
import numpy as np
import os


def load_cifar_batch(path):
    with open(path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data']
    y = np.array(batch[b'labels'])
    return X, y


def load_cifar10(root):
    X_train, y_train = [], []

    for i in range(1, 6):
        X, y = load_cifar_batch(os.path.join(root, f"data_batch_{i}"))
        X_train.append(X)
        y_train.append(y)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = load_cifar_batch(os.path.join(root, "test_batch"))

    return X_train, y_train, X_test, y_test


def load_data(root):
    X_train, y_train, X_test, y_test = load_cifar10(root)

    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test
