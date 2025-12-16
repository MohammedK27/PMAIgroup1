import numpy as np
from dataset import load_data

from NeuralNetwork import NeuralNetwork
from CrossEntropyLoss import CrossEntropyLoss
from optimisers import SGD, SGD_momentum

from layers.linear import Linear
from layers.relu import relu
from layers.sigmoid import sigmoid
from layers.dropout import Dropout
from layers.softmax import Softmax

# Load CIFAR-10
cifar_root = r"C:\Users\User\Documents\Uni\Year_3\IN3063 Programming and Mathematics for Artificial Intelligence\PMAIgroup1\cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_data(cifar_root)

# Flatten images: (N, 32, 32, 3) -> (N, 3072)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# normalise to [0, 1]
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0


y_train = y_train.flatten()
y_test = y_test.flatten()


input_dim = X_train.shape[1]
num_classes = 10

def run_model(name, layers, optimiser, epochs=20, batch_size=64):
    loss_fn = CrossEntropyLoss()
    net = NeuralNetwork(layers, loss_fn, optimiser)

    print("\n==============================")
    print("Running:", name)
    print("==============================")

    history = net.fit(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

    final_train = history["train_acc"][-1]
    final_test = history["test_acc"][-1]
    final_loss = history["train_loss"][-1]

    print(f"{name} final | loss={final_loss:.4f} train_acc={final_train:.3f} test_acc={final_test:.3f}")
    return history

# Original
layers0 = [
    Linear(input_dim, 128),
    relu(),
    Linear(128, num_classes),
    Softmax()
]
opt0 = SGD(lr=0.01)
run_model("Baseline_128_ReLU", layers0, opt0)

# variation 1
layers1 = [
    Linear(input_dim, 256),
    relu(),
    Linear(256, num_classes),
    Softmax()
]
opt1 = SGD(lr=0.01)
run_model("Wider_256_ReLU", layers1, opt1)

# variant 2, Dropout
layers2 = [
    Linear(input_dim, 256),
    relu(),
    Dropout(0.5),
    Linear(256, num_classes),
    Softmax()
]
opt2 = SGD(lr=0.01)
run_model("Wider_256_ReLU_Dropout0.5", layers2, opt2)


