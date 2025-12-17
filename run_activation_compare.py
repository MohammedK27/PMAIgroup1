import numpy as np

from dataset import load_data

from NeuralNetwork import NeuralNetwork
from CrossEntropyLoss import CrossEntropyLoss
from optimisers import SGD

from layers.linear import Linear
from layers.relu import relu
from layers.sigmoid import sigmoid
from layers.softmax import Softmax
import matplotlib.pyplot as plt

cifar_root = r"C:\Users\User\Documents\Uni\Year_3\IN3063 Programming and Mathematics for Artificial Intelligence\PMAIgroup1\cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_data(cifar_root)

#temporary
print(X_train.shape)
print(y_train.shape)


input_dim = X_train.shape[1] 
num_classes = 10


def run_model(name, layers, lr=0.01, epochs=20, batch_size=64):
    net = NeuralNetwork(layers, CrossEntropyLoss(), SGD(lr))
    print("\n==============================")
    print("Running:", name)
    print("==============================")
    hist = net.fit(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

    print(f"{name} FINAL | loss={hist['train_loss'][-1]:.4f} "
          f"train_acc={hist['train_acc'][-1]:.3f} test_acc={hist['test_acc'][-1]:.3f}")
    return hist


# 2) BASE: ReLU
layers_relu = [
    Linear(input_dim, 128),
    relu(),
    Linear(128, num_classes),
    Softmax()
]
hist_relu = run_model("ActivationCompare_ReLU", layers_relu)


# 3) COMPARISON: change to sigmoid now and see what happens
layers_sigmoid = [
    Linear(input_dim, 128),
    sigmoid(),
    Linear(128, num_classes),
    Softmax()
]
hist_sigmoid = run_model("ActivationCompare_Sigmoid", layers_sigmoid)



## PLOTTING GRAPHS FOR OUR RESULTS


epochs = range(1, len(hist_relu["train_loss"]) + 1)

# ---- Loss plot ----
plt.figure()
plt.plot(epochs, hist_relu["train_loss"], label="ReLU")
plt.plot(epochs, hist_sigmoid["train_loss"], label="Sigmoid")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
plt.show()

# ---- training accuracy plot ----
plt.figure()
plt.plot(epochs, hist_relu["train_acc"], label="ReLU")
plt.plot(epochs, hist_sigmoid["train_acc"], label="Sigmoid")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy vs Epoch")
plt.legend()
plt.show()

# ---- Test accuracy plot ----
plt.figure()
plt.plot(epochs, hist_relu["test_acc"], label="ReLU")
plt.plot(epochs, hist_sigmoid["test_acc"], label="Sigmoid")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Epoch")
plt.legend()
plt.show()
