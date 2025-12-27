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

import matplotlib.pyplot as plt


# Load CIFAR-10
# load training and test data
cifar_root = r"C:\Users\User\Documents\Uni\Year_3\IN3063 Programming and Mathematics for Artificial Intelligence\PMAIgroup1\cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_data(cifar_root)

# Flatten images: (N, 32, 32, 3) into (N, 3072)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


#make sure they are 1D arrays
y_train = y_train.flatten()
y_test = y_test.flatten()


input_dim = X_train.shape[1]
num_classes = 10


#helper funtcion to run model
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
hist_base = run_model("Baseline_128_ReLU", layers0, opt0)

# variation 1
layers1 = [
    Linear(input_dim, 256),
    relu(),
    Linear(256, num_classes),
    Softmax()
]
opt1 = SGD(lr=0.01)
hist_wide = run_model("Wider_256_ReLU", layers1, opt1)

# variant 2, Dropout
layers2 = [
    Linear(input_dim, 256),
    relu(),
    Dropout(0.5),
    Linear(256, num_classes),
    Softmax()
]
opt2 = SGD(lr=0.01)
hist_dropout = run_model("Wider_256_ReLU_Dropout0.5", layers2, opt2)


epochs = range(1, len(hist_base["train_loss"]) + 1)

##### PLOTTING GRAPHS ###


# Training Loss vs Epoch 

plt.figure()
plt.plot(epochs, hist_base["train_loss"], label="128 ReLU")
plt.plot(epochs, hist_wide["train_loss"], label="256 ReLU")
plt.plot(epochs, hist_dropout["train_loss"], label="256 ReLU + Dropout")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
plt.savefig("capacity_loss.png", dpi=200)
plt.close()



#Training accuracy vs Epoch
plt.figure()
plt.plot(epochs, hist_base["train_acc"], label="128 ReLU")
plt.plot(epochs, hist_wide["train_acc"], label="256 ReLU")
plt.plot(epochs, hist_dropout["train_acc"], label="256 ReLU + Dropout")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy vs Epoch")
plt.legend()
plt.savefig("capacity_train_acc.png", dpi=200)
plt.close()

#Test Accuracy vs Epoch

plt.figure()
plt.plot(epochs, hist_base["test_acc"], label="128 ReLU")
plt.plot(epochs, hist_wide["test_acc"], label="256 ReLU")
plt.plot(epochs, hist_dropout["test_acc"], label="256 ReLU + Dropout")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Epoch")
plt.legend()
plt.savefig("capacity_test_acc.png", dpi=200)
plt.close()





