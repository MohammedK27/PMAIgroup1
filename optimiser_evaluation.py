import numpy as np

import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from layers.linear import Linear
from layers.relu import relu
from layers.softmax import Softmax
from layers.dropout import Dropout
from CrossEntropyLoss import CrossEntropyLoss
from optimisers import SGD
from optimisers import SGD_momentum
from dataset import load_data


 
X_train, y_train, X_test, y_test = load_data("cifar-10-batches-py") 


def build_model(optimiser):
    layers = [
        Linear(3072, 512),
        relu(),
        Dropout(0.5),

        Linear(512, 256),
        relu(),

        Linear(256, 10),
        Softmax()
    ]

    return NeuralNetwork(
        layers=layers,
        loss_fn=CrossEntropyLoss(),
        optimiser=optimiser
    )


EPOCHS = 10
BATCH_SIZE = 64
LR = 0.01



print("\nTraining with SGD\n")

sgd = SGD(lr=LR)
model_sgd = build_model(sgd)

history_sgd = model_sgd.fit(
    X_train, y_train,
    X_test, y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)



print("\nTraining with SGD + Momentum\n")

sgd_m = SGD_momentum(lr=LR, beta=0.9)
model_m = build_model(sgd_m)

history_m = model_m.fit(
    X_train, y_train,
    X_test, y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# plotting the graph 

plt.figure()
plt.plot(history_sgd["train_loss"], label="SGD")
plt.plot(history_m["train_loss"], label="SGD + Momentum")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.show()

plt.figure()
plt.plot(history_sgd["test_acc"], label="SGD")
plt.plot(history_m["test_acc"], label="SGD + Momentum")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Comparison")
plt.legend()
plt.show()


print("\nFinal Test Accuracy:")
print("SGD:          ", history_sgd["test_acc"][-1])
print("SGD Momentum: ", history_m["test_acc"][-1])
