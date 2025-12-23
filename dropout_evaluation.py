import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
from layers.linear import Linear
from layers.relu import relu
from layers.softmax import Softmax
from layers.dropout import Dropout
from CrossEntropyLoss import CrossEntropyLoss
from optimisers import SGD_momentum
from dataset import load_data



X_train, y_train, X_test, y_test = load_data("cifar-10-batches-py")



def build_model(optimiser, use_dropout=False):
    layers = [
        Linear(3072, 512),
        relu()
    ]

    if use_dropout:
        layers.append(Dropout(0.5))

    layers += [
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



print("\nTraining WITHOUT dropout\n")

opt_no = SGD_momentum(lr=LR, beta=0.9)
model_no = build_model(opt_no, use_dropout=False)

history_no = model_no.fit(
    X_train, y_train,
    X_test, y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)



print("\nTraining WITH dropout (0.5)\n")

opt_do = SGD_momentum(lr=LR, beta=0.9)
model_do = build_model(opt_do, use_dropout=True)

history_do = model_do.fit(
    X_train, y_train,
    X_test, y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)



plt.figure()
plt.plot(history_no["train_loss"], label="No Dropout")
plt.plot(history_do["train_loss"], label="Dropout (0.5)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Effect of Dropout on Training Loss")
plt.legend()
plt.show()

# Test Accuracy
plt.figure()
plt.plot(history_no["test_acc"], label="No Dropout")
plt.plot(history_do["test_acc"], label="Dropout (0.5)")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Effect of Dropout on Test Accuracy")
plt.legend()
plt.show()



print("\nFinal Test Accuracy:")
print("No Dropout: ", history_no["test_acc"][-1])
print("Dropout:    ", history_do["test_acc"][-1])
