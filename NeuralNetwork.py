import numpy as np
from layers.dropout import Dropout


class NeuralNetwork:
    def __init__(self, layers, loss_fn, optimiser):
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimiser = optimiser

  
    def forward(self, x, training=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training
            x = layer.forward(x)
        return x


    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

    def get_params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            if hasattr(layer, "W"):
                params += [layer.W, layer.b]
                grads += [layer.dW, layer.db]
        return params, grads

    def train_batch(self, x, y):
        probs = self.forward(x, training=True)
        loss, d_out = self.loss_fn.forward(probs, y)
        self.backward(d_out)

        params, grads = self.get_params_and_grads()

        # If the optimiser does not yet have parameters, give them now
        if getattr(self.optimiser, "params", None) is None:
            self.optimiser.params = params

        self.optimiser.step(grads)

        return loss


    def predict(self, x):
        probs = self.forward(x, training=False)
        return np.argmax(probs, axis=1)

    def accuracy(self, x, y):
        preds = self.predict(x)
        return np.mean(preds == y)

    def fit(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):
        n = X_train.shape[0]
        history = {"train_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_train, y_train = X_train[idx], y_train[idx]

            losses = []
            for i in range(0, n, batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]
                loss = self.train_batch(xb, yb)
                losses.append(loss)

            train_acc = self.accuracy(X_train, y_train)
            test_acc = self.accuracy(X_test, y_test)

            history["train_loss"].append(np.mean(losses))
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

        return history
