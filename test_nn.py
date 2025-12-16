import numpy as np

from layers.softmax import Softmax
from layers.linear import Linear
from layers.relu import relu
from layers.dropout import Dropout

from CrossEntropyLoss import CrossEntropyLoss
from NeuralNetwork import NeuralNetwork
from optimisers import SGD


def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]

        x[idx] = old + eps
        fx1 = f(x)

        x[idx] = old - eps
        fx2 = f(x)

        x[idx] = old
        grad[idx] = (fx1 - fx2) / (2 * eps)

        it.iternext()
    return grad


# softmax forward test
def test_softmax_forward():
    np.random.seed(0)
    sm = Softmax()

    z = np.random.randn(5, 3)
    probs = sm.forward(z)

    print("=== Softmax forward test ===")
    print("probs shape:", probs.shape)
    print("row sums (should be 1):", np.sum(probs, axis=1))
    print()


# softmax backward test 
def test_softmax_backward():
    np.random.seed(1)
    sm = Softmax()

    z = np.random.randn(4, 5)
    grad_out = np.random.randn(4, 5)

    sm.forward(z)
    grad_analytic = sm.backward(grad_out)

    def L(z_in):
        probs = sm.forward(z_in)
        return float(np.sum(probs * grad_out))

    grad_numeric = numerical_grad(L, z.copy(), eps=1e-5)

    max_diff = np.max(np.abs(grad_analytic - grad_numeric))

    print("=== Softmax backward gradient check ===")
    print("max abs diff (want ~1e-6 to 1e-4):", max_diff)
    print()


def test_linear_backward():
    np.random.seed(2)
    layer = Linear(input_dim=3, output_dim=4)

    x = np.random.randn(6, 3)
    d_out = np.random.randn(6, 4)


    out = layer.forward(x)
    _ = out 
    layer.backward(d_out)

    dW_analytic = layer.dW.copy()
    db_analytic = layer.db.copy()

    def L_W(W_in):
        layer.W = W_in
        out_local = layer.forward(x)
        return float(np.sum(out_local * d_out))

    def L_b(b_in):
        layer.b = b_in
        out_local = layer.forward(x)
        return float(np.sum(out_local * d_out))

    dW_numeric = numerical_grad(L_W, layer.W.copy(), eps=1e-5)
    db_numeric = numerical_grad(L_b, layer.b.copy(), eps=1e-5)

    max_diff_W = np.max(np.abs(dW_analytic - dW_numeric))
    max_diff_b = np.max(np.abs(db_analytic - db_numeric))

    print("=== Linear backward gradient check ===")
    print("max abs diff W:", max_diff_W)
    print("max abs diff b:", max_diff_b)
    print()



def test_dropout():
    np.random.seed(3)
    dr = Dropout(dropout_ratio=0.5)

    x = np.ones((4, 6), dtype=np.float32)

    dr.training = True
    out_train = dr.forward(x)

    dr.training = False
    out_test = dr.forward(x)

    print("=== Dropout behaviour test ===")
    print("training output (should contain zeros):")
    print(out_train)
    print("testing output (should be unchanged ones):")
    print(out_test)
    print()


def make_toy_3class(n_per_class=80, seed=0):
    """
    Simple 2D blobs -> 3 classes. Returns X (N,2), y (N,)
    """
    rng = np.random.default_rng(seed)

    centres = np.array([
        [-2.0, -2.0],
        [ 2.0,  0.0],
        [ 0.0,  2.5],
    ], dtype=np.float64)

    X_list, y_list = [], []
    for c in range(3):
        Xc = centres[c] + 0.6 * rng.standard_normal((n_per_class, 2))
        yc = np.full(n_per_class, c, dtype=np.int64)
        X_list.append(Xc)
        y_list.append(yc)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def test_training_toy():
    np.random.seed(0)

    X, y = make_toy_3class(n_per_class=60, seed=1)
    # train/test split
    n = X.shape[0]
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # Build a simple net:
    # Linear(2->16) -> ReLU -> Linear(16->3) -> Softmax
    layers = [
        Linear(2, 16),
        relu(),
        Linear(16, 3),
        Softmax()
    ]

    loss_fn = CrossEntropyLoss()
    optimiser = SGD(params=None, lr=0.1)  # params set inside train_batch in your code

    net = NeuralNetwork(layers=layers, loss_fn=loss_fn, optimiser=optimiser)

    print("=== End-to-end training test (toy 3-class) ===")
    history = net.fit(
        X_train, y_train,
        X_test, y_test,
        epochs=20,
        batch_size=32
    )

    print("Final train acc:", history["train_acc"][-1])
    print("Final test  acc:", history["test_acc"][-1])
    print("If this is learning, loss should drop and acc should rise.\n")


if __name__ == "__main__":
    test_softmax_forward()
    test_softmax_backward()
    test_linear_backward()
    test_dropout()
    test_training_toy()

