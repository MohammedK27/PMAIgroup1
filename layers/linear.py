import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.dW = None
        self.db = None
        self.input = None

    def forward(self, x):
        self.input = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, d_out):
        self.dW = np.dot(self.input.T, d_out)
        self.db = np.sum(d_out, axis=0, keepdims=True)
        d_input = np.dot(d_out, self.W.T)
        return d_input
