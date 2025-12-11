import numpy as np

class sigmoid:
    #formula = f(x) = 1/1(1+e^(-x))
    def __init__(self):
        self.output = None

    def forward(self,x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, gout):
        #formula = derivative = sigmoid(x)(1-sigmoid(x))
        self.output = gout * (1 - self.output)
        return self.output
