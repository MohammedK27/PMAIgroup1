import numpy as np

class relu:
    #formula = relu(x) = max(0,x)
    def __init__(self):
        self.input = None

    def forward(self,x):
        self.input = x
        self.output = np.maximum(0,x)
        return self.output

    def backward(self,gout):
        #formula = derivative = f'(x)={1 | x > 0}
        #                             {0 | x <= 0}
        gin = gout * (self.input > 0)
        return gin
