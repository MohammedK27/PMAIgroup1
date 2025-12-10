import numpy as np

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self, grads):

        for i, loss_grad in enumerate(grads):
            self.params[i] -= self.lr * loss_grad
            
    def zero_grad(self, grads):
        for i in range(len(grads)):
            grads[i].fill(0.0)



class SGD_momentum:
    def __init__(self, params, lr=0.01, beta=0.9):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.momentums = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, loss_grad in enumerate(grads):
            self.momentums[i] = self.beta * self.momentums[i] + (1 - self.beta) * loss_grad
            self.params[i] -= self.lr * self.momentums[i]

    def zero_grad(self, grads):
        for i in range(len(grads)):
            grads[i].fill(0.0)
