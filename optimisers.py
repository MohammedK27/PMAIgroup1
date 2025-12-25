import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        #learning rate controlling the size of parameter updates
        self.params = None
        self.lr = lr

    def step(self, grads):
        #update parameter by moving opposite to the gradient
        for i, loss_grad in enumerate(grads):
            self.params[i] -= self.lr * loss_grad
            
    # reset gradients to zero after an update step        
    def zero_grad(self, grads):
        for i in range(len(grads)):
            grads[i].fill(0.0)



class SGD_momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.params = None
        self.lr = lr
        #momentum factor (how much past gradients influence the update, default val of 9)
        self.beta = beta
        self.momentums = None
        
    #initialise momentum terms the first time step() is called
    def step(self, grads):
        if self.momentums is None:
            self.momentums = [np.zeros_like(g) for g in grads]

        #update momentum using an exponential moving average
        for i, loss_grad in enumerate(grads):
            self.momentums[i] = self.beta * self.momentums[i] + (1 - self.beta) * loss_grad
            self.params[i] -= self.lr * self.momentums[i]

    # reset gradients to zero after an update step
    def zero_grad(self, grads):
        for i in range(len(grads)):
            grads[i].fill(0.0)
