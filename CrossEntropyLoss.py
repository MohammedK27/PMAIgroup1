import numpy as np

class CrossEntropyLoss:
    def __init__(self, name="cross_entropy"):
        self.name = name

    def forward(self, pred, actual):

        N = pred.shape[0]

    
        correct_logprobs = -np.log(pred[np.arange(N), actual] + 1e-15)
        loss = np.mean(correct_logprobs)

        grad = pred.copy()
        grad[np.arange(N), actual] -= 1
        grad /= N

        return loss, grad
