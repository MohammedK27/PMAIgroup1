import numpy as np

class CrossEntropyLoss:
    def __init__(self, name):
        self.name = name


    def forward(self, pred, actual):

        probs = self.softmax.forward(pred) 

        N = pred.shape[0]
        
        
        correct_probs = probs[range(N), actual]
        
        
        loss = -np.mean(np.log(correct_probs + 1e-15))

        
        grad = probs.copy()
        grad[range(N), actual] -= 1
        grad /= N  

        return loss, grad
