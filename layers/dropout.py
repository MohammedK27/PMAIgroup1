import numpy as np

class Dropout:
    def __init__(self, dropout_ratio: float = 0.5):
        assert 0 <= dropout_ratio <= 1
        self.dropout_ratio = dropout_ratio
        self.keep_probability = 1 - dropout_ratio
        self.mask = None
        self.training = True
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.dropout_ratio == 0:
            return x
        
        self.mask = (np.random.rand(*x.shape) < self.keep_probability).astype(np.float32) / self.keep_probability
        return x * self.mask
    
    def backward(self, gradient_out: np.ndarray) -> np.ndarray:
        if not self.training or self.dropout_ratio == 0:
            return gradient_out
        
        gradient_in = gradient_out * self.mask
        return gradient_in
