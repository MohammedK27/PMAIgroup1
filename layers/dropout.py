import numpy as np

class Dropout:
    def __init__(self, dropout_ratio: float = 0.5):
        # Ensure dropout_ratio is valid (0 = no dropout, 1 = drop everything)
        assert 0 <= dropout_ratio <= 1

        # Probability of dropping a unit during training
        self.dropout_ratio = dropout_ratio

        # Probability of keeping a unit (used for inverted dropout scaling)
        self.keep_probability = 1 - dropout_ratio

        # Mask sampled during the forward pass (stored so backward uses the SAME mask)
        self.mask = None

        # Flag to control behaviour:
        # True: training mode (apply dropout)
        # False: evaluation mode (no dropout)
        self.training = True
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # If we are evaluating OR dropout is disabled, return input unchanged
        if not self.training or self.dropout_ratio == 0:
            return x
        
        # Sample dropout mask:
        # np.random.rand(*x.shape) generates uniform random values in [0,1)
        # values < keep_probability become True (keep unit), else False (drop unit)
        # astype(np.float32) converts boolean mask to 0.0/1.0
        # Inverted dropout: divide by keep_probability so expected activation stays the same.
        self.mask = ((np.random.rand(*x.shape) < self.keep_probability).astype(np.float32)/ self.keep_probability)

        # Apply mask: dropped units become 0, kept units are scaled up (inverted dropout)
        return x * self.mask
    
    def backward(self, gradient_out: np.ndarray) -> np.ndarray:
        # If we are evaluating or dropout is disabled, gradients pass through unchanged
        if not self.training or self.dropout_ratio == 0:
            return gradient_out
        
        # Apply the same mask to gradients:
        # dropped units receive zero gradient
        # kept units receive scaled gradient (consistent with forward scaling)
        gradient_in = gradient_out * self.mask
        return gradient_in
