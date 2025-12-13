import numpy as np

class Softmax:
    def __init__(self):
        self.out = None  

    def forward(self, z):
        
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        sums = np.sum(exp_z, axis=1, keepdims=True)
        self.out = exp_z / sums      
        return self.out
    

    def backward(self, grad_output):
        
        """
        gradient_output: dL/ds, same shape as self.out (N, C)
        returns: dL/dz (N, C)
        """
        N, C = grad_output.shape
        grad_input = np.zeros_like(grad_output)

        
        for n in range(N):
            s = self.out[n]        
            g = grad_output[n]     

            # scalar: sum_j s_j * g_j
            s_dot_g = np.sum(s * g)

            # apply formula: dL/dz_i = s_i * (g_i - sum_j s_j * g_j)
            grad_input[n] = s * (g - s_dot_g)

        return grad_input
