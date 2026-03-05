import numpy as np

class Activation:
    """Base class for all activation functions."""
    def __init__(self):
        self.cache = None   # Store Z or A values here during forward pass

    def forward(self, Z):
        """Compute the activation and store the necessary cache."""
        raise NotImplementedError
    
    def backward(self, dA):
        """
        Compute gradient of the loss w.r.t the input Z.
        dA is the gradient of the loss w.r.t the activation output A.
        Returns dZ.
        """
        raise NotImplementedError
    
class Sigmoid(Activation):
    """Class for Sigmoid activation function."""
    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, Z):
        self.cache = self.sig(Z)
        return self.cache

    def backward(self, dA):
        return dA * self.cache * (1 - self.cache)

class ReLU(Activation):
    """Class for ReLU activation function."""

    def forward(self, Z):
        self.cache = np.maximum(0, Z)
        return self.cache

    def backward(self, dA):
        return dA * (self.cache > 0).astype(int)

class Tanh(Activation):
    """Class for Tanh activation function."""

    def forward(self, Z):
        self.cache = np.tanh(Z)
        return self.cache

    def backward(self, dA):
        return dA * (1 - self.cache ** 2)

class Softmax(Activation):
    """Class for Softmax activation function."""

    def forward(self, Z):
        # Shifting the Z values by subtracting max value of Z from entire array so that 
        # if any Z value is really high, we don't overflow
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        self.cache = np.exp(Z_shifted) / np.sum(np.exp(Z_shifted), axis=1, keepdims=True)
        return self.cache
        

    def backward(self, dA):
        return (self.cache * dA - self.cache * np.sum(self.cache * dA, axis=1, keepdims=True))
    
class Linear(Activation):
    """Class for Linear Activation function."""

    def forward(self, Z):
        self.cache = Z
        return Z
    
    def backward(self, dA):
        return dA