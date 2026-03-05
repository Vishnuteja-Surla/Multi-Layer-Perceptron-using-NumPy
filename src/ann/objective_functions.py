import numpy as np

class Loss:
    """Base class for all loss functions."""
    
    def forward(self, y_true, logits):
        """Calculates the scalar loss value."""
        raise NotImplementedError
    
    def backward(self, y_true, logits):
        """Calculate the gradient of loss w.r.t the predictions (dA)."""
        raise NotImplementedError
    
class MSE(Loss):
    """Class for Mean-Squared Error implementation."""
    
    def forward(self, y_true, logits):
        b = y_true.shape[0]    # Batch Size
        return 1/b * (np.sum((y_true - logits)**2))

    def backward(self, y_true, logits):
        b = y_true.shape[0]    # Batch Size
        return 2/b * (logits - y_true)


class CrossEntropy(Loss):
    """Class for Cross Entropy Loss implementation."""
    
    def forward(self, y_true, logits):
        b = y_true.shape[0]  # Batch Size

        # Applying softmax internally
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=1, keepdims=True)

        return -1/b * np.sum(y_true * np.log(np.maximum(probs, 1e-15)))   # Lower Bounding probability to avoid log(0) 

    def backward(self, y_true, logits):
        b = y_true.shape[0] # Batch Size

        # Applying softmax internally
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=1, keepdims=True)

        return (probs - y_true) / b