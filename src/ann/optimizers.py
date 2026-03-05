import numpy as np

class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers):
        """
        Iterates through network layers and updates their weights.

        Args: 
            layers: List containing all the NeuralLayer objects.
        """
        raise NotImplementedError
    
class SGD(Optimizer):
    """Class for implementing Stochastic Gradient Descent Optimizer."""

    def update(self, layers):
        for layer in layers:
            grad_W_with_decay = layer.grad_W + (self.weight_decay * layer.W)
            layer.W = layer.W - (self.lr * grad_W_with_decay)
            layer.b = layer.b - (self.lr * layer.grad_b)

class Momentum(Optimizer):
    """Class for implementing Gradient Descent with Momentum Optimizer."""

    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocities = []    # Hold a dictionary of velocity matrices for each layer

    def update(self, layers):
        if(len(self.velocities) == 0):
            for layer in layers:
                self.velocities.append({
                    "v_W": np.zeros_like(layer.W),
                    "v_b": np.zeros_like(layer.b)
                })

        for i, layer in enumerate(layers):
            # Velocities Update Step
            grad_W_with_decay = layer.grad_W + (self.weight_decay * layer.W)
            self.velocities[i]["v_W"] = self.beta * self.velocities[i]["v_W"] + (1 - self.beta) * grad_W_with_decay
            self.velocities[i]["v_b"] = self.beta * self.velocities[i]["v_b"] + (1 - self.beta) * layer.grad_b          

            # Weight Update Step
            layer.W = layer.W - (self.lr * self.velocities[i]["v_W"])
            layer.b = layer.b - (self.lr * self.velocities[i]["v_b"])

class NAG(Optimizer):
    """Class for implementing Nesterov Accelerated Gradient Optimizer."""
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocities = []    # Hold a dictionary of velocity matrices for each layer

    def update(self, layers):
        if(len(self.velocities) == 0):
            for layer in layers:
                self.velocities.append({
                    "v_W": np.zeros_like(layer.W),
                    "v_b": np.zeros_like(layer.b)
                })

        for i, layer in enumerate(layers):
            # Velocities Update Step
            grad_W_with_decay = layer.grad_W + (self.weight_decay * layer.W)
            self.velocities[i]["v_W"] = self.beta * self.velocities[i]["v_W"] + (1 - self.beta) * grad_W_with_decay
            self.velocities[i]["v_b"] = self.beta * self.velocities[i]["v_b"] + (1 - self.beta) * layer.grad_b            

            # Weight Update Step
            layer.W = layer.W - self.lr * (self.beta * self.velocities[i]["v_W"] + (1 - self.beta) * grad_W_with_decay)
            layer.b = layer.b - self.lr * (self.beta * self.velocities[i]["v_b"] + (1 - self.beta) * layer.grad_b)

class RMSprop(Optimizer):
    """Class for implementing Root Mean Square Propogation Optimizer."""
    def __init__(self, learning_rate=0.01, beta=0.99, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.caches = []    # Hold a dictionary of Cache for each layer

    def update(self, layers):
        if(len(self.caches) == 0):
            for layer in layers:
                self.caches.append({
                    "c_W": np.zeros_like(layer.W),
                    "c_b": np.zeros_like(layer.b)
                })

        for i, layer in enumerate(layers):
            # Cache Update Step
            grad_W_with_decay = layer.grad_W + (self.weight_decay * layer.W)
            self.caches[i]["c_W"] = self.beta * self.caches[i]["c_W"] + (1 - self.beta) * (grad_W_with_decay ** 2)
            self.caches[i]["c_b"] = self.beta * self.caches[i]["c_b"] + (1 - self.beta) * (layer.grad_b ** 2)

            # Weight Update Step
            layer.W = layer.W - (self.lr / (np.sqrt(self.caches[i]["c_W"]) + self.epsilon)) * grad_W_with_decay
            layer.b = layer.b - (self.lr / (np.sqrt(self.caches[i]["c_b"]) + self.epsilon)) * layer.grad_b

class Adam(Optimizer):
    """Class for implementing Adaptive Moment Estimation Optimizer."""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.moments = []   # Hold a dictionary of Moments Values for each layer

    def update(self, layers):
        if(len(self.moments) == 0):
            for layer in layers:
                self.moments.append({
                    "m_W": np.zeros_like(layer.W),
                    "m_b": np.zeros_like(layer.b),
                    "v_W": np.zeros_like(layer.W),
                    "v_b": np.zeros_like(layer.b)
                })

        self.t += 1
        for i, layer in enumerate(layers):
            # Moment Update Step
            grad_W_with_decay = layer.grad_W + (self.weight_decay * layer.W)
            self.moments[i]["m_W"] = self.beta1 * self.moments[i]["m_W"] + (1 - self.beta1) * grad_W_with_decay
            m_W_hat = self.moments[i]["m_W"] / (1 - (self.beta1 ** self.t))
            self.moments[i]["v_W"] = self.beta2 * self.moments[i]["v_W"] + (1 - self.beta2) * (grad_W_with_decay ** 2)
            v_W_hat = self.moments[i]["v_W"] / (1 - (self.beta2 ** self.t))

            self.moments[i]["m_b"] = self.beta1 * self.moments[i]["m_b"] + (1 - self.beta1) * layer.grad_b
            m_b_hat = self.moments[i]["m_b"] / (1 - (self.beta1 ** self.t))
            self.moments[i]["v_b"] = self.beta2 * self.moments[i]["v_b"] + (1 - self.beta2) * (layer.grad_b ** 2)
            v_b_hat = self.moments[i]["v_b"] / (1 - (self.beta2 ** self.t))

            # Weight Update Step
            layer.W = layer.W - (self.lr * m_W_hat) / (np.sqrt(v_W_hat) + self.epsilon)
            layer.b = layer.b - (self.lr * m_b_hat) / (np.sqrt(v_b_hat) + self.epsilon)

class NAdam(Optimizer):
    """Class for implementing Nesterov-accelerated Adaptive Moment Estimation Optimizer."""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.moments = []   # Hold a dictionary of Moments Values for each layer

    def update(self, layers):
        if(len(self.moments) == 0):
            for layer in layers:
                self.moments.append({
                    "m_W": np.zeros_like(layer.W),
                    "m_b": np.zeros_like(layer.b),
                    "v_W": np.zeros_like(layer.W),
                    "v_b": np.zeros_like(layer.b)
                })

        self.t += 1
        for i, layer in enumerate(layers):
            # Moment Update Step
            grad_W_with_decay = layer.grad_W + (self.weight_decay * layer.W)
            self.moments[i]["m_W"] = self.beta1 * self.moments[i]["m_W"] + (1 - self.beta1) * grad_W_with_decay
            m_W_hat = self.moments[i]["m_W"] / (1 - (self.beta1 ** self.t))
            m_W_nesterov = (self.beta1 * m_W_hat) + ((1 - self.beta1) * grad_W_with_decay) / (1 - (self.beta1 ** self.t))
            self.moments[i]["v_W"] = self.beta2 * self.moments[i]["v_W"] + (1 - self.beta2) * (grad_W_with_decay ** 2)
            v_W_hat = self.moments[i]["v_W"] / (1 - (self.beta2 ** self.t))
            
            self.moments[i]["m_b"] = self.beta1 * self.moments[i]["m_b"] + (1 - self.beta1) * layer.grad_b
            m_b_hat = self.moments[i]["m_b"] / (1 - (self.beta1 ** self.t))
            m_b_nesterov = (self.beta1 * m_b_hat) + ((1 - self.beta1) * layer.grad_b) / (1 - (self.beta1 ** self.t))
            self.moments[i]["v_b"] = self.beta2 * self.moments[i]["v_b"] + (1 - self.beta2) * (layer.grad_b ** 2)
            v_b_hat = self.moments[i]["v_b"] / (1 - (self.beta2 ** self.t))

            # Weight Update Step
            layer.W = layer.W - (self.lr * m_W_nesterov) / (np.sqrt(v_W_hat) + self.epsilon)
            layer.b = layer.b - (self.lr * m_b_nesterov) / (np.sqrt(v_b_hat) + self.epsilon)