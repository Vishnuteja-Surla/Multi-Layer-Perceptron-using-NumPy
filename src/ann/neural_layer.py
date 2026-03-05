import numpy as np

class NeuralLayer:
    """The Blue print of a Neural Layer in a Multi-Layer Perceptron."""

    def __init__(self, input_size, output_size, activation_func, weight_init_method="random"):
        """
        Initialize the weights and biases based on the chosen method and set up the layer.
        """
        # Base information
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation_func

        # Weights and bias initialization
        self.W = None
        self.b = None
        self._initialize_weights(weight_init_method)

        # Gradient placeholders
        self.grad_W = None
        self.grad_b = None

        # Cache for Backpropogation
        self.A_prev = None
        self.Z = None

    def _initialize_weights(self, method):
        """
        Setup the weights and bias in either Random method or Xavier method or Zero method based on choice.
        Weight Matrix: (input_size, output_size)
        Bias: (1, output_size)
        """
        if method == "random":
            self.W = np.random.randn(self.input_size, self.output_size) * 0.01  # Multiplying with a constant so as to not make the weights too large or too small
        elif method == "xavier":
            self.W = np.random.randn(self.input_size, self.output_size) * np.sqrt(2.0 / (self.input_size + self.output_size))
        elif method == "zero":
            self.W = np.zeros((self.input_size, self.output_size))  # For Question 2.9
        else:
            print(f"{method} is not a valid weight initialization strategy. Initializing using Random method...\n")
            self.W = np.random.randn(self.input_size, self.output_size) * 0.01

        self.b = np.zeros((1, self.output_size))

    def forward(self, A_prev):
        """
        The forward pass computation at the current Neural Network Layer.
        A_prev: (batch_size, input_size).
        Z: (batch_size, output_size).
        """
        self.A_prev = A_prev
        self.Z = np.dot(self.A_prev, self.W) + self.b
        return self.activation.forward(self.Z)

    def backward(self, dA):
        """
        The backpropogation pass the current Neural Network Layer.
        dA: (batch_size, output_size).
        dZ: (batch_size, output_size).
        grad_W: (input_size, output_size).
        """
        dZ = self.activation.backward(dA)
        self.grad_W = np.dot(self.A_prev.T, dZ)
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)
        return np.dot(dZ, self.W.T)