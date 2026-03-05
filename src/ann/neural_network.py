import numpy as np
import copy
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from ann.activations import ReLU, Sigmoid, Tanh, Softmax, Linear
from ann.neural_layer import NeuralLayer
from ann.optimizers import SGD, Momentum, NAG, RMSprop, Adam, NAdam
from ann.objective_functions import MSE, CrossEntropy

ACTIVATIONS = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax,
    'linear': Linear
}

OPTIMIZERS = {
    'sgd': SGD,
    'momentum': Momentum,
    'nag': NAG,
    'rmsprop': RMSprop,
    'adam': Adam,
    'nadam': NAdam
}

LOSS_FUNCTIONS = {
    'mean_squared_error': MSE,
    'cross_entropy': CrossEntropy
}

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """

        # Reading the CLI arguments
        self.loss_fn = LOSS_FUNCTIONS[cli_args.loss]()
        self.lr = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay
        
        if hasattr(cli_args, "num_layers"):
            self.num_layers = cli_args.num_layers
        else:
            self.num_layers = cli_args.num_hidden_layers

        if hasattr(cli_args, "hidden_size"):
            self.hidden_size = cli_args.hidden_size
        else:   
            self.hidden_size = cli_args.hidden_layer_sizes
            
        # self.activation = ACTIVATIONS[cli_args.activation]()
        self.weight_init = cli_args.weight_init
        self.optim = OPTIMIZERS[cli_args.optimizer](self.lr)

        # Creating the Neural Network
        self.layers = []
        input_size = 784
        num_classes = 10
        for i in range(0, self.num_layers):
            layer_activation = ACTIVATIONS[cli_args.activation]()
            layer = NeuralLayer(input_size, self.hidden_size[i], layer_activation, self.weight_init)
            input_size = self.hidden_size[i]
            self.layers.append(layer)
        layer = NeuralLayer(input_size, num_classes, ACTIVATIONS['linear'](), self.weight_init)
        self.layers.append(layer)

    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        A = X
        for i in range(0, len(self.layers)):
            A = self.layers[i].forward(A)
        return A
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        dA = self.loss_fn.backward(y_true, y_pred)

        grad_W_list = []
        grad_b_list = []

        for i in range(len(self.layers)-1, -1, -1):
            dA = self.layers[i].backward(dA)
            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(self.layers[i].grad_b)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optim.update(self.layers)
    
    def train(self, X_train, y_train, epochs=1, batch_size=32):
        """
        Train the network for specified epochs.
        """
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        num_samples = X_tr.shape[0]
        best_val_f1 = 0.0
        self.best_weights = None

        for epoch in range(epochs):
            
            shuffled_indices = np.random.permutation(num_samples)
            X_tr_shuffled = X_tr[shuffled_indices]
            y_tr_shuffled = y_tr[shuffled_indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_tr_shuffled[i : i+batch_size]
                y_batch = y_tr_shuffled[i : i+batch_size]

                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                self.update_weights()

            train_loss, train_acc, train_f1 = self.evaluate(X_tr, y_tr, batch_size)
            val_loss, val_acc, val_f1 = self.evaluate(X_val, y_val, batch_size)

            log_dict = {
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "layer_0_grad_norm": np.linalg.norm(self.layers[0].grad_W)  # For Question 2.4 we need this line
            }

            # Dynamically add dead neuron tracking for all hidden layers (Question 2.5)
            # We loop up to self.num_layers to exclude the final output layer
            for j in range(self.num_layers):
                if isinstance(self.layers[j].activation, ReLU):
                    # Calculate percentage of zero activations in the current batch's cache
                    dead_pct = np.mean(self.layers[j].activation.cache <= 0.0) * 100 
                    log_dict[f"layer_{j}_dead_neurons_pct"] = dead_pct

                    log_dict[f"layer_{j}_activation_dist"] = wandb.Histogram(self.layers[j].activation.cache)

                elif isinstance(self.layers[j].activation, Tanh):
                    zero_pct = np.mean(self.layers[j].activation.cache == 0.0) * 100
                    log_dict[f"layer_{j}_zero_neurons_pct"] = zero_pct

                    saturated_pct = np.mean(np.abs(self.layers[j].activation.cache) >= 0.99) * 100
                    log_dict[f"layer_{j}_saturated_neurons_pct"] = saturated_pct

            wandb.log(log_dict)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_weights = self.get_weights()
            
            print(f"Epoch-{epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, Val Accuracy = {val_acc:.4f}")
    

    def evaluate(self, X, y, batch_size=256):
        """
        Evaluate the network on given data.
        """
        num_samples = X.shape[0]
        total_loss = 0.0
        total_correct = 0

        all_predicted_classes = []
        
        for i in range(0, num_samples, batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            curr_batch_size = X_batch.shape[0]

            y_pred_batch = self.forward(X_batch)

            data_loss = self.loss_fn.forward(y_batch, y_pred_batch)
            total_loss += data_loss * curr_batch_size

            predicted_classes_batch = np.argmax(y_pred_batch, axis=1)
            actual_classes_batch = np.argmax(y_batch, axis=1)

            all_predicted_classes.extend(predicted_classes_batch)

            batch_correct = np.sum(predicted_classes_batch == actual_classes_batch)
            total_correct += batch_correct

        l2_penalty = 0
        for layer in self.layers:
            l2_penalty += np.sum(layer.W ** 2)
        l2_penalty = (self.weight_decay/2.0) * l2_penalty
        
        total_loss = (total_loss / num_samples) + l2_penalty
        accuracy = total_correct / num_samples

        y_true_classes = np.argmax(y, axis=1)

        f1 = f1_score(y_true_classes, all_predicted_classes, average='macro')

        return total_loss, accuracy, f1

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()