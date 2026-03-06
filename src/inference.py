import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import load_and_preprocess_data
from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'], help='Choose the dataset to train on.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Size of batch for training.')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'mean_squared_error'], help='Choose the Loss function used for training.')
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop', choices=['sgd', 'momentum', 'nag', 'rmsprop'], help='Choose the optimizer used for weight update.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Choose learning rate for training.')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Weight decay for L2 regulrization.')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help='Choose the number of hidden layers in the model.')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 64], help='List of hidden layer sizes.')
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'], help='Choose activation function for the Hidden Layers.')
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['random', 'xavier', 'zero'], help='Choose the weight initialization strategy.')
    parser.add_argument('-w_p', '--wandb_project', type=str, help='Choose a name for Wandb Report.')
    parser.add_argument('-mp', '--model_save_path', type=str, default="best_model.npy", help='Relative location to save the best model.')

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
    """
    logits = model.forward(X_test)
    loss_value = model.loss_fn.forward(y_test, logits)
    predicted_classes = np.argmax(logits, axis=1)
    actual_classes = np.argmax(y_test, axis=1)

    results = {
        'logits': logits,
        'loss': loss_value,
        'accuracy': accuracy_score(actual_classes, predicted_classes),
        'f1': f1_score(actual_classes, predicted_classes, average='macro'),
        'precision': precision_score(actual_classes, predicted_classes, average='macro'),
        'recall': recall_score(actual_classes, predicted_classes, average='macro')
    }

    return results


def main():
    """
    Main inference function.
    """
    args = parse_arguments()

    # Update the args to match the best model configuration
    try:
        with open("best_config.json", "r") as f:
            saved_config = json.load(f)
        vars(args).update(saved_config)
    except FileNotFoundError:
        print("No best_config.json found, relying on the CLI arguments...")

    print(f"Loading {args.dataset} dataset...")
    _, _, X_test, y_test_oh = load_and_preprocess_data(args.dataset)

    print("Loading the model...")
    model = NeuralNetwork(args)
    weights = load_model(args.model_save_path)
    model.set_weights(weights)

    print("Evaluating the model...")
    results = evaluate_model(model, X_test, y_test_oh)
    print(results)
    
    print("Evaluation complete!")

    return results


if __name__ == '__main__':
    main()