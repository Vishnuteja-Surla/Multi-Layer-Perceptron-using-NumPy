import argparse
import wandb
import json

import numpy as np
from utils.data_loader import load_and_preprocess_data
from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    # Adding the Arguments
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
    parser.add_argument('-mp', '--model_save_path', type=str, default="src/best_model.npy", help='Relative location to save the best model.')

    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    wandb.init(
        project=args.wandb_project if args.wandb_project else "da6401-assignment-1",
        config=vars(args)
    )

    print(f"Loading {args.dataset} dataset...")
    X_train, y_train_oh, X_test, y_test_oh = load_and_preprocess_data(args.dataset)

    print("Initializing the Neural Network...")
    model = NeuralNetwork(args)

    print("Starting the training...")
    model.train(X_train, y_train_oh, args.epochs, args.batch_size)

    if model.best_weights is not None:
        np.save(args.model_save_path, model.best_weights)

        with open("src/best_config.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        print("Best model and configuration saved successfully!")
    
    wandb.finish()
    print("Training complete!")


if __name__ == '__main__':
    main()