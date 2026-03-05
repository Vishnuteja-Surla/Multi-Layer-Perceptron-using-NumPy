import numpy as np
from keras.datasets import mnist, fashion_mnist

def load_and_preprocess_data(dataset_name):
    """
    Loads, flattens and one-hot encodes the dataset.
    """
    # 1. Load the Data
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'.")
    
    # 2. Flatten the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 4. One-hot encode the labels
    y_train_oh = np.eye(10)[y_train]
    y_test_oh = np.eye(10)[y_test]

    return X_train, y_train_oh, X_test, y_test_oh