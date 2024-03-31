import pandas as pd
import numpy as np


def load_data(train_path, test_path):
    """
    Function to load data and one hot encode.

    Args:
        train_path (str): File path to the training data CSV file.
        test_path (str): File path to the test data CSV file.

    Returns:
        tuple: A tuple containing four elements:
            - X_train (DataFrame): Training features.
            - y_train (ndarray): One-hot encoded training labels.
            - X_test (DataFrame): Test features.
            - num_classes (int): Number of classes in the dataset.
    """
    train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    X_train = train.drop('label', axis=1)
    labels = train['label']
    num_classes = labels.max() + 1
    y_train = np.eye(num_classes)[labels]
    y_train = y_train.astype(int)

    return X_train, y_train, X_test, num_classes
