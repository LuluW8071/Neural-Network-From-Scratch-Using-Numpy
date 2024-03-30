import pandas as pd
import numpy as np

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    X_train = train.drop('label', axis=1)
    labels = train['label']
    num_classes = labels.max() + 1
    y_train = np.eye(num_classes)[labels]
    y_train = y_train.astype(int)

    return X_train, y_train, X_test, num_classes
