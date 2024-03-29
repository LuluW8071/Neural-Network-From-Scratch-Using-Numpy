import pandas as pd

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    X_train = train.drop('label', axis=1)
    y_train = train['label']
    return X_train, y_train, X_test
