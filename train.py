from tqdm import tqdm
from neuralnet import *
import numpy as np

def get_batch(X, y, batch_size):
    n_batches = X.shape[0] // batch_size
    for i in range(n_batches):
        X_batch = X[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]
        yield X_batch, y_batch

def mean_squared_error(y_true, y_pred):
    return np.sum((y_pred - y_true) ** 2) / y_true.shape[0]

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def train(X, y, W1, b1, W2, b2, learning_rate, epochs, batch_size):
    histories = {
        "epoch": [],
        "step": [],
        "loss": [],
        "accuracy": []
    }
    
    step = 1
    loop = tqdm(range(epochs))
    for epoch in loop:
        for X_batch, y_batch in get_batch(X, y, batch_size):
            z1, a1, z2, a2 = forward(X_batch, W1, b1, W2, b2)

            # y_batch one-hot encoded
            # y_batch_onehot = np.eye(10)[y_batch]

            dW1, db1, dW2, db2 = backward(X_batch, np.eye(10)[y_batch], z1, a1, z2, a2, W1, W2, b1, b2)
            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

            if step % 100 == 0:
                loss = mean_squared_error(y_batch, a2)
                acc = accuracy(y_batch, a2)
                histories["epoch"].append(epoch + 1)
                histories["step"].append(step)
                histories["loss"].append(loss)
                histories["accuracy"].append(acc)
                loop.set_postfix(loss=loss, accuracy=acc)
            step += 1

    return W1, b1, W2, b2, histories

def predict(X, W1, b1, W2, b2):
    _, _, _, a2 = forward(X, W1, b1, W2, b2)
    return a2

