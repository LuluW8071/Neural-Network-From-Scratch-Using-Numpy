from tqdm import tqdm
from neuralnet import forward, backward, update_parameters
import numpy as np
import matplotlib.pyplot as plt

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

            dW1, db1, dW2, db2 = backward(X_batch, y_batch, z1, a1, z2, a2, W1, W2, b1, b2)
            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

            if step % 100 == 0:
                loss = mean_squared_error(y_batch, a2)
                acc = accuracy(y_batch, a2)
                histories["epoch"].append(epoch)
                histories["step"].append(step)
                histories["loss"].append(loss)
                histories["accuracy"].append(acc)
                loop.set_postfix(loss=loss, accuracy=acc)
            step += 1

    return W1, b1, W2, b2, histories

def plot_training_history(histories_df_grouped):
    accs = histories_df_grouped['accuracy']['mean']
    accs_min = histories_df_grouped['accuracy']['min']
    accs_max = histories_df_grouped['accuracy']['max']
    losses = histories_df_grouped['loss']['mean']
    losses_min = histories_df_grouped['loss']['min']
    losses_max = histories_df_grouped['loss']['max']

    plt.figure(figsize=(12, 5))
    plt.rcParams['axes.grid'] = True

    plt.subplot(1, 2, 1)
    plt.plot(accs, color='b')
    plt.fill_between(range(len(accs)), accs_min, accs_max, color='b', alpha=0.2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(losses, color='r')
    plt.fill_between(range(len(losses)), losses_min, losses_max, color='r', alpha=0.2)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

def predict(X, W1, b1, W2, b2):
    _, _, _, a2 = forward(X, W1, b1, W2, b2)
    return a2

