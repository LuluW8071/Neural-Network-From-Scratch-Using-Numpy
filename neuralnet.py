import numpy as np

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def backward(X, y, z1, a1, z2, a2, W1, W2, b1, b2):
    m = X.shape[0]
    
    # print(a2.shape, y.shape)
    delta_2 = a2 - y
    dW2 = a1.T @ delta_2 / m
    db2 = np.sum(delta_2 * 1, axis=0) / m
    
    delta_1 = delta_2 @ W2.T * relu_derivative(z1)
    dW1 = X.T @ delta_1 / m
    db1 = np.sum(delta_1 * 1, axis=0) / m
    
    return dW1, db1, dW2, db2


def init_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
    b2 = np.zeros(output_size)
    return W1, b1, W2, b2