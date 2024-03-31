import numpy as np


def relu(x):
    """
    ReLU activation function.
        ReLU(x) = max(0, x)
    
    Args:
        x (ndarray): Input array.

    Returns:
        ndarray: Output array after applying ReLU activation.
    """
    return np.maximum(x, 0)


def relu_derivative(x):
    """
    Derivative of ReLU activation function.
        ReLU′(x)={ 1     if x>0
                 { 0     otherwise
    
    Args:
        x (ndarray): Input array.

    Returns:
        ndarray: Derivative of ReLU activation.
    """
    return np.where(x > 0, 1, 0)


def softmax(x):
    """
    Softmax activation function.
        pi = e^zi / Σ(e^zj), where i ranges from 1 to n.
    
    Args:
        x (ndarray): Input array.

    Returns:
        ndarray: Output array after applying softmax activation.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    """
    Forward pass through the neural network.
    
    Args:
        X (ndarray): Input data.
        W1 (ndarray): Weights of the first layer.
        b1 (ndarray): Bias of the first layer.
        W2 (ndarray): Weights of the second layer.
        b2 (ndarray): Bias of the second layer.

    Returns:
        tuple: A tuple containing four elements:
            - z1 (ndarray): Output of the first layer before activation.
            - a1 (ndarray): Output of the first layer after activation.
            - z2 (ndarray): Output of the second layer before activation.
            - a2 (ndarray): Output of the second layer after activation.
    """
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def backward(X, y, z1, a1, z2, a2, W1, W2, b1, b2):
    """
    Backward pass through the neural network.
    
    Args:
        X (ndarray): Input data.
        y (ndarray): True labels.
        z1 (ndarray): Output of the first layer before activation.
        a1 (ndarray): Output of the first layer after activation.
        z2 (ndarray): Output of the second layer before activation.
        a2 (ndarray): Output of the second layer after activation.
        W1 (ndarray): Weights of the first layer.
        W2 (ndarray): Weights of the second layer.
        b1 (ndarray): Bias of the first layer.
        b2 (ndarray): Bias of the second layer.

    Returns:
        tuple: A tuple containing four elements:
            - dW1 (ndarray): Gradient of the weights of the first layer.
            - db1 (ndarray): Gradient of the bias of the first layer.
            - dW2 (ndarray): Gradient of the weights of the second layer.
            - db2 (ndarray): Gradient of the bias of the second layer.
    """
    m = X.shape[0]

    # print(a2.shape, y.shape)
    delta_2 = a2 - y
    dW2 = a1.T @ delta_2 / m
    db2 = np.sum(delta_2 * 1, axis=0) / m

    delta_1 = delta_2 @ W2.T * relu_derivative(z1)
    dW1 = X.T @ delta_1 / m
    db1 = np.sum(delta_1 * 1, axis=0) / m

    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    """
    Update parameters using gradient descent.
    
    Args:
        W1 (ndarray): Weights of the first layer.
        b1 (ndarray): Bias of the first layer.
        W2 (ndarray): Weights of the second layer.
        b2 (ndarray): Bias of the second layer.
        dW1 (ndarray): Gradient of the weights of the first layer.
        db1 (ndarray): Gradient of the bias of the first layer.
        dW2 (ndarray): Gradient of the weights of the second layer.
        db2 (ndarray): Gradient of the bias of the second layer.
        learning_rate (float): Learning rate for gradient descent.

    Returns:
        tuple: A tuple containing four elements:
            - Updated weights and biases of the first layer (W1, b1).
            - Updated weights and biases of the second layer (W2, b2).
    """
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


def init_weights(input_size, hidden_size, output_size):
    """
    Initialize weights and biases for the neural network.
    
    Args:
        input_size (int): Size of the input layer.
        hidden_size (int): Size of the hidden layer.
        output_size (int): Size of the output layer.

    Returns:
        tuple: A tuple containing four elements:
            - Initialized weights and biases of the first layer (W1, b1).
            - Initialized weights and biases of the second layer (W2, b2).
    """
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
    b2 = np.zeros(output_size)
    return W1, b1, W2, b2
