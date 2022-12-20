import numpy as np
import matplotlib.pyplot as plt
from matplotlib.style import use


def add_ones(X):
    return np.c_[np.ones((X.shape[0], 1)), X]


def rmse(Y, T):
    return np.sqrt(np.mean((T - Y) ** 2))


def forward_layer1(X, U):
    # Calculate the output of the first layer, using the tanh activation function
    Zu = np.tanh(X @ U)
    return Zu


def forward_layer2(Zu, V):
    Zv = np.tanh(Zu @ V.T)
    return Zv


def forward_layer3(Zv, W):
    Y = Zv @ W.T
    return Y


def forward(X, U, V, W):
    Zu = forward_layer1(X, U)
    Zv = forward_layer2(Zu, V)
    Y = forward_layer3(Zv, W)
    return Zu, Zv, Y


def backward_layer3(T, Y):
    delta_layer3 = T - Y
    return delta_layer3


def backward_layer2(delta_layer3, W, Zv):
    delta_layer2 = delta_layer3 @ W[:, 1:] * (1 - Zv ** 2)
    return delta_layer2


def backward_layer1(delta_layer2, V, Zu):
    delta_layer1 = delta_layer2 @ V[:, 1:] * (1 - Zu ** 2)
    return delta_layer1


def gradients(X, T, Zu, Zv, Y, V, W):
    delta_layer3 = backward_layer3(T, Y)
    delta_layer2 = backward_layer2(delta_layer3, W, Zv)
    delta_layer1 = backward_layer1(delta_layer2, V, Zu)
    grad_U = X.T @ delta_layer1
    grad_V = Zu.T @ delta_layer2
    grad_W = Zv.T @ delta_layer3
    return grad_U, grad_V, grad_W





