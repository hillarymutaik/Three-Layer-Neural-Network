#!/usr/bin/env python
# coding: utf-8

# # A1: Three-Layer Neural Network
import numpy as np


def add_ones(X):
    """Add a column of ones to X.

    Parameters
    ----------
    X : array
        The input array with shape (n_samples, n_features).

    Returns
    -------
    array
        The modified array with shape (n_samples, n_features+1).
    """
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def rmse(Y, T):
    """Calculate the root-mean-square error between the predicted values Y and the target values T.

    Parameters
    ----------
    Y : array
        The predicted values with shape (n_samples, n_outputs).
    T : array
        The target values with shape (n_samples, n_outputs).

    Returns
    -------
    float
        The root-mean-square error.
    """
    return np.sqrt(np.mean((Y - T)**2))


def forward_layer1(X, U):
    """Calculate the output of the first hidden layer.

    Parameters
    ----------
    X : array
        The input array with shape (n_samples, n_features).
    U : array
        The weight matrix for the first hidden layer with shape (n_features+1, n_units_U).

    Returns
    -------
    array
        The output of the first hidden layer with shape (n_samples, n_units_U).
    """
    Zu = np.tanh(np.dot(add_ones(X), U))
    return Zu


def forward_layer1(X, U):
    """Calculate output of first layer, using tanh activation function."""
    Zu = np.tanh(X @ U)
    return Zu


def forward_layer2(Zu, V):
    """Calculate output of second layer, using tanh activation function."""
    Zv = np.tanh(Zu @ V)
    return Zv


def forward_layer3(Zv, W):
    """Calculate output of third layer as just the weighted sum of the inputs, without an activation function."""
    Y = Zv @ W
    return Y


def forward(X, U, V, W):
    """Calculate outputs of all layers."""
    Zu = forward_layer1(X, U)
    Zv = forward_layer2(Zu, V)
    Y = forward_layer3(Zv, W)
    return Zu, Zv, Y


def backward_layer3(T, Y):
    """Calculate delta for layer 3."""
    delta_layer3 = T - Y
    return delta_layer3


def backward_layer2(delta_layer3, W, Zv):
    """Calculate delta for layer 2 by back-propagating delta_layer3 through W."""
    delta_layer2 = delta_layer3 @ W.T * (1 - Zv**2)
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
