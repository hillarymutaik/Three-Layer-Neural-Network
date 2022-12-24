#!/usr/bin/env python
# coding: utf-8

# # A1: Three-Layer Neural Network

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def add_ones(X):
    """Add a column of ones to X."""
    return np.hstack((np.ones((X.shape[0], 1)), X))


def rmse(Y, T):
    """Calculate root mean squared error between Y and T."""
    return np.sqrt(np.mean((Y - T)**2))


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
