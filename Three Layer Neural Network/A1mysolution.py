#!/usr/bin/env python
# coding: utf-8

# # A1: Three-Layer Neural Network

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


Xtrain = np.arange(4).reshape(-1, 1)
Ttrain = Xtrain ** 2

Xtest = Xtrain + 0.5
Ttest = Xtest ** 2

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


U = np.array([[1, 2, 3], [4, 5, 6]])  # 2 x 3 matrix, for 2 inputs (include constant 1) and 3 units
V = np.array([[-1, 3], [1, 3], [-2, 1], [2, -4]]) # 2 x 3 matrix, for 3 inputs (include constant 1) and 2 units
W = np.array([[-1], [2], [3]])  # 3 x 1 matrix, for 3 inputs (include constant 1) and 1 output unit
U.shape, V.shape, W.shape


X_means = np.mean(Xtrain, axis=0)
X_stds = np.std(Xtrain, axis=0)
Xtrain_st = (Xtrain - X_means) / X_stds



T_means = np.mean(Ttrain, axis=0)
T_stds = np.std(Ttrain, axis=0)
Ttrain_st = (Ttrain - T_means) / T_stds



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


Zu, Zv, Y = forward(Xtrain_st, U, V, W)
print(f'{Zu=}')
print(f'{Zv=}')
print(f'{Y=}')


def backward_layer3(T, Y):
    delta_layer3 = T - Y
    return delta_layer3

delta_layer3 = backward_layer3(Ttrain_st, Y)


def backward_layer2(delta_layer3, W, Zv):
    delta_layer2 = delta_layer3 @ W[:, 1:] * (1 - Zv ** 2)
    return delta_layer2

delta_layer2 = backward_layer2(delta_layer3, W, Zv)



def backward_layer1(delta_layer2, V, Zu):
    delta_layer1 = delta_layer2 @ V[:, 1:] * (1 - Zu ** 2)
    return delta_layer1

delta_layer1 = backward_layer1(delta_layer2, V, Zu)



def gradients(X, T, Zu, Zv, Y, V, W):
    delta_layer3 = backward_layer3(T, Y)
    delta_layer2 = backward_layer2(delta_layer3, W, Zv)
    delta_layer1 = backward_layer1(delta_layer2, V, Zu)
    grad_U = X.T @ delta_layer1
    grad_V = Zu.T @ delta_layer2
    grad_W = Zv.T @ delta_layer3
    return grad_U, grad_V, grad_W

grad_U, grad_V, grad_W = gradients(Xtrain_st, Ttrain_st, Zu, Zv, Y, U, V, W)
print(f'{grad_U=}')
print(f'{grad_V=}')
print(f'{grad_W=}')


from matplotlib.style import use

Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
Y


from torchio.transforms.preprocessing.intensity.histogram_standardization import train

rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 10, 10, 1000, 0.05)


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
np.hstack((Ttrain, Y))


plt.plot(rmse_trace)
plt.xlabel('Epoch')
plt.ylabel('RMSE')

n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))


rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 5, 5, 100, 0.01)


plt.plot(rmse_trace)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)


plt.plot(Xtrain, Ttrain)
plt.plot(Xtrain, Y)


rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 10, 5, 10000, 0.1)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rmse_trace)
plt.xlabel('Epoch')
plt.ylabel('RMSE')

plt.subplot(1, 2, 2)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.xlabel('Input')
plt.ylabel('Target and Output')
plt.legend()
