import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
#%% md
# Add code cells here to define the functions above.  Once these are correctly defined, the following cells should run and produce the same results as those here.
#%%
Xtrain = np.arange(4).reshape(-1, 1)
Ttrain = Xtrain ** 2

Xtest = Xtrain + 0.5
Ttest = Xtest ** 2

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape
#%%
U = np.array([[1, 2, 3], [4, 5, 6]])  # 2 x 3 matrix, for 2 inputs (include constant 1) and 3 units
V = np.array([[-1, 3], [1, 3], [-2, 1], [2, -4]]) # 2 x 3 matrix, for 3 inputs (include constant 1) and 2 units
W = np.array([[-1], [2], [3]])  # 3 x 1 matrix, for 3 inputs (include constant 1) and 1 output unit
U.shape, V.shape, W.shape
#%%
X_means = np.mean(Xtrain, axis=0)
X_stds = np.std(Xtrain, axis=0)
Xtrain_st = (Xtrain - X_means) / X_stds
Xtrain_st
#%%
T_means = np.mean(Ttrain, axis=0)
T_stds = np.std(Ttrain, axis=0)
Ttrain_st = (Ttrain - T_means) / T_stds
Ttrain_st
#%%
from assignment1 import forward_layer1

Zu = forward_layer1(Xtrain_st, U)
Zu
#%%
from assignment1 import forward_layer2

Zv = forward_layer2(Zu, V)
Zv
#%%
from assignment1 import forward_layer3

Y = forward_layer3(Zv, W)
Y
#%%
from assignment1 import forward

Zu, Zv, Y = forward(Xtrain_st, U, V, W)
print(f'{Zu=}')
print(f'{Zv=}')
print(f'{Y=}')
#%%
from assignment1 import backward_layer3

delta_layer3 = backward_layer3(Ttrain_st, Y)
delta_layer3
#%%
from assignment1 import backward_layer2

delta_layer2 = backward_layer2(delta_layer3, W, Zv)
delta_layer2
#%%
from assignment1 import backward_layer1

delta_layer1 = backward_layer1(delta_layer2, V, Zu)
delta_layer1
#%%
from assignment1 import gradients

grad_U, grad_V, grad_W = gradients(Xtrain_st, Ttrain_st, Zu, Zv, Y, V, W)
print(f'{grad_U=}')
print(f'{grad_V=}')
print(f'{grad_W=}')
#%%
from matplotlib.style import use

Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
Y
#%%
from assignment1 import train
rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 10, 10, 1000, 0.05)
#%%
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
np.hstack((Ttrain, Y))
#%%
plt.plot(rmse_trace)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
#%% md
# Here is another example with a little more interesting data.
#%%
n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))
#%%
from assignment1 import train

rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 5, 5, 100, 0.01)
#%%
plt.plot(rmse_trace)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
#%%
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
#%%
plt.plot(Xtrain, Ttrain)
plt.plot(Xtrain, Y);
#%%
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
plt.legend();