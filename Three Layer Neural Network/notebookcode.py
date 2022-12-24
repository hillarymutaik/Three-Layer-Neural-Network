#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')


# Add code cells here to define the functions above.  Once these are correctly defined, the following cells should run and produce the same results as those here.

# In[178]:


Xtrain = np.arange(4).reshape(-1, 1)
Ttrain = Xtrain ** 2

Xtest = Xtrain + 0.5
Ttest = Xtest ** 2

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[179]:


U = np.array([[1, 2, 3], [4, 5, 6]])  # 2 x 3 matrix, for 2 inputs (include constant 1) and 3 units
V = np.array([[-1, 3], [1, 3], [-2, 1], [2, -4]]) # 2 x 3 matrix, for 3 inputs (include constant 1) and 2 units
W = np.array([[-1], [2], [3]])  # 3 x 1 matrix, for 3 inputs (include constant 1) and 1 output unit
U.shape, V.shape, W.shape


# In[180]:


X_means = np.mean(Xtrain, axis=0)
X_stds = np.std(Xtrain, axis=0)
Xtrain_st = (Xtrain - X_means) / X_stds
Xtrain_st


# In[181]:


T_means = np.mean(Ttrain, axis=0)
T_stds = np.std(Ttrain, axis=0)
Ttrain_st = (Ttrain - T_means) / T_stds
Ttrain_st


# In[182]:


from A1mysolution import forward_layer1

Zu = forward_layer1(Xtrain_st, U)
Zu


# In[ ]:


def forward_layer2(Zu, V):
    Zv = np.tanh(Zu @ V.T)
    return Zv

Zv = forward_layer2(Zu, V)
Zv


# In[ ]:


def forward_layer3(Zv, W):
    Y = Zv @ W.T
    return Y

Y = forward_layer3(Zv, W)
Y


# In[ ]:


def forward(X, U, V, W):
    Zu = forward_layer1(X, U)
    Zv = forward_layer2(Zu, V)
    Y = forward_layer3(Zv, W)
    return Zu, Zv, Y

Zu, Zv, Y = forward(Xtrain_st, U, V, W)
print(f'{Zu=}')
print(f'{Zv=}')
print(f'{Y=}')


# In[ ]:


def backward_layer3(T, Y):
    delta_layer3 = T - Y
    return delta_layer3

delta_layer3 = backward_layer3(Ttrain_st, Y)
delta_layer3


# In[ ]:


def backward_layer2(delta_layer3, W, Zv):
    delta_layer2 = delta_layer3 @ W[:, 1:] * (1 - Zv ** 2)
    return delta_layer2

delta_layer2 = backward_layer2(delta_layer3, W, Zv)
delta_layer2


# In[ ]:


def backward_layer1(delta_layer2, V, Zu):
    delta_layer1 = delta_layer2 @ V[:, 1:] * (1 - Zu ** 2)
    return delta_layer1

delta_layer1 = backward_layer1(delta_layer2, V, Zu)
delta_layer1


# In[ ]:


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


# In[ ]:


from matplotlib.style import use

Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
Y


# In[ ]:

from torchio.transforms.preprocessing.intensity.histogram_standardization import train

rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 10, 10, 1000, 0.05)


# In[ ]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
np.hstack((Ttrain, Y))


# In[ ]:


plt.plot(rmse_trace)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


# Here is another example with a little more interesting data.

# In[ ]:


n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))


# In[ ]:


rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 5, 5, 100, 0.01)


# In[ ]:


plt.plot(rmse_trace)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


# In[ ]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)


# In[ ]:


plt.plot(Xtrain, Ttrain)
plt.plot(Xtrain, Y);


# In[ ]:


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


# Your plots will probably differ from these results, because you start with different random weight values.

# ## Discussion

# In this markdown cell, describe what difficulties you encountered in completing this assignment. What parts were easy for you and what parts were hard?

# # Grading
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A1grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A1grader.tar) <font color="red">(updated August 28th)</font> and extract `A1grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 10 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  A perfect execution score from this grading script does not guarantee that you will receive a perfect execution score from the final grading script.
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A1.ipynb' with 'Lastname' being your last name, and then save this notebook.

# In[ ]:


get_ipython().run_line_magic('run', '-i A1grader.py')


# # Check-In
# 
# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A1.ipynb```.  So, for me it would be ```Anderson-A1.ipynb```.  Submit the file using the ```Assignment 1``` link on [Canvas](https://colostate.instructure.com/courses/151263).

# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.
