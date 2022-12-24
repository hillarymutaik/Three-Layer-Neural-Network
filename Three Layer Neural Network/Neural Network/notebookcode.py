#!/usr/bin/env python
# coding: utf-8

# # A1: Three-Layer Neural Network

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-2">Example Results</a></span></li><li><span><a href="#Discussion" data-toc-modified-id="Discussion-3">Discussion</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will start with code from lecture notes 04 and add code to do the following. You will implement and apply a neural network as in lecture notes 04 but now with an additional hidden layer.  The resulting three-layer network will consist of three weight matrices, `U`, `V` and `W`.
# 
# First, implement the forward pass to calculate outputs of each layer:
# 
# * Define functions `add_ones` and `rmse` by copying it from the lecture notes.
# * Define function `forward_layer1` with two arguments, the input `X` and the first layer's weights `U`. It calculates and returns the output, `Zu`, of the first layer, using the `tanh` activation function.
# * Define function `forward_layer2` with two arguments, the input `Zu` and the second layer's weights `V`. It calculates and returns the output, `Zv`, of the second layer, using the `tanh` activation function.
# * Define function `forward_layer3` with two arguments, the input `Zv` and the third layer's weights `W`. It calculates and returns the output, `Y`, of the third layer as just the weighted sum of the inputs, without an activation function.
# * Define function `forward` with four arguments, the input `X` to the network and the weight matrices, `U`, `V` and `W` of the three layers. It calls the above three functions and returns the outputs of all layers, `Zu`, `Zv`, `Y`.
# 
# Now implement the backward pass that calculates `delta` values for each layer:
# 
# * Define function `backward_layer3` that accepts as arguments the target values `T` and the predicted values `Y` calculated by function `forward`. It calculates and returns `delta_layer3` for layer 3, which is just `T - Y`.
# * Define function `backward_layer2` that accepts as arguments `delta_layer3`, `W` and `Zv` and calculates and returns `delta` for layer 2 by back-propagating `delta_layer3` through `W`.
# * Define function `backward_layer1` that accepts as arguments `delta_layer2`, `V` and `ZU` and calculates and returns `delta` for layer 1 by back-propagating `delta_layer2` through `V`.
# * Define function `gradients` that accepts as arguments `X`, `T`, `Zu`, `Zv`, `Y`, `U`, `V`, and `W`, and calls the above three functions and uses the results to calculate the gradient of the mean squared error between `T` and `Y` with respect to `U`, `V` and `W` and returns those three gradients.
# 
# Now you can use `forward` and `gradients` to define the function `train` to train a three-layer neural network.
#           
# * Define function `train` that returns the resulting values of `U`, `V`, and `W` and the `X` and `T` standardization parameters.  Arguments are unstandardized `X` and `T`, the number of units in each of the two hidden layers, the number of epochs and the learning rate. This function standardizes `X` and `T`, initializes `U`, `V` and `W` to uniformly distributed random values between -0.1 and 0.1, and updates `U`, `V` and `W` by the learning rate times their gradients for `n_epochs` times as shown in lecture notes 04.  This function must call `forward`, `gradients` and `add_ones`.  It must also collect in a list called `rmse_trace` the root-mean-square errors for each epoch between `T` and `Y`.
# 
#       def train(X, T, n_units_U, n_units_V, n_epochs, rho):
#           .
#           .
#           .
#           return rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds
# 
# Then we need a function `use` that calculates an output `Y` for new samples.  
# 
# * Define function `use` that accepts unstandardized `X`, standardization parameters, and weight matrices `U`, `V`, and `W` and returns the unstandardized output.
# 
#       def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
#           .
#           .
#           .
#           Y = ....
#           return Y

# ## Example Results

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import train as train
import use as use
get_ipython().run_line_magic('matplotlib', 'inline')


# Add code cells here to define the functions above.  Once these are correctly defined, the following cells should run and produce the same results as those here.

# In[2]:


Xtrain = np.arange(4).reshape(-1, 1)
Ttrain = Xtrain ** 2

Xtest = Xtrain + 0.5
Ttest = Xtest ** 2

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[3]:


U = np.array([[1, 2, 3], [4, 5, 6]])  # 2 x 3 matrix, for 2 inputs (include constant 1) and 3 units
V = np.array([[-1, 3], [1, 3], [-2, 1], [2, -4]]) # 2 x 3 matrix, for 3 inputs (include constant 1) and 2 units
W = np.array([[-1], [2], [3]])  # 3 x 1 matrix, for 3 inputs (include constant 1) and 1 output unit
U.shape, V.shape, W.shape


# In[4]:


X_means = np.mean(Xtrain, axis=0)
X_stds = np.std(Xtrain, axis=0)
Xtrain_st = (Xtrain - X_means) / X_stds
Xtrain_st


# In[5]:


T_means = np.mean(Ttrain, axis=0)
T_stds = np.std(Ttrain, axis=0)
Ttrain_st = (Ttrain - T_means) / T_stds
Ttrain_st


# In[7]:


from A1mysolution import forward_layer1

Zu = forward_layer1(Xtrain_st, U)
Zu


# In[8]:


from A1mysolution import forward_layer2

Zv = forward_layer2(Zu, V)
Zv


# In[9]:


from A1mysolution import forward_layer3

Y = forward_layer3(Zv, W)
Y


# In[10]:


from A1mysolution import forward

Zu, Zv, Y = forward(Xtrain_st, U, V, W)
print(f'{Zu=}')
print(f'{Zv=}')
print(f'{Y=}')


# In[11]:


from A1mysolution import backward_layer3

delta_layer3 = backward_layer3(Ttrain_st, Y)
delta_layer3


# In[12]:


from A1mysolution import backward_layer2

delta_layer2 = backward_layer2(delta_layer3, W, Zv)
delta_layer2


# In[13]:


from A1mysolution import backward_layer1

delta_layer1 = backward_layer1(delta_layer2, V, Zu)
delta_layer1


# In[14]:


from A1mysolution import gradients

grad_U, grad_V, grad_W = gradients(Xtrain_st, Ttrain_st, Zu, Zv, Y, U, V, W)
print(f'{grad_U=}')
print(f'{grad_V=}')
print(f'{grad_W=}')


# In[15]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
Y


# In[16]:


rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train


# In[17]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
np.hstack((Ttrain, Y))


# In[18]:


plt.plot(rmse_trace)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


# Here is another example with a little more interesting data.

# In[19]:


n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))


# In[20]:


rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 5, 5, 100, 0.01)


# In[21]:


plt.plot(rmse_trace)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


# In[22]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)


# In[23]:


plt.plot(Xtrain, Ttrain)
plt.plot(Xtrain, Y);


# In[24]:


from matplotlib.style import use

rmse_trace, U, V, W, X_means, X_stds, T_means, T_stds = train
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

# In[31]:


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
