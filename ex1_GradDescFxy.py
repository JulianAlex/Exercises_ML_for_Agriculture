#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:35:38 2020

@author: julian

Gradient descent for simple function of 2 parameters (Rosenbrock function)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.close()

a = 1    # parameters of Rosenbrock function
b = 100

def f(x1, x2):
    return ((a-x1)**2 + b*(x2-x1**2)**2)
    
def df1(x1, x2): # derivative d/dx1 f1(x)
    return 2*(x1-a) + 4*b*x1*(x1**2-x2)
    
def df2(x1, x2): # derivative d/dx1 f1(x)
    return 2*b*(x2-x1**2)

# Random starting point:
x1 = np.random.uniform(-2,2)
x2 = np.random.uniform(-2,2)

print("Startvalues:  ", x1, x2, f(x1, x2) )
print()

lr      = 0.001   # learning_rate
n_iter  = 10000   # number of iterations
cnt     = 0       # counter
n_print = 100     # print steps

f_steps = []

print("Learning Rate:  ", lr)
print("Interations:    ", n_iter)
print()


while cnt < n_iter: 
    
    x1_new = x1 - lr*df1(x1, x2)
    x2_new = x2 - lr*df2(x1, x2)
    
    x1 = x1_new
    x2 = x2_new
    
    cnt += 1
        
    if cnt % n_print == 0:
        print("Current values:  ", cnt, x1, x2, f(x1, x2))
        f_steps.append(f(x1,x2))

plt.figure()    
plt.xlabel("steps") 
plt.ylabel("f(x1,x2)")
plt.title("Find Minimum of a Function with Gradient Descent")
plt.plot(f_steps)
plt.show()

# --- Plot the Function -------------------------------------------------------

fig = plt.figure()
ax  = fig.gca(projection='3d')

# calculate data on grid
X1 = np.arange(-2, 2, 0.1)
X2 = np.arange(-1, 3, 0.1)
X1, X2 = np.meshgrid(X1, X2)
F = f(X1, X2)

# plot the surface
surf = ax.plot_surface(X1, X2, F, cmap=cm.coolwarm)#, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

