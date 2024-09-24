#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:35:38 2020
@author: julian

Gradient descent for REGRESSION: quadratic loss function with implicit bias 
and with quadratic regularization
choose here 2 dimensions, i.e. vec_w = w. 
i.e. d = 2, n = 10
"""

import matplotlib.pyplot as plt
import numpy as np

plt.close()

d =   2       # number of features
n =  50       # number of training examples

offset = 2   # y-axis intercept

lam = 0.1 # regularization parameter lambda

np.random.seed(42)

X1 = np.random.uniform(-2, 2, n)
X2 = np.ones(n)
y  = X1 + offset + np.random.normal( 0, 0.5, n)

# Loss-function 
def L(w1, w2):
    return 1/n*np.sum((w1*X1 + w2*X2 - y)**2) + lam*(w1**2 + w2**2)
    
# derivative dL/dw
def dLw1(w1, w2): 
    return 2/n*np.sum((w1*X1 + w2*X2 - y)*X1) + 2*lam*w1

# derivative dL/dw
def dLw2(w1, w2): 
    return 2/n*np.sum((w1*X1 + w2*X2 - y)*X2) + 2*lam*w2


# Random start value:
w1 = np.random.uniform(-10, 10)
w2 = np.random.uniform(-10, 10)

print()
print("Startvalue:  ", w1, w2, L(w1, w2) )
print()

lr      = 0.001    # learning_rate
n_iter  = 10000  # number of iterations
cnt     = 0       # counter
n_print = 100     # print steps

print("Learning Rate:  ", lr)
print("Interations:    ", n_iter)
print()


while cnt < n_iter: 
    
    w1_new = w1 - lr*dLw1(w1, w2)
    w2_new = w2 - lr*dLw2(w2, w2)
    
    w1 = w1_new
    w2 = w2_new
    
    if cnt % n_print == 0:
        print("Current values:  ", cnt, w1, w2, L(w1, w2))
    
    cnt += 1
    
print()
print("Gradient Descent - slope, intercept:      ", np.round(w1, 4), np.round(w2, 4))
    
fit = w1*X1 + w2*X2
   
plt.xlabel("x-axis") 
plt.ylabel("y-axis")
plt.title("Linear Regression with Gradient Descent")
plt.plot(X1, y, 'o')
plt.plot(X1, fit)
plt.show()

#===============================================================================

## Calculate Closed-Form Solution

X  = np.stack((X1, X2), axis=1)

XT = np.transpose(X)

XX = np.linalg.inv( np.matmul(XT, X) + lam*np.eye(2) )

ww = np.matmul( np.matmul(XX, XT), y) 

print()
print("Closed-Form Solution - slope, intercept:  ", np.round(ww[0], 4), np.round(ww[1], 4))