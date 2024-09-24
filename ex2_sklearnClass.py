#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:35:38 2020
@author: julian

Gradient descent for CLASSIFICATION: 
loss function -log(p) without regularization
choose here 1 dimensions, i.e. vec_w = w. 
i.e. d = 2, k = 3
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn.linear_model import LogisticRegression

d =  2  # number of features (x, y coordinate) 
k =  3  # number of classes (0, 1, 2) 
n = 18  # number of training examples

np.random.seed(1)

# Start value for W-Matrix:
W = np.random.uniform(-2, 2, d*k).reshape(k, d)

X = np.array([(-1.,2.5),(-2.,5.),(-1.5,4.),(-1.,2.3),(-2.5,6.5),(-1.8,4.), 
              (-1.2,-2.5),(-2.3,-3.),(-1.8,-4.),(-1.9,-2.3),(-2.9,-3.5),(-1.7,-4.), 
              (1.,-4.5),(0.2,5.),(0.5,-3.),(1.3,2.3),(2.5,-1.0),(1.8,3.)])

# class labels: 
labels = np.array([0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2])
colors = ['red', 'blue', 'green']


clf = LogisticRegression(penalty='l2', tol=0.0001, C=100., fit_intercept=False, random_state=0, 
                         solver='lbfgs', max_iter=100, multi_class='auto', 
                         verbose=0, warm_start=False, n_jobs=None)

fit = clf.fit(X, labels)

# W-Matrix from the sklearn-function
W = fit.coef_

## intersection line of the planes (no bias, hence they go through (0,0):
## f_i seperates class 0 from 1;    f_j: 1 from 2;   f_k:  3 from 1
m_i = (W[0,0]-W[1,0])/(W[1,1]-W[0,1]) 
m_j = (W[1,0]-W[2,0])/(W[2,1]-W[1,1]) 
m_k = (W[0,0]-W[2,0])/(W[2,1]-W[0,1]) 
xx  = np.linspace(-6, 6, 2)
f_i = m_i*xx 
f_j = m_j*xx 
f_k = m_k*xx 
print()
print("Decision boundary slopes: ", np.round(m_i, 4), np.round(m_j, 4), np.round(m_k, 4) )

plt.figure()
plt.xlabel("x1") 
plt.ylabel("x2")
plt.title("Training data with intersection lines")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.scatter(X[:,0], X[:,1], c = labels, cmap = mpl.colors.ListedColormap(colors))
plt.plot(xx, f_i, '--k')
plt.plot(xx, f_j, '--k')
plt.plot(xx, f_k, '--k')
plt.show()


### Make class-predictions for some test-values: ------------------------------ 

x_test = np.array([(-4., 4.5), (-4., -5.), (0.1, 4.), (2.5, 5.5), (-0.1,-1.0)])
pred   = clf.predict(x_test)

print("Test Samples:")
print(x_test)
print()
print("Class-Prediction")
print(pred)

plt.figure()
plt.xlabel("x1") 
plt.ylabel("x2")
plt.title("Test Samples (cyan)")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.scatter(X[:,0], X[:,1], c = labels, cmap = mpl.colors.ListedColormap(colors))
plt.plot(x_test[:,0], x_test[:,1], 'oc' )
plt.plot(xx, f_i, '--k')
plt.plot(xx, f_j, '--k')
plt.plot(xx, f_k, '--k')
plt.show()


### Vizualize the decision boundaries -----------------------------------------

XX     = (np.random.random_sample((2000, 2)) - 0.5)*12.
y_pred = clf.predict(XX)

plt.figure()
plt.xlabel("x1") 
plt.ylabel("x2")
plt.title("Classification Boundaries")
plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.scatter(XX[:,0], XX[:,1], c = y_pred, cmap = mpl.colors.ListedColormap(colors))
plt.plot(xx, f_i, '--k')
plt.plot(xx, f_j, '--k')
plt.plot(xx, f_k, '--k')
plt.show()


