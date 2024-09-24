#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:55:06 2020

@author: julian

# Fit a Linear Model

1. Define the model.
2. Define a loss function.
3. Obtain training data.
4. Run through the training data and use an "optimizer" to adjust the variables 
   to fit the data.
   
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d # import Axes3D

# f(x) = x*W + b
# W = 3.0, b = 2.0

# 1. Define the model.
class Model(object):
  def __init__(self):
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    self.W1 = tf.Variable(5.0)
    self.W2 = tf.Variable(2.0)
    self.b  = tf.Variable(1.0)

  def __call__(self, x1, x2):
    return self.W1 * x1 + self.W2 * x2 + self.b

model = Model()


# 2. Define a loss function.
def loss(target_y, predicted_y):
    #return tf.reduce_mean(tf.abs(target_y - predicted_y))
    return tf.reduce_mean(tf.square(target_y - predicted_y))

# 3. Obtain training data.
TRUE_W1 = 3.0
TRUE_W2 = 1.0
TRUE_b  = 2.0
NUM_EXAMPLES = 1000

input1  = tf.random.uniform(shape=[NUM_EXAMPLES], minval = -2, maxval = 2, seed = 1)
input2  = tf.random.uniform(shape=[NUM_EXAMPLES], minval = -2, maxval = 2, seed = 1)
noise   = tf.random.normal(shape=[NUM_EXAMPLES], stddev = 2.0)

outputs = input1 * TRUE_W1 + input2 * TRUE_W2 + TRUE_b + noise

print('Start loss: %1.6f' % loss(model(input1, input2), outputs).numpy())
print()

def train(model, input1, input2, outputs, learning_rate):
    
    with tf.GradientTape() as t:
        current_loss = loss(outputs, model(input1, input2))

    dW1, dW2, db = t.gradient(current_loss, [model.W1, model.W2, model.b])
    
    model.W1.assign_sub(learning_rate * dW1)
    model.W2.assign_sub(learning_rate * dW2)
    model.b.assign_sub(learning_rate * db)


W1s, W2s, bs = [], [], []

epochs = range( 20 )

for epoch in epochs:

    W1s.append(model.W1.numpy())
    W2s.append(model.W2.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(outputs, model(input1, input2))

    train(model, input1, input2, outputs, learning_rate=0.1)
    print('Epoch %2d:   W1=%1.2f  W2 =%1.2f  b=%1.2f,   loss=%2.5f' %
          (epoch, W1s[-1], W2s[-1], bs[-1], current_loss))


plt.plot(epochs, W1s, 'r', epochs, W2s, 'g', epochs, bs, 'b')
plt.plot([TRUE_W1] * len(epochs), 'r--', [TRUE_W2] * len(epochs), 'g--', [TRUE_b] * len(epochs), 'b--')
plt.xlabel('epochs')
plt.ylabel('values of param. W1, W2, b')
plt.legend(['W1', 'W2', 'b', 'True_W1', 'True_W2', 'True_b'], loc='upper right')
plt.show()


# --- Plot the Function -------------------------------------------------------

def plane(W1, W2, b, x1, x2):
    return W1 * x1 + W2 * x2 + b

fig = plt.figure()
ax  = plt.axes(projection='3d')

# calculate data on grid
X1     = np.arange(-2, 2, 0.1)
X2     = np.arange(-2, 2, 0.1)
X1, X2 = np.meshgrid(X1, X2)
ffit   = plane(W1s[-1], W2s[1], bs[-1], X1, X2)
#forg   = plane(TRUE_W1, TRUE_W2, TRUE_b, X1, X2)

surf = ax.plot_surface(X1, X2, ffit, cmap=cm.coolwarm)  # plane
#fig.colorbar(surf, shrink=0.5, aspect=5)
ax.plot3D(input1, input2, outputs, 'ob')  # point cloud

plt.xlabel('x')
plt.ylabel('y')

plt.show()