#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:36:15 2020

@author: julian

Automatic differentiation and gradient tape
"""

import tensorflow as tf

print("\nTF-Version:  ", tf.__version__)

x0 = tf.Variable(1.2)  # Create a Tensorflow variable initialized to 1.2
x1 = tf.Variable(3.1)
w0 = tf.Variable(1.5)
w1 = tf.Variable(-1.5)
w2 = tf.Variable(2.0)

# =============================================================================

# Function  f(x0,x1,w0,w1,w2) = ( tanh(1./(w0*x0 + w1*x1 + w2)) - 1 )**2

with tf.GradientTape(persistent = True) as t:  
    # the persistent = True means that one can execute the function multiple times 
                
    f = ( tf.math.tanh(1./(w0*x0 + w1*x1 + w2)) - 1 )**2
        
    dfw0 = t.gradient(f, w0)        
    dfw1 = t.gradient(f, w1)        
    dfw2 = t.gradient(f, w2)    
    dfx0 = t.gradient(f, x0)
    dfx1 = t.gradient(f, x1)    

print("\n\nGradients:")
print()
print("df/dw0 = %9.6f" % dfw0)
print("df/dw1 = %9.6f" % dfw1)
print("df/dw2 = %9.6f" % dfw2)
print()
print("df/dx0 = %9.6f" % dfx0)
print("df/dx1 = %9.6f" % dfx1)
print()


# =============================================================================

# Function  f(x0,x1,w0,w1,w2) = ( tanh(1./(w0*x0 + w1*x1 + w2)) - 1 )**2

with tf.GradientTape(persistent = True) as t:  
                
    # forward pass:
    g7 = x0*w0
    g6 = x1*w1    
    g5 = g7 + g6          
    g4 = g5 + w2
    g3 = 1./g4
    g2 = tf.math.tanh(g3)    
    g1 = g2 - 1
    f  = (g1)**2
    
    dfg1 = t.gradient(f, g1)  
    dfg2 = t.gradient(f, g2) 
    dfg3 = t.gradient(f, g3) 
    dfg4 = t.gradient(f, g4)
    dfg5 = t.gradient(f, g5)
    dfw2 = t.gradient(f, w2)    
    dfg6 = t.gradient(f, g6)        
    dfg7 = t.gradient(f, g7)
    dfw1 = t.gradient(f, w1)        
    dfx1 = t.gradient(f, x1)    
    dfw0 = t.gradient(f, w0)        
    dfx0 = t.gradient(f, x0)

print("\nGradients:")
print()
print("df/dg1 = %9.6f" % dfg1)
print("df/dg2 = %9.6f" % dfg2)
print("df/dg3 = %9.6f" % dfg3)
print("df/dg4 = %9.6f" % dfg4)
print("df/dg5 = %9.6f" % dfg5)
print()
print("df/dg7 = %9.6f" % dfg7)
print("df/dw0 = %9.6f" % dfw0)
print("df/dx0 = %9.6f" % dfx0)
print()
print("df/dg6 = %9.6f" % dfg6)
print("df/dw1 = %9.6f" % dfw1)
print("df/dx1 = %9.6f" % dfx1)
print()
