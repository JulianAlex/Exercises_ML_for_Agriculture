#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:42:38 2020

@author: julian
"""

import numpy as np

np.random.seed(1)

m  =  7  # image height
l  =  7  # image width
d  =  2  # no of channels
k  =  3  # conv.filter size k*k*d
p  =  1  # padding with p-zeros-margin  
kp =  2  # pooling kernel, stride = pooling kernel size  

im_min =  0  # min value image array random initialization
im_max =  3  # max value image array random initialization

#==============================================================================

m += 2*p 
l += 2*p

mm = m - k + 1  # image hight after convolution 
ll = l - k + 1  # image width after convolution

# numpy 3D array (d, m, l):  d-th 2D array, m-th row, l-th column 
image = np.zeros((d, m, l))

# Padding with zeros:
if(p > 0):
    image[:, p:-p, p:-p] = np.random.randint( im_min, im_max, size = (d, m-2*p, l-2*p))  
else:
    image = np.random.randint( im_min, im_max, size = (d, m, l))  

#==============================================================================
# Convolution:
    
filt = np.random.randint(0,  3, size = (d, k, k)) # squared conv.filter
conv = np.zeros((mm, ll))

def convolution(image, filt):
    for dd in range(d):
        for i in range(mm):
            for j in range(ll):
                # += for sum over the different channels s
                conv[i, j] += np.tensordot(image[dd, i : i+k, j : j+k], filt[dd, :, :])
    return conv

conv = convolution(image, filt)

print("\nImage:")
print(image)
print("\nFilter:")
print(filt)
print("\nConvolution:")
print(conv)
print()

# === Pooling =================================================================
# mm*ll image size after convolution 

# pooling filter fits exactly n times in image: 
if( (mm % kp == 0) and (ll % kp == 0) ):
    
    pool = np.zeros((mm//kp, ll//kp))
    
    def pooling(image):
        # stepsize = stride = pooling kernel size
        for i in range(0, mm, kp):
            for j in range(0, ll, kp):
                pool[i//kp, j//kp] = np.average(conv[i : i+kp, j : j+kp])
                #print(i, j, pool)
        return pool
    
    pool = pooling(conv)
    
    print("\nAverage Pooling:")
    print(pool)
    

else:
# padding / filling with zeros    
    mmod = mm % kp   # reminder = modulus in height direction
    lmod = ll % kp   # reminder = modulus in width direction
    
    mpol = mm//kp + 1  # new size after pooling
    lpol = ll//kp + 1
    
    pool = np.zeros((mpol, lpol))
    
    # even rest => padd both sides with rest/2 zeros    
    pad_conv = np.zeros((mm + kp - mmod, ll + kp - lmod))
    pad_conv[(kp-mmod)//2 : -(kp-mmod)//2, (kp-lmod)//2 : -(kp-lmod)//2] = conv   

    print("\nConvolution after Padding:")
    print(pad_conv)
    print()

    def pooling(image):
        for i in range(0, pad_conv.shape[0], kp):
            for j in range(0, pad_conv.shape[1], kp):
                pool[i//kp, j//kp] = np.average(pad_conv[i : i+kp, j : j+kp])
        return pool

    pool = pooling(pad_conv)

    print("\nAverage Pooling:")
    print(np.round(pool, 1))


    
    
    















