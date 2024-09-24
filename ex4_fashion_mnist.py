#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:58:55 2020

@author: julian
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255.
test_images  = test_images/255.

class_names = ['T-Shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle Boot' ]

total_classes = len(class_names)

#train_vec_labels = keras.utils.to_categorical(train_labels, total_classes)
#test_vec_labels  = keras.utils.to_categorical(test_labels, total_classes)
'''
plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

model = keras.Sequential([
    
    keras.layers.Flatten(input_shape = (28, 28)), 
    
    keras.layers.Dense(1024, activation = 'relu'),
    keras.layers.Dense( 512, activation = 'relu'),
    keras.layers.Dense( 256, activation = 'relu'),
    keras.layers.Dense( 128, activation = 'relu'),
    keras.layers.Dense(  64, activation = 'relu'),
    
    keras.layers.Dense(  10 )   
])

# 
# logit = log_e(probability)
# SparseCategCrossentrop for n categories with n labels
# CategoricalCrossentrop for n categories in one-hot-encoding 
model.compile(optimizer = 'adam', 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics = ['accuracy'])

model.fit(train_images, train_labels, batch_size=32, validation_data=(test_images, test_labels), epochs = 10)

eval_loss, eval_accuracy = model.evaluate(test_images, test_labels, verbose=False)

print()
print('Model accuracy:  %.4f' % eval_accuracy)


