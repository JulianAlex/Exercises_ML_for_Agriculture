#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:58:55 2020

@author: julian
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical

#import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

#  X <-> images, y <-> labels
(X_train, y_train), (X_test, y_test) = data.load_data()

X_train = X_train/255.
X_test  = X_test/255.

# X_train.shape = (60000, 28, 28)  =>  (60000, 28, 28, 1) 
X_train = X_train.reshape(len(X_train), 28, 28, 1) 
X_test  = X_test.reshape( len(X_test),  28, 28, 1)

class_names = ['T-Shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle Boot' ]

total_classes = len(class_names)

# change to one-hot-encoding
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)


plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# create sequential model:
model = Sequential()

# add layers:
model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(14,14,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', input_shape=(7,7,1)))
model.add(Flatten()) 

model.add(Dense(1024, activation = 'relu'))
model.add(Dense( 512, activation = 'relu'))
model.add(Dense( 256, activation = 'relu'))

model.add(Dense(  10, activation='softmax'))  # one-hot-encoding !!

# compile the model:
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

# train the model:
model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), epochs = 12)



eval_loss, eval_accuracy = model.evaluate(X_test, y_test, verbose=False)
print()
print('Model accuracy:  %.4f' % eval_accuracy)

