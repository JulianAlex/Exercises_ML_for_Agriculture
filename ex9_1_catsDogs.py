# -*- coding: utf-8 -*-
"""
Ex9_1, Julian Adolphs

using keras.io/guides/transfer_learning/

tested for Tensorflow 2.2.0
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers 

print("TF-Version:     ", tf.__version__)
print("Keras-Version:  ", keras.__version__)

t_start = time.time()

# epochs for training and fine-tuning the pretrained model
EPOCHS      = 20
EPOCHS_FINE = 50
# Start-Learning_rates for Learning Rate adaptation
LR_BEGIN    = 1e-3
LR_FINE_BEG = 1e-4

tfds.disable_progress_bar()

# load only 40% of the dataset for training, 10% for val. and 10% for testing
train_ds, valid_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  
)
print("No of train samples: %d" % tf.data.experimental.cardinality(train_ds))
print("No of train samples: %d" % tf.data.experimental.cardinality(valid_ds))
print("No of train samples: %d" % tf.data.experimental.cardinality(test_ds))

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")
    
# resize images to 150 for Xception-Net 
size = (150, 150)

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
valid_ds = valid_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds  = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

batch_size = 32

train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
valid_ds = valid_ds.cache().batch(batch_size).prefetch(buffer_size=10)
test_ds  = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

# Data Augmentation:
data_augm = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"), 
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
    layers.experimental.preprocessing.RandomCrop(130, 130)
])

# Show augmented data
for images, labels in train_ds.take(1):
    
    plt.figure(figsize=(10, 10))
    first_image = images[0]
    for i in range(9):
        ax         = plt.subplot(3, 3, i + 1)
        augm_image = data_augm(tf.expand_dims(first_image, 0), training=True)      
        plt.imshow(augm_image[0].numpy().astype("int32"))
        plt.title(int(labels[i]))
        plt.axis("off")

#--- Build the Base-Model -----------------------------------------------------
base_model = keras.applications.Xception(
    weights = "imagenet",  # Load weights pre-trained on ImageNet.
    input_shape = (150, 150, 3),
    include_top = False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape = (150, 150, 3))
x      = data_augm(inputs)  # Apply random data augmentation
#x = inputs  #no augmentation

# Pre-trained Xception weights requires that input is normalized
# from (0, 255) to a range (-1., +1.), the normalization layer
# does the following, outputs = (inputs - mean) / sqrt(var)
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean       = np.array([127.5] * 3)
var        = mean ** 2
# Scale inputs to [-1, +1]
x = norm_layer(x)
norm_layer.set_weights([mean, var])

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x       = base_model(x, training = False)
x       = keras.layers.GlobalAveragePooling2D()(x)
x       = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model   = keras.Model(inputs, outputs)

model.summary()

# adept learning rates 
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: LR_BEGIN*10**(-epoch/(EPOCHS)))

lr_fine_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: LR_FINE_BEG*10**(-epoch*2/(EPOCHS_FINE)))

optimizer = tf.keras.optimizers.Adam(learning_rate = LR_BEGIN)   # start lr

# --- Train the top layer -----------------------------------------------------
model.compile(
    optimizer = optimizer,
    loss      = keras.losses.BinaryCrossentropy(from_logits=True),
    metrics   = [keras.metrics.BinaryAccuracy()],
)

# evaluete the base-model without training
model.evaluate(valid_ds)

epochs  = EPOCHS
history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, 
                    callbacks = [lr_schedule])

acc    = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss     = history.history['loss']
val_loss  = history.history['val_loss']
learn_rate = history.history['lr']

#------------------------------------------------------------------------------
# --- Fine-Tune the whole model -----------------------------------------------

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.

base_model.trainable = True
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate = LR_FINE_BEG) 

model.compile(
    optimizer = optimizer, 
    loss      = keras.losses.BinaryCrossentropy(from_logits=True),
    metrics   = [keras.metrics.BinaryAccuracy()],
)
epochs = EPOCHS_FINE

history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, 
                    callbacks = [lr_fine_schedule])

ft_acc    = history.history['binary_accuracy']
ft_val_acc = history.history['val_binary_accuracy']
ft_loss     = history.history['loss']
ft_val_loss  = history.history['val_loss']
ft_learn_rate = history.history['lr']

#------------------------------------------------------------------------------

print(" ")
t_run = (time.time() - t_start) / 60
print(" --- %s minutes --- " % t_run )

# --- Plot the accuracy during epochs -----------------------------------------
epochs = range(1, len(loss) + len(ft_loss) + 1)

acc    += ft_acc
val_acc += ft_val_acc 
loss     += ft_loss
val_loss  += ft_val_loss
learn_rate += ft_learn_rate

plt.figure()
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, learn_rate)
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()