"""
#Trains a ResNet on the CIFAR10 dataset.

"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras import models, layers, optimizers
import tensorflow as tf
import numpy as np
import os
import pdb
import sys
import argparse

parser = argparse.ArgumentParser(description='Tensorflow Cifar10 Training')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--model', metavar='MODEL', type=str, help='specific model name', default='vgg19')
parser.add_argument('--lr', metavar='LEARNING_RATE', type=float, help='learning rate', default=0.001)
args = parser.parse_args()


# Training parameters
batch_size = args.batch_size  # orig paper trained all networks with batch_size=128
epochs = 10
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True


# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = args.lr #1e-3
    print('Learning rate: ', lr)
    return lr

model = models.Sequential()

if '16' in args.model:
    base_model = VGG16(weights=None, include_top=False, input_shape=(32, 32, 3), pooling=None)
elif '19' in args.model:
    base_model = VGG19(weights=None, include_top=False, input_shape=(32, 32, 3), pooling=None)




#base_model.summary()

#pdb.set_trace()

#model.add(layers.UpSampling2D((2,2)))
#model.add(layers.UpSampling2D((2,2)))
#model.add(layers.UpSampling2D((2,2)))
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))#, kernel_initializer='he_uniform'))
#model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))#, kernel_initializer='he_uniform'))
#model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))#, kernel_initializer='he_uniform'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])


# Saves the model after each epoch. Saved to the filepath, the latest best 
# model according to the quantity monitored will not be overwritten
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# Run training
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=[])

model.save('vgg19.h5')

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
