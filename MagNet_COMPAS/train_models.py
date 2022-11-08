## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified for the needs of MagNet.

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf
from keras.optimizer_v2.gradient_descent import SGD
from keras.regularizers import L1L2

from setup_mnist import MNIST, COMPAS
import os


def train(data, file_name, params, num_epochs=50, batch_size=128):
    """
    Standard neural network training procedure.
    """
    model = Sequential()
    model.add(Dense(6,
                    activation='relu',
                    kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                    input_dim=data.train_data.shape[1]))
    model.add(Dense(2,
                    kernel_regularizer=L1L2(l1=0.0, l2=0.1)))

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])

    print(data.train_labels.shape)

    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)

    if file_name != None:
        model.save(file_name)

    return model


if not os.path.isdir('models'):
    os.makedirs('models')

train(COMPAS(), "models/example_classifier", [32, 32, 64, 64, 200, 200],
      num_epochs=20)
