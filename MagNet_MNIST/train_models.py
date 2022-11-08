

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf
from keras.optimizer_v2.gradient_descent import SGD

from setup_mnist import MNIST, AADMNIST
import os

seed = 8
tf.random.set_seed(
    seed
)


def train(data, file_name, params, num_epochs=50, batch_size=128):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    model.add(Conv2D(params[0], (3, 3), input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

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

train(AADMNIST(), "models/example_classifier", [32, 32, 64, 64, 200, 200], num_epochs=50)
