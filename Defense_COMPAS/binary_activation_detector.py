# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison."""
from __future__ import absolute_import, division, print_function, unicode_literals

import pprint
import json

from art.defences.detector.evasion import BinaryInputDetector, BinaryActivationDetector
from art.estimators.classification.classifier import ClassifierNeuralNetwork
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence, ProvenanceDefense
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import warnings

from keras.regularizers import L1L2

warnings.filterwarnings("ignore")

tf.compat.v1.disable_eager_execution()

seed = 18

np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


def main():
    # Read dataset:
    train_data = np.load('data/AD_compas_train.npz')
    test_data = np.load('data/AD_compas_test.npz')

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    # Shuffle training data so poison is not together
    n_train = X_train.shape[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    # Create Keras neural network - basic architecture from Keras examples

    model = Sequential()
    model.add(Dense(11,
                    activation='relu',
                    kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                    input_dim=X_train.shape[1]))
    model.add(Dense(6,
                    activation='softmax',
                    kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
    model.add(Dense(2,
                    activation='softmax',
                    kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model1 = Sequential()
    model1.add(Dense(11,
                     activation='relu',
                     kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                     input_dim=X_train.shape[1]))
    model1.add(Dense(6,
                     activation='softmax',
                     kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
    model1.add(Dense(2,
                     activation='softmax',
                     kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
    model1.compile(optimizer='sgd',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    classifier = KerasClassifier(model=model, clip_values=None)
    detector = KerasClassifier(model=model1, clip_values=None)

    classifier.fit(X_train, to_categorical(y_train[:, 0]), nb_epochs=20, batch_size=50)
    detector.fit(X_train, to_categorical(y_train[:, 1]), nb_epochs=20, batch_size=50)


    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(X_test), axis=1)
    acc = np.sum(preds == np.argmax(to_categorical(y_test[:, 0]), axis=1)) / y_test[:, 0].shape[0]
    print("\nTest accuracy: %.2f%%" % (acc * 100))

    # Evaluate the classifier on poisonous data
    preds = np.argmax(classifier.predict(X_test[y_test[:, 1] == 1]), axis=1)
    acc = np.sum(preds == np.argmax(to_categorical(y_test[y_test[:, 1] == 1][:, 0]), axis=1)) / \
          y_test[y_test[:, 1] == 1].shape[0]
    print("\nPoisonous test set accuracy (i.e. effectiveness of poison): %.2f%%" % (acc * 100))

    # Evaluate the classifier on clean data
    preds = np.argmax(classifier.predict(X_test[y_test[:, 1] == 0]), axis=1)
    acc = np.sum(preds == np.argmax(to_categorical(y_test[y_test[:, 1] == 0][:, 0]), axis=1)) / \
          y_test[y_test[:, 1] == 0].shape[0]
    print("\nClean test set accuracy: %.2f%%" % (acc * 100))

    # Calling evasion defence:
    defense = BinaryActivationDetector(classifier, detector, layer=0)

    defense.fit(X_train, to_categorical(y_train[:, 1]), nb_epochs=20, batch_size=50)

    # Evaluate the classifier on the test set
    preds = np.argmax(defense.predict(X_test, batch_size=50), axis=1)
    print(preds[0:30])
    acc = np.sum(preds == np.argmax(to_categorical(y_test[:, 1]), axis=1)) / y_test[:, 1].shape[0]
    print("\nTest detection accuracy: %.2f%%" % (acc * 100))


if __name__ == "__main__":
    main()
