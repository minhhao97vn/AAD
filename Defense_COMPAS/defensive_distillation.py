# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison."""
from __future__ import absolute_import, division, print_function, unicode_literals

import pprint
import json

from art.defences.transformer.evasion import DefensiveDistillation
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence, ProvenanceDefense, RONIDefense, SpectralSignatureDefense
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import warnings

from keras.regularizers import L1L2
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

tf.compat.v1.disable_eager_execution()

seed = 123

np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


def main():
    # Read dataset:
    train_data = np.load('data/AAD_COMPAS_train.npz')
    test_data = np.load('data/AAD_COMPAS_test.npz')

    X_train = train_data['X_all']
    y_train = train_data['y_all']
    X_test = test_data['X_all']
    y_test = test_data['y_all']

    X_train_group = train_data['X_groups']
    y_train_group = train_data['y_groups']
    X_test_group = test_data['X_groups']
    y_test_group = test_data['y_groups']

    # Shuffle training data so poison is not together
    n_train = X_train.shape[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    # Shuffle test data so poison is not together
    n_test = X_test.shape[0]
    shuffled_test_indices = np.arange(n_test)
    np.random.shuffle(shuffled_test_indices)
    X_test_shuffle = X_test[shuffled_test_indices]
    y_test_shuffle = y_test[shuffled_test_indices]

    # Create Keras neural network - basic architecture from Keras examples

    model = Sequential()
    model.add(Dense(6,
                    activation='relu',
                    kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                    input_dim=X_train.shape[1]))
    model.add(Dense(2,
                    activation='softmax',
                    kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    classifier_F = KerasClassifier(model=model, clip_values=None)
    classifier_F.fit(X_train[y_train[:, 1] == 0], to_categorical(y_train[:, 0][y_train[:, 1] == 0]), nb_epochs=30,
                     batch_size=50)

    model_F_prime = Sequential()
    model_F_prime.add(Dense(6,
                            activation='relu',
                            kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                            input_dim=X_train.shape[1]))
    model_F_prime.add(Dense(2,
                            activation='softmax',
                            kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
    model_F_prime.compile(optimizer='sgd',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    classifier_F_prime = KerasClassifier(model=model, clip_values=None)

    defense = DefensiveDistillation(classifier_F, nb_epochs=1)
    transformed_classifier = defense(X_train[y_train[:, 1] == 0], classifier_F_prime)


    # End-to-end method:
    print("------------------- Results using size metric -------------------")
    # print(defence.get_params())
    preds = np.argmax(transformed_classifier.predict(X_test_group[0:7].reshape((700, 11))), axis=1)
    acc = np.sum(preds == np.argmax(to_categorical(y_test_group[0:7, :, 0].reshape(-1)), axis=1)) / \
          to_categorical(y_test_group[0:7, :, 0].reshape(-1)).shape[0]
    print("\nTest accuracy - all: %.2f%%" % (acc * 100))

    print(">>>>>>>>F1 score - all: ",
          f1_score(y_test_group[0:7, :, 1].reshape(-1), preds))

    acc = np.sum(preds[-300:700] == np.argmax(to_categorical(y_test_group[4:7, :, 0].reshape(-1)), axis=1)) / 300
    print("\nTest accuracy - 1,2,3: %.2f%%" % (acc * 100))

    acc = np.sum(preds[0:100] == np.argmax(to_categorical(y_test_group[0, :, 0].reshape(-1)), axis=1)) / 100
    print("\nTest accuracy - 4: %.2f%%" % (acc * 100))

    acc = np.sum(preds[100:200] == np.argmax(to_categorical(y_test_group[1, :, 0].reshape(-1)), axis=1)) / 100
    print("\nTest accuracy - 5: %.2f%%" % (acc * 100))

    acc = np.sum(preds[200:300] == np.argmax(to_categorical(y_test_group[2, :, 0].reshape(-1)), axis=1)) / 100
    print("\nTest accuracy - 6: %.2f%%" % (acc * 100))

    acc = np.sum(preds[300:400] == np.argmax(to_categorical(y_test_group[3, :, 0].reshape(-1)), axis=1)) / 100
    print("\nTest accuracy - 7: %.2f%%" % (acc * 100))


if __name__ == "__main__":
    main()
