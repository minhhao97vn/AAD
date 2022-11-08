# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison."""
from __future__ import absolute_import, division, print_function, unicode_literals

import pprint
import json

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

seed = 2012

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

    classifier = KerasClassifier(model=model, clip_values=None)

    classifier.fit(X_train, to_categorical(y_train[:, 0]), nb_epochs=20, batch_size=50)

    # Calling poisoning defence:
    defense = SpectralSignatureDefense(classifier, X_test_group[0:7].reshape((700, 11)), to_categorical(y_test_group[0:7, :, 0].reshape(-1)), expected_pp_poison=0.2, batch_size=50, eps_multiplier=0.2)


    # End-to-end method:
    print("------------------- Results using size metric -------------------")
    # print(defence.get_params())
    results = defense.detect_poison(nb_clusters=2, nb_dims=9, reduce="PCA")
    is_predicted_clean = results[1]
    is_predicted_poisoned = [1 - i for i in is_predicted_clean]

    print(y_test_group[0:7, :, 1].reshape(-1).shape)

    print(">>>>>>>>Accuracy of detection - size metric: ",
          sum(is_predicted_poisoned == y_test_group[0:7, :, 1].reshape(-1)) / 700)
    print(">>>>>>>>F1 of detection - size metric: ",
          f1_score(y_test_group[0:7, :, 1].reshape(-1), is_predicted_poisoned))

    print(">>>>>>>>Accuracy of detection - size metric - group 1, 2, 3: ",
          sum(is_predicted_poisoned[-300:700] == y_test_group[4:7, :, 1].reshape(-1)) / 300)

    print(">>>>>>>>Accuracy of detection - size metric - group 4: ",
          sum(is_predicted_poisoned[0:100] == y_test_group[0, :, 1].reshape(-1)) / 100)

    print(">>>>>>>>Accuracy of detection - size metric - group 5: ",
          sum(is_predicted_poisoned[100:200] == y_test_group[1, :, 1].reshape(-1)) / 100)

    print(">>>>>>>>Accuracy of detection - size metric - group 6: ",
          sum(is_predicted_poisoned[200:300] == y_test_group[2, :, 1].reshape(-1)) / 100)

    print(">>>>>>>>Accuracy of detection - size metric - group 7: ",
          sum(is_predicted_poisoned[300:400] == y_test_group[3, :, 1].reshape(-1)) / 100)


if __name__ == "__main__":
    main()
