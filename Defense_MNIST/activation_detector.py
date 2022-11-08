# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison
 This code is revised from IBM ART."""
from __future__ import absolute_import, division, print_function, unicode_literals

import pprint
import json

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence, ProvenanceDefense, RONIDefense
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import warnings
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from keras.regularizers import L1L2

warnings.filterwarnings("ignore")

tf.compat.v1.disable_eager_execution()

seed = 1818

np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


def main():
    # Read dataset:
    train_data = np.load('data/AAD_MNIST_Comb_1_train.npz')
    test_data = np.load('data/AAD_MNIST_Comb_1_test.npz')

    x_train = train_data['X_all']
    y_train = train_data['y_all']
    x_test = test_data['X_all']
    y_test = test_data['y_all']
    x_test_group = test_data['X_groups']
    y_test_group = test_data['y_groups']

    # Shuffle training data so poison is not together
    n_train = x_train.shape[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    # Shuffle test data so poison is not together
    n_test = x_test.shape[0]
    shuffled_test_indices = np.arange(n_test)
    np.random.shuffle(shuffled_test_indices)
    x_test_shuffle = x_test[shuffled_test_indices]
    y_test_shuffle = y_test[shuffled_test_indices]

    # Create Keras neural network - basic architecture from Keras examples
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

    classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

    classifier.fit(x_test_shuffle.reshape((-1, 28, 28, 1)), y_test_shuffle[:, 0:-1], batch_size=64, nb_epochs=5)

    # Calling poisoning defence:
    defense = ActivationDefence(classifier, x_test.reshape((-1, 28, 28, 1)),
                                y_test[:, 0:-1])

    # End-to-end method:
    print("------------------- Results using size metric -------------------")
    # print(defence.get_params())
    results = defense.detect_poison(nb_clusters=5, nb_dims=10, reduce="PCA")
    is_predicted_clean = results[1]
    is_predicted_poisoned = [1 - i for i in is_predicted_clean]

    print(y_test.shape)

    print(">>>>>>>>Accuracy of detection - size metric: ",
          sum(is_predicted_poisoned == y_test[:, -1]) / 14000)
    print(">>>>>>>>F1 of detection - size metric: ",
          f1_score(y_test[:, -1], is_predicted_poisoned))

    # print(y_test[0:50, -1])
    # print(y_test[-50:, -1])

    print(">>>>>>>>Accuracy of detection - size metric - group 1, 2, 3: ",
          sum(is_predicted_poisoned[8000:14000] == y_test[8000:14000, -1]) / 6000)

    print(">>>>>>>>Accuracy of detection - size metric - group 4: ",
          sum(is_predicted_poisoned[0:2000] == y_test[0:2000, -1]) / 2000)

    print(">>>>>>>>Accuracy of detection - size metric - group 5: ",
          sum(is_predicted_poisoned[2000:4000] == y_test[2000:4000, -1]) / 2000)

    print(">>>>>>>>Accuracy of detection - size metric - group 6: ",
          sum(is_predicted_poisoned[4000:6000] == y_test[4000:6000, -1]) / 2000)

    print(">>>>>>>>Accuracy of detection - size metric - group 7: ",
          sum(is_predicted_poisoned[6000:8000] == y_test[6000:8000, -1]) / 2000)


if __name__ == "__main__":
    main()
