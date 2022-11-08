# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison.
 This code is revised from IBM ART."""
from __future__ import absolute_import, division, print_function, unicode_literals

import pprint
import json

from art.defences.preprocessor import FeatureSqueezing
from art.defences.transformer.evasion import DefensiveDistillation
from art.defences.transformer.poisoning import NeuralCleanse
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
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

tf.compat.v1.disable_eager_execution()

seed = 1881

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

    print('x: ', x_train.shape)
    print('y: ', y_test.shape)

    y_labels = np.argmax(y_test[:, 0:-1], axis=1)
    (value, counts) = np.unique(y_labels, return_counts=True)
    for idx in range(len(counts)):
        print("Class {} has {} samples".format(value[idx], counts[idx]))

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
    classifier = KerasClassifier(model=model, clip_values=(0.0, 1.0))

    classifier.fit(x_train[y_train[:, -1] == 0].reshape((-1, 28, 28, 1)), y_train[y_train[:, -1] == 0][:, 0:-1],
                   nb_epochs=3,
                   batch_size=64)

    # Calling defence:
    test_squeezed_samples = FeatureSqueezing(clip_values=(0.0, 1.0), apply_fit=False, apply_predict=True)(
        x_test.reshape((-1, 28, 28, 1)), y_test[:, 0:-1])

    prediction_squeezed = classifier.predict(test_squeezed_samples[0])
    prediction_original = classifier.predict(x_test.reshape((-1, 28, 28, 1)))
    detection = []

    threshold = 0.002
    for idx in range(prediction_original.shape[0]):
        distance = l1_distance(prediction_squeezed[idx], prediction_original[idx])
        detection.append(distance > threshold)

    # End-to-end method:
    print("------------------- Results using size metric -------------------")
    # print(defence.get_params())
    preds = np.argmax(classifier.predict(test_squeezed_samples[0]), axis=1)
    acc = np.sum(preds == np.argmax(test_squeezed_samples[1], axis=1)) / test_squeezed_samples[1].shape[0]
    print("Test accuracy - all: %.2f%%" % (acc * 100))
    print("F1 score - all: ",
          f1_score(test_squeezed_samples[1], np.eye(10)[preds], average='macro'))
    acc = np.sum(detection == y_test[:, -1]) / test_squeezed_samples[1].shape[0]
    print("Detection Test accuracy - all: %.2f%%" % (acc * 100))
    print("Detection F1 score - all: ",
          f1_score(detection, y_test[:, -1]))

    acc = np.sum(preds[-6000:14000] == np.argmax(test_squeezed_samples[1][-6000:14000], axis=1)) / 6000
    print("\nTest accuracy - 1,2,3: %.2f%%" % (acc * 100))
    acc = np.sum(detection[-6000:14000] == y_test[-6000:14000, -1]) / 6000
    print("Detection Test accuracy - 1,2,3: %.2f%%" % (acc * 100))

    acc = np.sum(preds[0:2000] == np.argmax(test_squeezed_samples[1][0:2000], axis=1)) / 2000
    print("\nTest accuracy - 4: %.2f%%" % (acc * 100))
    acc = np.sum(detection[0:2000] == y_test[0:2000, -1]) / 2000
    print("Detection Test accuracy - 4: %.2f%%" % (acc * 100))

    acc = np.sum(preds[2000:4000] == np.argmax(test_squeezed_samples[1][2000:4000], axis=1)) / 2000
    print("\nTest accuracy - 5: %.2f%%" % (acc * 100))
    acc = np.sum(detection[2000:4000] == y_test[2000:4000, -1]) / 2000
    print("Detection Test accuracy - 5: %.2f%%" % (acc * 100))

    acc = np.sum(preds[4000:6000] == np.argmax(test_squeezed_samples[1][4000:6000], axis=1)) / 2000
    print("\nTest accuracy - 6: %.2f%%" % (acc * 100))
    acc = np.sum(detection[4000:6000] == y_test[4000:6000, -1]) / 2000
    print("Detection Test accuracy - 6: %.2f%%" % (acc * 100))

    acc = np.sum(preds[6000:8000] == np.argmax(test_squeezed_samples[1][6000:8000], axis=1)) / 2000
    print("\nTest accuracy - 7: %.2f%%" % (acc * 100))
    acc = np.sum(detection[6000:8000] == y_test[6000:8000, -1]) / 2000
    print("Detection Test accuracy - 7: %.2f%%" % (acc * 100))


def l1_distance(p, p_squeezed):
    return np.sum(np.abs(p - p_squeezed))


if __name__ == "__main__":
    main()
