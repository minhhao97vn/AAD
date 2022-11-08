# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison."""
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

warnings.filterwarnings("ignore")

tf.compat.v1.disable_eager_execution()

seed = 1881

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

    classifier = KerasClassifier(model=model, clip_values=None)

    classifier.fit(X_train[y_train[:, 1] == 0], to_categorical(y_train[y_train[:, 1] == 0][:, 0]), nb_epochs=30,
                   batch_size=50)

    X_test_all = X_test_group[0:7].reshape((700, 11))
    y_test_all = y_test_group[0:7, :, 0].reshape(-1)
    y_dec_all = y_test_group[0:7, :, 1].reshape(-1)

    # Calling defence:
    test_squeezed_samples = FeatureSqueezing(clip_values=(np.amin(X_test_all), np.amax(y_test_all)), apply_fit=False,
                                             apply_predict=True)(X_test_all, to_categorical(y_test_all))

    prediction_squeezed = classifier.predict(test_squeezed_samples[0])
    prediction_original = classifier.predict(X_test_all)
    detection = []

    threshold = 0.008
    for idx in range(prediction_original.shape[0]):
        distance = l1_distance(prediction_squeezed[idx], prediction_original[idx])
        detection.append(distance > threshold)

    # End-to-end method:
    print("------------------- Results using size metric -------------------")
    # print(defence.get_params())
    preds = np.argmax(classifier.predict(test_squeezed_samples[0]), axis=1)
    acc = np.sum(preds == np.argmax(test_squeezed_samples[1], axis=1)) / test_squeezed_samples[1].shape[0]
    print("\nTest accuracy - all: %.2f%%" % (acc * 100))
    print("F1 score - all: ",
          f1_score(np.argmax(test_squeezed_samples[1], axis=1), preds))
    acc = np.sum(detection == y_dec_all) / y_dec_all.shape[0]
    print("Detection Test accuracy - all: %.2f%%" % (acc * 100))
    print("Detection F1 score - all: ",
          f1_score(detection, y_dec_all))

    acc = np.sum(preds[-300:700] == np.argmax(test_squeezed_samples[1][-300:700], axis=1)) / 300
    print("\nTest accuracy - 1,2,3: %.2f%%" % (acc * 100))
    acc = np.sum(detection[-300:700] == y_dec_all[-300:700]) / 300
    print("Detection accuracy - 1,2,3: %.2f%%" % (acc * 100))

    acc = np.sum(preds[0:100] == np.argmax(test_squeezed_samples[1][0:100], axis=1)) / 100
    print("\nTest accuracy - 4: %.2f%%" % (acc * 100))
    acc = np.sum(detection[0:100] == y_dec_all[0:100]) / 100
    print("Detection accuracy - 4: %.2f%%" % (acc * 100))

    acc = np.sum(preds[100:200] == np.argmax(test_squeezed_samples[1][100:200], axis=1)) / 100
    print("\nTest accuracy - 5: %.2f%%" % (acc * 100))
    acc = np.sum(detection[100:200] == y_dec_all[100:200]) / 100
    print("Detection accuracy - 5: %.2f%%" % (acc * 100))

    acc = np.sum(preds[200:300] == np.argmax(test_squeezed_samples[1][200:300], axis=1)) / 100
    print("\nTest accuracy - 6: %.2f%%" % (acc * 100))
    acc = np.sum(detection[200:300] == y_dec_all[200:300]) / 100
    print("Detection accuracy - 6: %.2f%%" % (acc * 100))

    acc = np.sum(preds[300:400] == np.argmax(test_squeezed_samples[1][300:400], axis=1)) / 100
    print("\nTest accuracy - 7: %.2f%%" % (acc * 100))
    acc = np.sum(detection[300:400] == y_dec_all[300:400]) / 100
    print("Detection accuracy - 7: %.2f%%" % (acc * 100))

def l1_distance(p, p_squeezed):
    return np.sum(np.abs(p - p_squeezed))

if __name__ == "__main__":
    main()
