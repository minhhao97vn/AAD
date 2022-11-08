# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison."""
from __future__ import absolute_import, division, print_function, unicode_literals

import pprint
import json
from ctypes import Union

from art.attacks import EvasionAttack
from art.attacks.evasion import BoundaryAttack, ZooAttack
from art.defences.trainer import AdversarialTrainer, AdversarialTrainerFBF, AdversarialTrainerFBFPyTorch
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier, PyTorchClassifier
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence, ProvenanceDefense, RONIDefense
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import warnings

from keras.regularizers import L1L2
from sklearn.metrics import f1_score

from adversarial_trainer_mod import AdversarialTrainerPredefinedAttacks

warnings.filterwarnings("ignore")

tf.compat.v1.disable_eager_execution()
from torch import nn, optim
import torch.nn.functional as F
import torch

seed = 1818

np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


class Net(torch.nn.Module):
    def __init__(self, input_size=11):
        super(Net, self).__init__()
        self.fc_1 = nn.Linear(in_features=input_size, out_features=6)
        self.fc_2 = nn.Linear(in_features=6, out_features=2)

    def forward(self, x):
        hidden1 = self.fc_1(x.float())
        relu1 = F.relu(hidden1)
        hidden2 = self.fc_2(relu1)
        output = F.softmax(hidden2)
        return output


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

    model = Net(input_size=11)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(model=model, clip_values=None, loss=criterion, optimizer=optimizer,
                                   input_shape=(-1, 11), nb_classes=2)

    # y_train_groups_cate = [to_categorical(group[:, 0]) for group in y_train_groups]

    # Calling poisoning defence:
    defense = AdversarialTrainerFBFPyTorch(classifier, eps=0.1, use_amp=False)  # prediction test label
    defense.fit(x=X_train[y_train[:, 1] == 0], y=to_categorical(y_train[y_train[:, 1] == 0][:, 0]),
                nb_epochs=200)

    # End-to-end method:
    print("------------------- Results using size metric -------------------")
    # print(defence.get_params())
    preds = np.argmax(defense.predict(X_test_group[0:7].reshape((700, 11))), axis=1)
    acc = np.sum(preds == np.argmax(to_categorical(y_test_group[0:7, :, 0].reshape(-1)), axis=1)) / \
          to_categorical(y_test_group[0:7, :, 0].reshape(-1)).shape[0]
    print("\nTest accuracy - all: %.2f%%" % (acc * 100))
    print(">>>>>>>>F1 score - all: ",
          f1_score(y_test_group[0:7, :, 1].reshape(-1), preds))

    acc = np.sum(preds[-300:700] == np.argmax(to_categorical(y_test_group[-3:7, :, 0].reshape(-1)), axis=1)) / 300
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
