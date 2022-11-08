## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified for MagNet's use.

import numpy as np
import os
import gzip
import urllib.request

from keras.models import load_model
from keras.utils.np_utils import to_categorical


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:
                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)+0.5
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)+0.5
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

    @staticmethod
    def print():
        return "MNIST"


class COMPAS:
    def __init__(self):
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
        ori_train_data = X_train[shuffled_indices]
        ori_train_labels = y_train[shuffled_indices]

        train_data = ori_train_data[ori_train_labels[:, 1] == 0]
        train_labels = ori_train_labels[ori_train_labels[:, 1] == 0][:, 0]

        ori_test_data = X_test
        ori_test_labels = y_test

        self.test_data = ori_test_data[ori_test_labels[:, 1] == 0]
        self.test_labels = to_categorical(ori_test_labels[ori_test_labels[:, 1] == 0][:, 0])

        VALIDATION_SIZE = 200


        self.validation_data = train_data[:VALIDATION_SIZE, :]
        self.validation_labels = to_categorical(train_labels[:VALIDATION_SIZE])
        self.train_data = train_data[VALIDATION_SIZE:, :]
        self.train_labels = to_categorical(train_labels[VALIDATION_SIZE:])

        self.train_attack_data = ori_train_data[ori_train_labels[:, 1] == 1]
        self.train_attack_labels = to_categorical(ori_train_labels[ori_train_labels[:, 1] == 1][:, 0])

        self.test_attack_data = ori_test_data[ori_test_labels[:, 1] == 1]
        self.test_attack_labels = to_categorical(ori_test_labels[ori_test_labels[:, 1] == 1][:, 0])


        self.test_group_data = X_test_group[0:7].reshape((700, 11))
        self.test_group_pred_labels = to_categorical(y_test_group[0:7, :, 0].reshape(-1))
        self.test_group_detc_labels = y_test_group[0:7, :, 1].reshape(-1)


    @staticmethod
    def print():
        return "MNIST"


class MNISTModel:
    def __init__(self, restore, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.model = load_model(restore)

    def predict(self, data):
        return self.model(data)
