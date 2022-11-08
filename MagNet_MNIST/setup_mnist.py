

import numpy as np
import os
import gzip
import urllib.request

from keras.models import load_model


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images * 28 * 28)
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
                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/" + name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000) + 0.5
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000) + 0.5
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)

        VALIDATION_SIZE = 5000

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

    @staticmethod
    def print():
        return "MNIST"


class AADMNIST:
    def __init__(self):
        train_data_load = np.load('data/AAD_MNIST_Comb_1_train.npz')
        test_data_load = np.load('data/AAD_MNIST_Comb_1_test.npz')

        x_train = train_data_load['X_all'].reshape((-1, 28, 28, 1))
        y_train = train_data_load['y_all']
        x_test = test_data_load['X_all'].reshape((-1, 28, 28, 1))
        y_test = test_data_load['y_all']

        train_data = x_train[y_train[:, -1] == 0]
        train_labels = y_train[y_train[:, -1] == 0]
        self.test_data = x_test[y_test[:, -1] == 0]
        self.test_labels = y_test[y_test[:, -1] == 0][:, 0:-1]

        VALIDATION_SIZE = 4000

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE][:, 0:-1]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:][:, 0:-1]

        self.test_attack_data = x_test[y_test[:, -1] == 1]
        self.test_attack_labels = y_test[y_test[:, -1] == 1][:, 0:-1]

        self.test_group_data = x_test
        self.test_group_pred_labels = y_test[:, 0:-1]
        self.test_group_detc_labels = y_test[:, -1]

    @staticmethod
    def print():
        return "AAD-MNIST"


class MNISTModel:
    def __init__(self, restore, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.model = load_model(restore)

    def predict(self, data):
        return self.model(data)
