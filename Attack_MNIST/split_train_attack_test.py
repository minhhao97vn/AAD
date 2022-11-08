from art.utils import load_mnist
import numpy as np
from sklearn.model_selection import train_test_split


def load_mnist_train_attack_test():
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    x_train_n, x_attack, y_train_n, y_attack = train_test_split(x_train, y_train, train_size=44 / 60, random_state=8)

    np.savez('mnist_all.npz', X_train=x_train_n, y_train=y_train_n, X_test=x_test, y_test=y_test, X_attack=x_attack,
             y_attack=y_attack)

    return (x_train_n, y_train_n), (x_test, y_test), (x_attack, y_attack), min_pixel_value, max_pixel_value