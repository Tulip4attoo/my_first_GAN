import os
import numpy as np
from os import walk, getcwd
from tensorflow.keras.datasets import mnist


def load_safari(np_file):
    array = np.load(np_file)
    new_array = np.reshape(array, (-1, 28, 28, 1))
    return new_array


def load_zeros_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    new_x_train = x_train[y_train == 0]
    new_x_train = np.reshape(new_x_train, (-1, 28, 28, 1))
    return new_x_train


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    new_x_train = np.reshape(x_train, (-1, 28, 28, 1))
    return new_x_train
