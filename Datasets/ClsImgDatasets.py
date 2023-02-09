import tensorflow as tf
from keras import datasets
from abc import ABC


# Base Class
class BaseCLSImgDatasets(ABC):
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def get_dataset(self):
        return self.x_train, self.y_train, self.x_test, self.y_test


class MNISTDataset(BaseCLSImgDatasets):
    def __init__(self):
        super().__init__()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()


class CIFAR10Dataset(BaseCLSImgDatasets):
    def __init__(self):
        super().__init__()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()


class CIFAR100Dataset(BaseCLSImgDatasets):
    def __init__(self):
        super().__init__()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()


class IMDBDataset(BaseCLSImgDatasets):
    # no img
    def __init__(self):
        super().__init__()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.imdb.load_data()


class ReutresDataset(BaseCLSImgDatasets):
    # no img
    def __init__(self):
        super().__init__()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.reuters.load_data()
