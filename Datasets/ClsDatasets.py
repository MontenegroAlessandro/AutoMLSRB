# This Python module is thought to extract datasets for Classification tasks

# Libraries
from abc import ABC, abstractmethod
from sklearn import datasets
import pandas as pd
import numpy as np


class BaseCLSData(ABC):
    def __init__(self):
        self.input = None
        self.target = None
        self.full_dataset = None

    def get_dataset(self):
        return self.input, self.target, self.full_dataset


class IrisDataset(BaseCLSData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.load_iris()
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']


class WineDataset(BaseCLSData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.load_wine()
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']


class DigitsDataset(BaseCLSData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.load_digits()
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']


class BreastCancerDataset(BaseCLSData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.load_breast_cancer()
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']


class CreditGDataset(BaseCLSData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.fetch_openml(name="credit-g", as_frame=False, parser="auto", version="active")
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']


class SteelPlatesFaultDataset(BaseCLSData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.fetch_openml(name="steel-plates-fault", as_frame=False, parser="auto",
                                                  version="active")
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']

