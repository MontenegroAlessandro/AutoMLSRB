# This Python module is thought to extract datasets for Regression tasks

# Libraries
from abc import ABC, abstractmethod
from sklearn import datasets
import pandas as pd
import numpy as np


class BaseRegData(ABC):
    def __init__(self):
        self.input = None
        self.target = None
        self.full_dataset = None

    def get_dataset(self):
        return self.input, self.target, self.full_dataset


class DiabetesDataset(BaseRegData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.load_diabetes()
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']


class LinnerudDataset(BaseRegData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.load_linnerud()
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']


class BostonDataset(BaseRegData):
    def __init__(self):
        super().__init__()
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        self.full_dataset = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        self.data = np.hstack([self.full_dataset.values[::2, :], self.full_dataset.values[1::2, :2]])
        self.target = self.full_dataset.values[1::2, 2]


class CaliforniaDataset(BaseRegData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.fetch_california_housing()
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']


class AmesDataset(BaseRegData):
    def __init__(self):
        super().__init__()
        self.full_dataset = datasets.fetch_openml(name="house_prices", as_frame=True)
        self.input = self.full_dataset['data']
        self.target = self.full_dataset['target']