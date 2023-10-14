import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SimpleScaler(BaseEstimator, TransformerMixin):

    def __init__(self, scale=100):
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X / self.scale

    def inverse_transform(self, X):
        return X * self.scale


class OnMaxAbsValueScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scale = 1

    def fit(self, X, y=None):
        self.scale = np.max(np.abs(X), axis=0)
        return self

    def transform(self, X):
        return X / self.scale

    def inverse_transform(self, X):
        return X * self.scale

