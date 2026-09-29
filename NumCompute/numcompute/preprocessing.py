"""
preprocessing.py

Implement:
- StandardScaler   (z-score standardization)
- MinMaxScaler     (scale to a given range)
- OneHotEncoder    (for categorical variables)
- (Optional) Simple imputer to replace NaN with a constant

API:
- fit(X) -> self
- transform(X) -> X_out
- fit_transform(X) -> X_out

Fully vectorized using NumPy; avoid Python loops.
"""

from .pipeline import Transformer
import numpy as np

class StandardScaler(Transformer):
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self):
        pass

    def fit(self, X):

        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_
    

class MinMaxScaler(Transformer):
    """Transform features by scaling each feature to a given range."""

    def __init__(self, feature_range=(0, 1)):
        raise NotImplementedError

class OneHotEncoder(Transformer):
    """Encode categorical features as a one-hot numeric array."""

    def __init__(self):
        raise NotImplementedError

class SimpleImputer(Transformer):
    """Impute missing values (NaN) with a constant value."""

    def __init__(self, fill_value=0):
        raise NotImplementedError