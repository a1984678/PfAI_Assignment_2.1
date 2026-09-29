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
        if X.size == 0:
            raise ValueError("Cannot fit StandardScaler on an empty array.")
        if X.ndim != 2:
            raise ValueError("Input data must be at least 2 dimensions. Got shape: {}".format(X.shape))
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ <= 1e-8] = 1  # Avoid division by 0 and very small numbers

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.size == 0:
            return np.empty((0, self.mean_.shape[0]))  # Return empty array with correct shape
        if X.ndim != 2:
            raise ValueError("Input data must be at least 2 dimensions. Got shape: {}".format(X.shape))
        if not hasattr(self, 'mean_'):
            raise RuntimeError("fit must be called before transform.")
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("Incompatible dimensionality.")
        # if np.any(self.scale_ == 0):
        #     return np.full_like(X, 0)  # Return all 0s for constant columns
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