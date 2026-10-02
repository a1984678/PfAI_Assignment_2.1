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

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit StandardScaler on an empty array.")

        # Check if entire columns are NaN and raise an error if so
        if np.any(np.all(np.isnan(X), axis=0)):
            raise ValueError("Cannot fit StandardScaler on a column with all NaN values.")

        # Handle NaN values by using the nanmean and nanstd functions, which ignore NaN values in
        # the computation
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)

        # Handle constant columns by setting scale to 1 to avoid division by zero
        is_constant = (np.nanmax(X, axis=0) - np.nanmin(X, axis=0)) == 0
        self.scale_[is_constant] = 1.0  

        return self

    def transform(self, X):

        if not hasattr(self, 'mean_'):
            raise RuntimeError("fit must be called before transform.")        

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("Incompatible dimensionality.")        

        return (X - self.mean_) / self.scale_


class MinMaxScaler(Transformer):
    """Transform features by scaling each feature to a given range."""

    def __init__(self, feature_range=(0, 1)):

        if (feature_range[0] >= feature_range[1]):
            raise ValueError("feature range must follow low < hi")

        self.feature_range = feature_range

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit MinMaxScaler on an empty array.")

        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        self.range_ = self.max_ - self.min_

        # Replace zero ranges before dividing, so no inf is produced
        is_constant = self.range_ == 0
        safe_range = np.where(is_constant, 1.0, self.range_)

        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / safe_range

        return self

    def transform(self, X):

        if not hasattr(self, 'scale_'):
            raise RuntimeError("fit must be called before transform.") 

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))        

        if X.shape[1] != self.min_.shape[0]:
            raise ValueError("Incompatible dimensionality.")  

        return (X - self.min_) * self.scale_ + self.feature_range[0]


class OneHotEncoder(Transformer):
    """Encode categorical features as a one-hot numeric array."""

    def __init__(self):
        pass

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit OneHotEncoder on an empty array.")

        # Using a loop here is acceptable i think since we need to find unique categories for each 
        # column
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):

        if not hasattr(self, 'categories_'):
            raise RuntimeError("fit must be called before transform.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.shape[1] != len(self.categories_):
            raise ValueError("Incompatible dimensionality.")

        encoded_columns = []
        for i, categories in enumerate(self.categories_):
            col = X[:, i:i+1]
            encoded = np.equal(col, categories) | (np.isnan(col) & np.isnan(categories))

            # check for unseen categories in the input data
            if not np.all(np.any(encoded, axis=1)):
                raise ValueError("Found unknown categories during transform")

            encoded_columns.append(encoded)

        return np.hstack(encoded_columns).astype(np.float64)


class SimpleImputer(Transformer):
    """Impute missing values (NaN) with a constant value."""

    def __init__(self, fill_value=0):

        # Validate fill_value to ensure it's a numeric type and not NaN or Inf and accept large 
        # numbers e.g. np.int64, np.float64
        if (not isinstance(fill_value, (int, float, np.integer, np.floating))
            or np.isnan(fill_value)
            or np.isinf(fill_value)
            ):

            raise ValueError("fill_value must be a numeric type")

        self.fill_value = fill_value

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit SimpleImputer on an empty array.")

        self.columns_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, 'columns_'):
            raise RuntimeError("fit must be called before transform.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.shape[1] != self.columns_:
            raise ValueError("Incompatible dimensionality.")

        return np.where(np.isnan(X), self.fill_value, X)
