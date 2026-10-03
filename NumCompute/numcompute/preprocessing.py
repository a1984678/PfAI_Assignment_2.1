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
    """
    Standardize features by removing the mean and scaling to unit variance.

    Attributes:
    ----------
    - mean_: array of shape (n_features,), the mean value for each feature in the training set.
    - scale_: array of shape (n_features,), the standard deviation for each feature in
    training set or 1 for constant columns.

    Complexity:
    ----------
    - Time complexity:
        - fit: O(n*m) where n is the number of samples and m is the number of features.
        - transform: O(n*m) where n is the number of samples and m is the number of features.
    - Space complexity: O(n*m) for storing the transformed data in memory.

    Note:
    ----
    - NaN values are ignored in the computation of mean and standard deviation.
    - in a constant column (where all values are the same), the standard deviation is set to 1
    to avoid division by zero, so these training values map to 0 after transformation.
    - Uses population standard deviation (ddof=0) for scaling.
    - NaN values in the input data will remain NaN in the output data after transformation.
    """

    def __init__(self):
        pass

    def fit(self, X):
        """
        Fit the scaler on the data by computing the mean and standard deviation for each feature.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - self: object

        Raises:
        ------
        - ValueError: if the input data is empty or not 2-dimensional
        - ValueError: if any column has all NaN values
        """

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
        """
        Transform the data by standardizing each feature to have mean 0 and variance 1.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: array of shape (n_samples, n_features) with standardized features

        Raises:
        ------
        - RuntimeError: if fit has not been called before transform
        - ValueError: if the input data is not 2-dimensional or has a different number
        of features than the fitted data
        """
        if not hasattr(self, 'mean_'):
            raise RuntimeError("fit must be called before transform.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("Incompatible dimensionality.")

        return (X - self.mean_) / self.scale_


class MinMaxScaler(Transformer):
    """
    Transform features by scaling each feature to a given range.

    Parameters:
    ----------
    - feature_range: tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    Attributes:
    ----------
    - min_: array of shape (n_features,), the minimum value for each feature in the training set.
    - max_: array of shape (n_features,), the maximum value for each feature in the training set.
    - scale_: array of shape (n_features,), the scaling factor for each feature in the training set.

    Complexity:
    ----------
    - Time complexity:
        - fit: O(n*m) where n is the number of samples and m is the number of features.
        - transform: O(n*m) where n is the number of samples and m is the number of features.
    - Space complexity: O(n*m) for storing the transformed data in memory.

    Note:
    ----
    - NaN values are ignored in the computation of min and max.
    - In a constanct column the range is treated as 1 to avoid division by zero, so these
    training values map to feature_range[0].
    - NaN values in the input data will remain NaN in the output data after transformation.
    - There is no clipping of values outside the feature_range; values can exceed the specified
    range if they are outside the min and max of the training data.
    """

    def __init__(self, feature_range=(0, 1)):
        """
        Initialize the MinMaxScaler with the desired feature range.

        Parameters:
        ----------
        - feature_range: tuple (min, max), default=(0, 1)
            Desired range of transformed data.

        Raises:
        ------
        - ValueError: if feature_range low >= hi
        """

        if (feature_range[0] >= feature_range[1]):
            raise ValueError("feature range must follow low < hi")

        self.feature_range = feature_range

    def fit(self, X):
        """
        Fit the scaler on the data by computing the min and max for each feature.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - self: object

        Raises:
        ------
        - ValueError: if the input data is empty or not 2-dimensional
        - ValueError: if any column has all NaN values
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit MinMaxScaler on an empty array.")

        if np.any(np.all(np.isnan(X), axis=0)):
            raise ValueError("Cannot fit MinMaxScaler on a column with all NaN values.")

        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        self.range_ = self.max_ - self.min_

        # Replace zero ranges before dividing, so no inf is produced
        is_constant = self.range_ == 0
        safe_range = np.where(is_constant, 1.0, self.range_)

        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / safe_range

        return self

    def transform(self, X):
        """
        Transform the data by scaling each feature to the specified range.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: array of shape (n_samples, n_features) with features scaled to the specified range

        Raises:
        ------
        - RuntimeError: if fit has not been called before transform
        - ValueError: if the input data is not 2-dimensional or has a different number
        of features than the fitted data
        """
        if not hasattr(self, 'scale_'):
            raise RuntimeError("fit must be called before transform.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.shape[1] != self.min_.shape[0]:
            raise ValueError("Incompatible dimensionality.")

        return (X - self.min_) * self.scale_ + self.feature_range[0]


class OneHotEncoder(Transformer):
    """
    Encode categorical features as a one-hot numeric array.

    Attributes:
    ----------
    - categories_: list of arrays, each array contains the unique categories for a feature.

    Complexity:
    ----------
    - Time complexity:
        - fit: O(m * n log n) where n is the number of samples and m is the number of features.
        - transform: O(n*m*k) where n is the number of samples, m is the number of features,
        and k is the average number of unique categories per feature.
    - Space complexity: O(n*m*k) for storing the transformed data in memory.

    Note:
    ----
    - NaN values are treated as a separate category.
    - If a feature has the same value for all samples, it will be transformed to a single
    column of ones.
    - The order of the categories in the output will be the same as the order of the unique
    categories found during fit this is ascending with Nan values placed at the end.
    - The output will be a binary array where each column corresponds to a unique category,
    and the value will be 1 if the sample belongs to that category and 0 otherwise.
    """

    def __init__(self):
        pass

    def fit(self, X):
        """
        Fit the encoder on the data by finding the unique categories for each feature.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - self: object

        Raises:
        ------
        - ValueError: if the input data is empty or not 2-dimensional
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit OneHotEncoder on an empty array.")

        # Using a loop here is acceptable since we need to find unique categories for each
        # column
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        """
        Transform the data into a one-hot encoded array based on the fitted categories.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: array of shape (n_samples, n_encoded_features) with one-hot encoding applied

        Raises:
        ------
        - RuntimeError: if fit has not been called before transform
        - ValueError: if the input data is not 2-dimensional or has a different number
        of features than the fitted data
        - ValueError: if any category in the input data was not seen during fit
        """
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
    """
    Impute missing values (NaN) with a constant value.

    Parameters:
    ----------
    - fill_value: scalar, default=0
        The value to replace missing values with.

    Attributes:
    ----------
    - columns_: int, the number of features in the training set.

    Complexity:
    ----------
    - Time complexity: O(n*m) where n is the number of samples and m is the number of features.
    - Space complexity: O(n*m) for storing the transformed data in memory.

    Note:
    ----
    - All NaN column is allowed and will be filled with the fill_value.
    """

    def __init__(self, fill_value=0):
        """
        Initialize the SimpleImputer with the specified fill_value.

        Parameters:
        ----------
        - fill_value: scalar, default=0
            The value to replace missing values with.

        Raises:
        ------
        - ValueError: if fill_value is not a numeric type or is NaN or Inf
        """

        # Validate fill_value to ensure it's a numeric type and not NaN or Inf and accept large
        # numbers e.g. np.int64, np.float64
        if (not isinstance(fill_value, (int, float, np.integer, np.floating))
            or np.isnan(fill_value)
            or np.isinf(fill_value)
            ):

            raise ValueError("fill_value must be a numeric type")

        self.fill_value = fill_value

    def fit(self, X):
        """
        Fit the imputer on the data. For SimpleImputer, this just checks the input data and
        stores the number of features.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - self: object

        Raises:
        ------
        - ValueError: if the input data is empty or not 2-dimensional

        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit SimpleImputer on an empty array.")

        self.columns_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Does a find and replace of NaN values with the specified fill_value.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: array of shape (n_samples, n_features) with NaN values replaced by fill_value

        Raises:
        ------
        - RuntimeError: if fit has not been called before transform
        - ValueError: if the input data is not 2-dimensional or has a different number of features
        than the fitted data
        """
        if not hasattr(self, 'columns_'):
            raise RuntimeError("fit must be called before transform.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.shape[1] != self.columns_:
            raise ValueError("Incompatible dimensionality.")

        return np.where(np.isnan(X), self.fill_value, X)
