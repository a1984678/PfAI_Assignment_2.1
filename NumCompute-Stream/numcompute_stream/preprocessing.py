"""Streaming preprocessing transformers.

- Scalers (e.g., ``StandardScaler``) must support ``.partial_fit()``:

  - Maintain running mean/var (e.g., Welford or EMA)

- ``Imputer`` updates missing-value estimates on the fly
- ``OneHotEncoder`` expands categories incrementally

Baseline (adapted from NumCompute, Assignment 2.1)
--------------------------------------------------
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

import numpy as np

from ._validation import check_array, check_n_features
from .pipeline import Transformer
from .stats import StreamingStats


class StandardScaler(Transformer):
    """
    Standardize features by removing the mean and scaling to unit variance.

    Supports streaming: ``partial_fit`` updates running per-feature mean and variance
    (vectorised Chan/Welford merge in ``stats.StreamingStats``) one chunk at a time,
    without storing the data. ``fit`` is a reset followed by one ``partial_fit``.

    Attributes:
    ----------
    - mean_: array of shape (n_features,), the mean of each feature over all data seen
    (NaN for a feature with no non-NaN values yet, streaming only).
    - scale_: array of shape (n_features,), the population standard deviation (ddof=0) of
    each feature, or 1 for a constant feature or one with no values yet.
    - n_features_in_: int, number of features seen during fit / partial_fit.

    Complexity:
    ----------
    - Time complexity:
        - fit / partial_fit: O(n*m) for n samples (in the chunk) and m features.
        - transform: O(n*m).
    - Space complexity: O(m) state; O(n*m) for the transformed output.

    Note:
    ----
    - NaN values are ignored in the computation of mean and standard deviation, and stay
    NaN after transformation.
    - A constant feature (min == max so far) gets scale 1, so its values map to 0 and not
    0 / 0. Once later chunks add spread, the scale becomes the true standard deviation.
    - fit raises on a column that is entirely NaN; partial_fit allows it (an early chunk of
    a sparse stream may have no values yet) and the column transforms to NaN until a later
    chunk provides data.
    - Infinite values are rejected (they would corrupt the running variance).
    - Adapted from NumCompute (Assignment 2.1). The 2.1 fit also rescaled values near the
    float64 limit (~1e308) to avoid overflow; the streaming version does not.
    """

    supports_partial_fit = True

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit the scaler from scratch, discarding anything learned earlier.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: ignored, accepted for Pipeline compatibility

        Returns:
        -------
        - self: object

        Raises:
        ------
        - ValueError: if the input data is empty or not 2-dimensional
        - ValueError: if any column has all NaN values
        - ValueError: if X contains infinity
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit StandardScaler on an empty array.")

        if np.any(np.all(np.isnan(X), axis=0)):
            raise ValueError("Cannot fit StandardScaler on a column with all NaN values.")

        self._reset()
        return self.partial_fit(X)

    def _reset(self):
        """Forget all streaming state so the next partial_fit starts from scratch."""
        self._stats = None
        for attr in ("mean_", "scale_", "n_features_in_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def partial_fit(self, X, y=None):
        """
        Update the running mean and standard deviation with one chunk of data.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), one chunk; NaN allowed
        - y: ignored, accepted for Pipeline compatibility

        Returns:
        -------
        - self: object

        Raises:
        ------
        - ValueError: if X is not numeric, not 2-D, empty, or contains infinity
        - ValueError: if the number of features differs from earlier chunks

        Complexity:
        ----------
        - Time complexity: O(n*m) for a chunk of n samples and m features
        - Space complexity: O(m) state
        """
        X = check_array(X, name="X", allow_nan=True)
        n_features = check_n_features(X, getattr(self, "n_features_in_", None))
        if getattr(self, "_stats", None) is None:
            self._stats = StreamingStats()
        self._stats.update_stats(X)

        self.n_features_in_ = n_features
        self.mean_ = self._stats.mean
        std = self._stats.std()
        # std is NaN for a column with no values yet and exactly 0 for a constant one;
        # both get scale 1 (the NaN comparison is False, so no warning is raised).
        self.scale_ = np.where(std > 0, std, 1.0)
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
    - In a constant column the range is treated as 1 to avoid division by zero, so these
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
        - ValueError: if feature_range is not a pair of finite numbers with low < hi
        """

        try:
            low, high = (float(v) for v in feature_range)
        except (TypeError, ValueError):
            raise ValueError("feature_range must be a pair of numbers (low, high).")

        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError("feature_range values must be finite.")

        if low >= high:
            raise ValueError("feature range must follow low < hi")

        self.feature_range = (low, high)

    def fit(self, X, y=None):
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

        # max - min overflows to inf for a range like -1e308 to 1e308, so work with half the
        # range. Halving is exact in floating point, so normal data gives the same result.
        half_range = self.max_ / 2 - self.min_ / 2

        # Replace zero ranges before dividing, so no inf is produced (a range of 1 is used)
        is_constant = half_range == 0
        safe_half_range = np.where(is_constant, 0.5, half_range)

        # scale_ is (high - low) / (max - min), written so that max - min is never formed
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / 2 / safe_half_range

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

        # (X - min_) * scale_, with both factors halved and doubled so X - min_ cannot overflow
        return (X / 2 - self.min_ / 2) * (2 * self.scale_) + self.feature_range[0]


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
    - Categories may be numbers or strings. Integer input keeps its integer dtype, so 2**53 and
    2**53 + 1 stay separate categories. A list that mixes integers and floats is kept as an
    object array when converting it to float64 would change any value. Strings stay strings,
    so a string such as "1" is never confused with the number 1. An object array (e.g. the
    output of load_csv with dtype=object) is also accepted.
    - A NumPy float array is used as it is. If large integers were already stored as float64
    before reaching the encoder, values that float64 cannot tell apart become one category.
    - Each column must contain either only numbers or only strings (NaN is allowed in both).
    - NaN values are treated as a separate category.
    - If a feature has the same value for all samples, it will be transformed to a single
    column of ones.
    - The order of the categories in the output will be the same as the order of the unique
    categories found during fit; this is ascending with NaN values placed at the end.
    - The output will be a binary array where each column corresponds to a unique category,
    and the value will be 1 if the sample belongs to that category and 0 otherwise.
    """

    def __init__(self):
        pass

    @staticmethod
    def _as_array(X):
        """
        Convert X to an array without changing the values in a list.

        Numbers keep their own dtype (int stays int, so large integers are not rounded by a
        float64 conversion). A list that mixes integers and floats would become float64, so
        it is kept as an object array when that conversion changes any value (2**53 + 1
        becomes 2**53 in float64). Strings and anything else (a list that mixes strings and
        numbers, a ragged list) become an object array, as NumPy would otherwise turn a list
        such as [1, "a"] into the strings "1" and "a". A NumPy array that is already numeric
        is used as it is.
        """
        try:
            arr = np.asarray(X)
        except ValueError:
            return np.asarray(X, dtype=object)
        if arr.dtype.kind == "f" and not isinstance(X, np.ndarray):
            # Compare each original value with its float64 copy. Python compares int and
            # float exactly, so a rounded integer shows up as a mismatch (NaN is allowed).
            original = np.asarray(X, dtype=object)
            if not np.all((original == arr) | (arr != arr)):
                return original
        if arr.dtype.kind in "biuf" or (arr.dtype.kind == "U" and isinstance(X, np.ndarray)):
            return arr
        return np.asarray(X, dtype=object)

    @staticmethod
    def _unique(col):
        """
        Return the sorted unique values of a column with a single NaN placed at the end.

        Raises:
        ------
        - TypeError: if the column mixes strings and numbers
        """
        # NaN is the only value not equal to itself, this works for both float and object dtypes
        is_nan = col != col
        values = col[~is_nan]

        # np.unique sorts, and sorting a column that mixes strings and numbers raises TypeError.
        # Using that error avoids checking the type of every value in a Python loop.
        try:
            categories = np.unique(values)
        except TypeError:
            raise TypeError("Mixed types in columns, use all numbers or all strings.") from None

        if np.any(is_nan):
            categories = np.append(categories, np.nan)

        return categories

    def fit(self, X, y=None):
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
        - TypeError: if a column mixes strings and numbers
        """
        X = self._as_array(X)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit OneHotEncoder on an empty array.")

        # The number of categories differs between columns, so each column is handled by its own
        # call to _unique, which uses whole-array NumPy calls.
        self.categories_ = [self._unique(X[:, i]) for i in range(X.shape[1])]
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

        X = self._as_array(X)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.shape[1] != len(self.categories_):
            raise ValueError("Incompatible dimensionality.")

        encoded_columns = []
        for i, categories in enumerate(self.categories_):
            col = X[:, i:i+1]
            # x != x is True only for NaN, unlike np.isnan this also works on strings
            encoded = (col == categories) | ((col != col) & (categories != categories))

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

        # fill_value must be a real number (bool is rejected, although Python counts it as an
        # int), not NaN or Inf. np.int64 and np.float64 are accepted, so large numbers work.
        is_number = (isinstance(fill_value, (int, float, np.integer, np.floating))
                     and not isinstance(fill_value, bool))
        if not is_number or np.isnan(fill_value) or np.isinf(fill_value):
            raise ValueError("fill_value must be a numeric type")

        self.fill_value = fill_value

    def fit(self, X, y=None):
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
