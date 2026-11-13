"""Streaming preprocessing transformers.

Every class here follows the ``pipeline.Transformer`` API:

- ``fit(X) -> self``: learn from scratch (discards earlier state)
- ``partial_fit(X_chunk) -> self``: update the learned state with one chunk
- ``transform(X) -> X_out`` and ``fit_transform(X) -> X_out``

Streaming classes (``supports_partial_fit = True``):

- ``StandardScaler``: running mean / variance (vectorised Chan/Welford merge in
  ``stats.StreamingStats``)
- ``MinMaxScaler``: running per-feature min / max
- ``Imputer``: fill values updated on the fly (mean, median or constant)
- ``OneHotEncoder``: categories added incrementally; new categories get new output columns
  at the end, so existing columns never move
- ``SimpleImputer``: constant fill (kept from Assignment 2.1); ``partial_fit`` only checks
  the chunk

``fit`` is a reset followed by one ``partial_fit``, so batch and streaming results share one
code path. All classes are adapted from NumCompute (Assignment 2.1) and vectorised
with NumPy; the only Python loops are over features in ``OneHotEncoder``, whose number of
categories differs per feature.
"""

import numpy as np

from ._validation import check_array, check_n_features, check_positive_int
from .pipeline import Transformer
from .stats import StreamingQuantiles, StreamingStats


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
    - Values up to the float64 limit (~1e308) are handled without overflow, also when they
    arrive in different chunks: StreamingStats stores each feature in units of a power of
    two, which replaces the rescaling in the 2.1 fit.
    - Adapted from NumCompute (Assignment 2.1).
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

    Supports streaming: ``partial_fit`` keeps a running per-feature min and max (NaN-aware
    np.fmin / np.fmax, no stored data). ``fit`` is a reset followed by one ``partial_fit``.

    Parameters:
    ----------
    - feature_range: tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    Attributes:
    ----------
    - min_: array of shape (n_features,), the minimum of each feature over all data seen
    (NaN for a feature with no non-NaN values yet, streaming only).
    - max_: array of shape (n_features,), the maximum of each feature over all data seen.
    - scale_: array of shape (n_features,), the scaling factor for each feature.
    - n_features_in_: int, number of features seen during fit / partial_fit.

    Complexity:
    ----------
    - Time complexity:
        - fit: O(n*m) where n is the number of samples and m is the number of features.
        - transform: O(n*m) where n is the number of samples and m is the number of features.
    - Space complexity: O(m) state; O(n*m) for the transformed output.

    Note:
    ----
    - NaN values are ignored in the computation of min and max.
    - fit raises on a column that is entirely NaN; partial_fit allows it and the column
    transforms to NaN until a later chunk provides data (as StandardScaler).
    - In a constant column the range is treated as 1 to avoid division by zero, so these
    training values map to feature_range[0].
    - NaN values in the input data will remain NaN in the output data after transformation.
    - There is no clipping of values outside the feature_range; values can exceed the specified
    range if they are outside the min and max of the training data.
    - Adapted from NumCompute (Assignment 2.1), including its overflow-safe half-range
    arithmetic, which now runs after every chunk.
    """

    supports_partial_fit = True

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
            raise ValueError("Cannot fit MinMaxScaler on an empty array.")

        if np.any(np.all(np.isnan(X), axis=0)):
            raise ValueError("Cannot fit MinMaxScaler on a column with all NaN values.")

        self._reset()
        return self.partial_fit(X)

    def _reset(self):
        """Forget all streaming state so the next partial_fit starts from scratch."""
        for attr in ("min_", "max_", "scale_", "n_features_in_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def partial_fit(self, X, y=None):
        """
        Update the running min and max with one chunk of data.

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

        # fmin / fmax ignore NaN (NaN only where both sides are NaN) and never warn, unlike
        # np.nanmin on an all-NaN column.
        chunk_min = np.fmin.reduce(X, axis=0)
        chunk_max = np.fmax.reduce(X, axis=0)
        if hasattr(self, "min_"):
            chunk_min = np.fmin(self.min_, chunk_min)
            chunk_max = np.fmax(self.max_, chunk_max)
        self.min_, self.max_ = chunk_min, chunk_max
        self.n_features_in_ = n_features
        self._update_scale()
        return self

    def _update_scale(self):
        """Recompute scale_ from min_ and max_ (overflow-safe, constant columns -> range 1)."""
        # max - min overflows to inf for a range like -1e308 to 1e308, so work with half the
        # range. Halving is exact in floating point, so normal data gives the same result.
        half_range = self.max_ / 2 - self.min_ / 2

        # Replace zero ranges before dividing, so no inf is produced (a range of 1 is used).
        # A column with no values yet has a NaN range, so its scale_ (and output) is NaN.
        is_constant = half_range == 0
        safe_half_range = np.where(is_constant, 0.5, half_range)

        # scale_ is (high - low) / (max - min), written so that max - min is never formed
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / 2 / safe_half_range

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

    Supports streaming: ``partial_fit`` adds categories as they first appear. Output columns
    never move: the first chunk gives the 2.1 layout (one block per feature, categories
    sorted, NaN last) and every category discovered later gets a new column at the end of
    the output, so a column index always means the same (feature, category). The output
    width therefore grows; rows encoded earlier only need zeros appended on the right.

    Parameters:
    ----------
    - handle_unknown: "error" (default, as in 2.1) raises on a category not seen yet;
    "ignore" encodes it as all zeros in that feature's columns. "ignore" suits streams,
    where a chunk is usually scored before the encoder has been updated with it.

    Attributes:
    ----------
    - categories_: list of arrays, one per feature, holding its categories in the order they
    were discovered (sorted within the first chunk, and within each later batch of new ones).
    - columns_: list of (feature_index, category) tuples, one per output column, in output
    order.
    - n_features_in_: int, number of input features.

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
    - After fit (or the first partial_fit) the categories of each feature are ascending with
    NaN placed at the end. Categories found by later partial_fit calls follow, at the end of
    the output.
    - A column must not switch between numbers and strings across chunks (TypeError).
    - The output will be a binary array where each column corresponds to a unique category,
    and the value will be 1 if the sample belongs to that category and 0 otherwise.
    """

    supports_partial_fit = True

    def __init__(self, handle_unknown="error"):
        """
        Raises:
        ------
        - ValueError: if handle_unknown is not "error" or "ignore"
        """
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'. "
                             f"Got: {handle_unknown!r}")
        self.handle_unknown = handle_unknown

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
        Fit the encoder from scratch by finding the unique categories for each feature.

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
        - TypeError: if a column mixes strings and numbers
        """
        X = self._as_array(X)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit OneHotEncoder on an empty array.")

        self._reset()
        return self.partial_fit(X)

    def _reset(self):
        """Forget all categories so the next partial_fit starts from scratch."""
        for attr in ("categories_", "columns_", "_positions", "n_features_in_"):
            if hasattr(self, attr):
                delattr(self, attr)

    @staticmethod
    def _matches(values, categories):
        """
        Boolean matrix of shape (len(values), len(categories)): values[i] equals categories[j].

        x != x is True only for NaN, so NaN matches NaN; unlike np.isnan this also works on
        strings and object arrays.
        """
        values = values.reshape(-1, 1)
        both_nan = (values != values) & (categories != categories)
        return (values == categories) | both_nan

    def partial_fit(self, X, y=None):
        """
        Add the categories first seen in this chunk.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), numbers or strings per column
        - y: ignored, accepted for Pipeline compatibility

        Returns:
        -------
        - self: object

        Raises:
        ------
        - ValueError: if X is not 2-D or empty, or its feature count differs from earlier
        chunks
        - TypeError: if a column mixes strings and numbers, within the chunk or compared with
        the categories already seen

        A rejected chunk leaves the encoder unchanged (all features are checked first).

        Complexity:
        ----------
        - Time complexity: O(m * (n log n + u * k)) for n samples, m features, u unique values
        in the chunk and k known categories per feature
        - Space complexity: O(total categories)
        """
        X = self._as_array(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional (n_samples, n_features). "
                             f"Got shape: {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"X must have at least one row and one column. "
                             f"Got shape: {X.shape}")
        n_features = check_n_features(X, getattr(self, "n_features_in_", None))

        if not hasattr(self, "categories_"):
            # First chunk: the 2.1 layout, one block of sorted categories per feature. The
            # number of categories differs between columns, so each column is handled by
            # its own call to _unique, which uses whole-array NumPy calls.
            self.categories_ = [self._unique(X[:, i]) for i in range(n_features)]
            self.columns_, self._positions = [], []
            for i, categories in enumerate(self.categories_):
                start = len(self.columns_)
                self.columns_.extend((i, c) for c in categories.tolist())
                self._positions.append(np.arange(start, len(self.columns_)))
            self.n_features_in_ = n_features
            return self

        # Pass 1: find every feature's new categories, so a TypeError in any feature is
        # raised before anything changes (a rejected chunk leaves the encoder as it was).
        new_categories = []
        for known, col in zip(self.categories_, X.T):
            seen = self._unique(col)
            # Sorting old and new categories together raises TypeError if one is a string
            # and the other a number, so a column cannot switch type between chunks.
            self._unique(np.concatenate([known.astype(object), seen.astype(object)]))
            new_categories.append(seen[~self._matches(seen, known).any(axis=1)])

        # Pass 2: append the new categories and give them output columns at the end.
        for i, (known, new) in enumerate(zip(self.categories_, new_categories)):
            if new.size == 0:
                continue
            start = len(self.columns_)
            self.categories_[i] = np.concatenate([known, new])
            self.columns_.extend((i, c) for c in new.tolist())
            self._positions[i] = np.concatenate(
                [self._positions[i], np.arange(start, len(self.columns_))])
        return self

    def transform(self, X):
        """
        Transform the data into a one-hot encoded array based on the fitted categories.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: float array of shape (n_samples, len(columns_)) with one-hot encoding
        applied; column j is 1 where the row's feature columns_[j][0] equals category
        columns_[j][1]

        Raises:
        ------
        - RuntimeError: if fit has not been called before transform
        - ValueError: if the input data is not 2-dimensional or has a different number
        of features than the fitted data
        - ValueError: if any category in the input data was not seen during fit (only when
        handle_unknown="error")
        """
        if not hasattr(self, 'categories_'):
            raise RuntimeError("fit must be called before transform.")

        X = self._as_array(X)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.shape[1] != len(self.categories_):
            raise ValueError("Incompatible dimensionality.")

        out = np.zeros((X.shape[0], len(self.columns_)))
        for i, categories in enumerate(self.categories_):
            encoded = self._matches(X[:, i], categories)

            # check for unseen categories in the input data
            if self.handle_unknown == "error" and not np.all(np.any(encoded, axis=1)):
                raise ValueError("Found unknown categories during transform")

            # this feature's categories may be spread over the output (later ones at the end)
            out[:, self._positions[i]] = encoded

        return out


def _check_fill_value(fill_value):
    """
    Check an imputer fill value is a finite real number.

    Parameters:
    ----------
    - fill_value: scalar

    Returns:
    -------
    - fill_value, unchanged

    Raises:
    ------
    - ValueError: if fill_value is not a number (bool is rejected, although Python counts it
    as an int), or is NaN or infinite
    """
    # np.int64 and np.float64 are accepted, so large numbers work.
    is_number = (isinstance(fill_value, (int, float, np.integer, np.floating))
                 and not isinstance(fill_value, bool))
    if not is_number or np.isnan(fill_value) or np.isinf(fill_value):
        raise ValueError("fill_value must be a numeric type")
    return fill_value


class SimpleImputer(Transformer):
    """
    Impute missing values (NaN) with a constant value.

    Supports streaming: the fill value never changes, so ``partial_fit`` only validates the
    chunk and fixes the number of features. ``fit`` is a reset followed by one
    ``partial_fit``. ``Imputer`` offers fill values learned from the data.

    Parameters:
    ----------
    - fill_value: scalar, default=0
        The value to replace missing values with.

    Attributes:
    ----------
    - columns_: int, the number of features in the training set (2.1 name).
    - n_features_in_: int, the same number (name shared with the other transformers).

    Complexity:
    ----------
    - Time complexity: O(n*m) where n is the number of samples and m is the number of features.
    - Space complexity: O(n*m) for storing the transformed data in memory.

    Note:
    ----
    - All NaN column is allowed and will be filled with the fill_value.
    - Infinite values are rejected when fitting, as in the other transformers.
    """

    supports_partial_fit = True

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

        self.fill_value = _check_fill_value(fill_value)

    def fit(self, X, y=None):
        """
        Fit the imputer from scratch. For SimpleImputer, this just checks the input data and
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
        - ValueError: if X contains infinity
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit SimpleImputer on an empty array.")

        self._reset()
        return self.partial_fit(X)

    def _reset(self):
        """Forget the feature count so the next partial_fit starts from scratch."""
        for attr in ("columns_", "n_features_in_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def partial_fit(self, X, y=None):
        """
        Check one chunk of data; the first chunk sets the number of features.

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
        - Time complexity: O(n*m) for the validation of a chunk of n samples and m features
        - Space complexity: O(1) state
        """
        X = check_array(X, name="X", allow_nan=True)
        self.n_features_in_ = check_n_features(X, getattr(self, "n_features_in_", None))
        self.columns_ = self.n_features_in_
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


class Imputer(Transformer):
    """
    Replace missing values (NaN) with per-feature estimates updated on the fly.

    Streaming counterpart of SimpleImputer: every ``partial_fit`` updates the running
    estimate of each feature, and ``transform`` fills NaN with the current estimates.

    Parameters:
    ----------
    - strategy: str, default="mean"
        - "mean": running mean per feature (``stats.StreamingStats``, exact)
        - "median": median of a reservoir sample of rows (``stats.StreamingQuantiles``;
        exact while at most k rows have been seen, an estimate after that). Robust to
        outliers.
        - "constant": always fill_value
    - fill_value: finite number, default=0. Used by "constant", and by the other strategies
    for a feature with no non-NaN values seen yet.
    - k: int >= 1, reservoir size for "median" (default 1000)
    - seed: int, random seed for "median" (default 0)

    Attributes:
    ----------
    - statistics_: array of shape (n_features,), the current fill value for each feature
    - n_features_in_: int, number of features seen during fit / partial_fit

    Raises:
    ------
    - ValueError: if strategy is unknown, fill_value is not a finite number, or k is not a
    positive integer

    Complexity:
    ----------
    - Time complexity: O(n*m) per partial_fit / transform for n samples and m features
    ("median" adds O(k*m log k) per partial_fit to recompute the medians)
    - Space complexity: O(m) for "mean" / "constant", O(k*m) for "median"

    Note:
    ----
    - A feature with no values yet is filled with fill_value rather than left as NaN:
    removing NaN is the imputer's job, and later steps (e.g. the tree) need complete data.
    - fit is a reset followed by one partial_fit; like SimpleImputer it allows all-NaN
    columns.
    """

    supports_partial_fit = True
    _STRATEGIES = ("mean", "median", "constant")

    def __init__(self, strategy="mean", fill_value=0, k=1000, seed=0):
        if strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}. Got: {strategy!r}")
        self.strategy = strategy
        self.fill_value = _check_fill_value(fill_value)
        self.k = check_positive_int(k, "k")
        self.seed = seed

    def fit(self, X, y=None):
        """
        Fit the imputer from scratch, discarding anything learned earlier.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features); NaN allowed
        - y: ignored, accepted for Pipeline compatibility

        Returns:
        -------
        - self: object

        Raises:
        ------
        - ValueError: if the input data is empty or not 2-dimensional
        - ValueError: if X contains infinity
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.size == 0:
            raise ValueError("Cannot fit Imputer on an empty array.")

        self._reset()
        return self.partial_fit(X)

    def _reset(self):
        """Forget all streaming state so the next partial_fit starts from scratch."""
        self._stats = None
        for attr in ("statistics_", "n_features_in_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def partial_fit(self, X, y=None):
        """
        Update the per-feature fill estimates with one chunk of data.

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
        """
        X = check_array(X, name="X", allow_nan=True)
        n_features = check_n_features(X, getattr(self, "n_features_in_", None))

        if getattr(self, "_stats", None) is None and self.strategy != "constant":
            self._stats = (StreamingStats() if self.strategy == "mean"
                           else StreamingQuantiles(k=self.k, seed=self.seed))

        if self.strategy == "mean":
            estimate = self._stats.update_stats(X).mean
        elif self.strategy == "median":
            estimate = self._stats.update_stats(X).quantile(0.5)
        else:
            estimate = np.full(n_features, np.nan)

        # NaN means "no values seen for this feature yet" (or strategy "constant").
        self.statistics_ = np.where(np.isnan(estimate), float(self.fill_value), estimate)
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        """
        Replace NaN in each feature with that feature's current estimate.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: array of shape (n_samples, n_features) without NaN; other values unchanged

        Raises:
        ------
        - RuntimeError: if fit / partial_fit has not been called before transform
        - ValueError: if the input data is not 2-dimensional or has a different number of
        features than the fitted data
        """
        if not hasattr(self, "statistics_"):
            raise RuntimeError("fit must be called before transform.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Incompatible dimensionality.")

        return np.where(np.isnan(X), self.statistics_, X)
