"""Shared input validation for streaming components (private).

Every streaming ``partial_fit`` / ``transform`` / ``predict`` receives one chunk at a
time, so the same checks run on every chunk:

- ``check_array``: coerce a chunk to a 2-D float64 array (NaN optionally allowed).
- ``check_X_y``: coerce X as above and check y is 1-D with one label per row.
- ``check_n_features``: the number of columns must not change between chunks.
- ``check_positive_int``: integer hyperparameters such as ``chunk_size``.

All functions raise ``ValueError`` with the argument name in the message.
"""

import numpy as np

_NUMERIC_KINDS = "biuf"


def check_array(X, name="X", allow_nan=False):
    """
    Convert one chunk of data to a 2-D float64 array.

    Parameters:
    ----------
    - X: array-like of shape (n_samples, n_features), numeric
    - name: str, argument name used in error messages (default "X")
    - allow_nan: bool, if True NaN is kept (missing values); infinity is always rejected

    Returns:
    -------
    - np.ndarray of dtype float64 and shape (n_samples, n_features). A new array is made
    only when X is not already float64.

    Raises:
    ------
    - ValueError: if X is not numeric, not 2-D, has no rows or no columns, contains
    infinity, or contains NaN while allow_nan is False

    Complexity:
    ----------
    - Time complexity: O(n_samples * n_features) for the finiteness check
    - Space complexity: O(n_samples * n_features) if a dtype conversion is needed
    """
    arr = np.asarray(X)
    if arr.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"{name} must be numeric. Got dtype: {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional (n_samples, n_features). "
                         f"Got shape: {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column. "
                         f"Got shape: {arr.shape}")
    arr = arr.astype(np.float64, copy=False)

    if np.isinf(arr).any():
        raise ValueError(f"{name} must not contain infinity.")
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(f"{name} must not contain NaN.")
    return arr


def check_X_y(X, y, allow_nan=False):
    """
    Validate a chunk of features and its labels together.

    Parameters:
    ----------
    - X: array-like of shape (n_samples, n_features), numeric
    - y: array-like of shape (n_samples,), labels of any dtype (numbers or strings)
    - allow_nan: bool, passed to check_array for X (default False)

    Returns:
    -------
    - X: np.ndarray of dtype float64 and shape (n_samples, n_features)
    - y: np.ndarray of shape (n_samples,), dtype unchanged

    Raises:
    ------
    - ValueError: anything check_array raises for X
    - ValueError: if y is not 1-D or its length differs from the number of rows in X

    Complexity:
    ----------
    - Time complexity: O(n_samples * n_features)
    - Space complexity: as check_array
    """
    X = check_array(X, name="X", allow_nan=allow_nan)
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional (n_samples,). Got shape: {y.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. "
                         f"Got {X.shape[0]} and {y.shape[0]}.")
    return X, y


def check_n_features(X, expected):
    """
    Check a chunk has the same number of features as earlier chunks.

    Typical use inside partial_fit, where the first chunk sets the count:

        self.n_features_in_ = check_n_features(X, getattr(self, "n_features_in_", None))

    Parameters:
    ----------
    - X: np.ndarray of shape (n_samples, n_features), already validated
    - expected: int or None, the feature count seen so far (None for the first chunk)

    Returns:
    -------
    - int, the number of features in X

    Raises:
    ------
    - ValueError: if expected is not None and X.shape[1] != expected

    Complexity:
    ----------
    - Time and space complexity: O(1)
    """
    n_features = X.shape[1]
    if expected is not None and n_features != expected:
        raise ValueError(f"X has {n_features} features, but {expected} features were "
                         "seen in earlier chunks.")
    return n_features


def check_positive_int(value, name):
    """
    Check an integer hyperparameter is at least 1.

    Parameters:
    ----------
    - value: int-like, e.g. chunk_size or buffer_size
    - name: str, argument name used in the error message

    Returns:
    -------
    - int, the value as a plain Python int

    Raises:
    ------
    - ValueError: if value is a bool, not an integer, or less than 1

    Complexity:
    ----------
    - Time and space complexity: O(1)
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer. Got: {value!r}")
    return int(value)
