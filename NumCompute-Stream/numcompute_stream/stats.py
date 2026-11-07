"""Streaming statistics.

- All statistical functions redesigned for streaming
- Implement chunk-based versions of:

  - Mean, variance, quantiles
  - Histograms (optionally with sliding window)

- Add ``update_stats(X_chunk)`` API

Baseline (adapted from NumCompute, Assignment 2.1)
--------------------------------------------------
- Basic descriptive statistics:
  - Mean, median, standard deviation, min, max
- Histogram
- Quantiles (with NaN handling)
- Axis-wise stats with clear dimension/shape behaviour
- Streaming statistics (Welford's algorithm) with RunningStats
"""

import warnings

import numpy as np

from ._validation import check_array, check_n_features


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _as_float_array(x):
    """
    Convert input to a float64 array after checking it is numeric and non-empty.

    Parameters:
    ----------
    - x: array-like of any shape

    Returns:
    -------
    - np.ndarray of dtype float64 with the same shape as x

    Raises:
    ------
    - ValueError: if x is not numeric or is empty

    Complexity:
    ----------
    - Time complexity: O(n) where n is the number of elements
    - Space complexity: O(n) if a conversion is needed, O(1) otherwise
    """
    x = np.asarray(x)
    if x.dtype.kind not in "biuf":
        raise ValueError("Input must be numeric. Got dtype: {}".format(x.dtype))
    if x.size == 0:
        raise ValueError("Input must not be empty.")
    return x.astype(np.float64, copy=False)


def _check_axis(axis, ndim):
    """
    Validate an axis argument.

    Parameters:
    ----------
    - axis: None or int (negative values count from the end)
    - ndim: int, number of dimensions of the data

    Returns:
    -------
    - None or a non-negative int

    Raises:
    ------
    - ValueError: if axis is not None/int or is out of bounds

    Complexity:
    ----------
    - Time and space complexity: O(1)
    """
    if axis is None:
        return None
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be None or an integer. Got: {!r}".format(axis))
    if not -ndim <= axis < ndim:
        raise ValueError("axis {} is out of bounds for input with {} dimension(s)."
                         .format(axis, ndim))
    return int(axis) % ndim


def _finish(result):
    """Return a Python float for 0-d results, the array otherwise."""
    return float(result) if np.ndim(result) == 0 else result


def _reduce(nan_func, plain_func, x, axis, skipna, **kwargs):
    """
    Run a NumPy reduction with shared validation and NaN handling.

    Parameters:
    ----------
    - nan_func: NaN-ignoring NumPy function (e.g. np.nanmean)
    - plain_func: NaN-propagating NumPy function (e.g. np.mean)
    - x, axis, skipna: as in the public functions
    - kwargs: extra keyword arguments passed to the NumPy function

    Returns:
    -------
    - float (axis=None) or np.ndarray (x.shape with `axis` removed)

    Complexity:
    ----------
    - Time complexity: O(n)
    - Space complexity: O(n / size of the reduced axis) for the output
    """
    x = _as_float_array(x)
    axis = _check_axis(axis, x.ndim)
    func = nan_func if skipna else plain_func

    # all-NaN slices give NaN on purpose; silence NumPy's warning about them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = func(x, axis=axis, **kwargs)

    return _finish(result)


# ---------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------

def mean(x, axis=None, skipna=True):
    """
    Arithmetic mean.

    Parameters:
    ----------
    - x: array-like of any shape, numeric
    - axis: None (whole array) or int, the axis to average over
    - skipna: bool, if True (default) NaN values are ignored; if False any NaN
    in a slice makes that result NaN

    Returns:
    -------
    - float if axis is None, otherwise np.ndarray of shape x.shape with `axis` removed
    (e.g. shape (n_samples, n_features) with axis=0 gives (n_features,))

    Raises:
    ------
    - ValueError: if x is empty or non-numeric, or axis is invalid

    Complexity:
    ----------
    - Time complexity: O(n) where n is the number of elements
    - Space complexity: O(size of the output)

    Note:
    ----
    - A slice that is entirely NaN gives NaN (no warning is raised).
    """
    return _reduce(np.nanmean, np.mean, x, axis, skipna)


def std(x, axis=None, ddof=0, skipna=True):
    """
    Standard deviation.

    Parameters:
    ----------
    - x: array-like of any shape, numeric
    - axis: None or int, the axis to reduce over
    - ddof: int >= 0, delta degrees of freedom; the divisor is (N - ddof).
    0 (default) gives the population std, 1 the sample std
    - skipna: bool, if True (default) NaN values are ignored

    Returns:
    -------
    - float if axis is None, otherwise np.ndarray of shape x.shape with `axis` removed

    Raises:
    ------
    - ValueError: if x is empty or non-numeric, axis is invalid, or ddof is not a
    non-negative integer

    Complexity:
    ----------
    - Time complexity: O(n)
    - Space complexity: O(n) for the deviations from the mean

    Note:
    ----
    - Uses NumPy's two-pass method (mean first, then squared deviations), which avoids
    the cancellation error of the one-pass E[x^2] - E[x]^2 formula.
    - If a slice has N - ddof <= 0 valid values the result for that slice is NaN.
    """
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer. Got: {!r}".format(ddof))

    result = _reduce(np.nanstd, np.std, x, axis, skipna, ddof=ddof)

    # make "not enough values" consistently NaN (NumPy can give inf or nan here)
    x = _as_float_array(x)
    counts = np.sum(~np.isnan(x), axis=_check_axis(axis, x.ndim)) if skipna else \
        np.full(np.shape(result), x.size if axis is None else x.shape[axis])
    result = np.where(counts - ddof <= 0, np.nan, result)
    return _finish(result)


def min(x, axis=None, skipna=True):
    """
    Minimum value.

    Parameters:
    ----------
    - x: array-like of any shape, numeric
    - axis: None or int, the axis to reduce over
    - skipna: bool, if True (default) NaN values are ignored

    Returns:
    -------
    - float if axis is None, otherwise np.ndarray of shape x.shape with `axis` removed

    Raises:
    ------
    - ValueError: if x is empty or non-numeric, or axis is invalid

    Complexity:
    ----------
    - Time complexity: O(n)
    - Space complexity: O(size of the output)

    Note:
    ----
    - Named to match the spec. Inside this module the built-in min is not used.
    """
    return _reduce(np.nanmin, np.min, x, axis, skipna)


def max(x, axis=None, skipna=True):
    """
    Maximum value.

    Parameters:
    ----------
    - x: array-like of any shape, numeric
    - axis: None or int, the axis to reduce over
    - skipna: bool, if True (default) NaN values are ignored

    Returns:
    -------
    - float if axis is None, otherwise np.ndarray of shape x.shape with `axis` removed

    Raises:
    ------
    - ValueError: if x is empty or non-numeric, or axis is invalid

    Complexity:
    ----------
    - Time complexity: O(n)
    - Space complexity: O(size of the output)

    Note:
    ----
    - Named to match the spec. Inside this module the built-in max is not used.
    """
    return _reduce(np.nanmax, np.max, x, axis, skipna)


def quantile(x, q, axis=None, skipna=True):
    """
    Quantile(s) with linear interpolation, computed with a vectorised sort.

    Parameters:
    ----------
    - x: array-like of any shape, numeric
    - q: float or 1D array-like of floats, each in [0, 1] (0.5 is the median)
    - axis: None (flatten) or int, the axis to compute along
    - skipna: bool, if True (default) NaN values are ignored; if False any NaN in a
    slice makes that result NaN

    Returns:
    -------
    - scalar q: float if axis is None, else np.ndarray of x.shape with `axis` removed
    - 1D q of length k: np.ndarray with an extra leading axis of length k,
    i.e. shape (k,) or (k, *x.shape without axis) (same convention as NumPy)

    Raises:
    ------
    - ValueError: if x is empty or non-numeric, axis is invalid, or q is not in [0, 1]

    Complexity:
    ----------
    - Time complexity: O(n log m) where m is the length of the axis (sorting each slice)
    - Space complexity: O(n) for the sorted copy

    Note:
    ----
     - For a slice with N valid values the position is q * (N - 1); results between two
    sorted values are linearly interpolated, written so that huge values (e.g. -1e308 and
    1e308) do not overflow.
    - NaN values sort to the end, so counting the valid values tells us where to look.
    - A slice that is entirely NaN gives NaN.
    """
    x = _as_float_array(x)
    axis = _check_axis(axis, x.ndim)

    try:
        q_arr = np.asarray(q, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("q must be a number or a 1D array of numbers. Got: {!r}".format(q))
    if q_arr.ndim > 1 or q_arr.size == 0:
        raise ValueError("q must be a number or a non-empty 1D array.")
    if np.isnan(q_arr).any() or (q_arr < 0).any() or (q_arr > 1).any():
        raise ValueError("q values must be between 0 and 1.")

    # put the reduced axis last: data has shape (..., m)
    data = x.reshape(1, -1) if axis is None else np.moveaxis(x, axis, -1)
    sorted_data = np.sort(data, axis=-1)
    m = sorted_data.shape[-1]

    n_valid = np.sum(~np.isnan(sorted_data), axis=-1, keepdims=True)  # (..., 1)
    pos = q_arr.ravel() * (n_valid - 1)                               # (..., k)
    lower_idx = np.clip(np.floor(pos).astype(np.intp), 0, m - 1)
    upper_idx = np.clip(np.ceil(pos).astype(np.intp), 0, m - 1)

    lower = np.take_along_axis(sorted_data, lower_idx, axis=-1)
    upper = np.take_along_axis(sorted_data, upper_idx, axis=-1)
    frac = pos - np.floor(pos)
    # lower * (1 - frac) + upper * frac, not lower + frac * (upper - lower): the subtraction
    # overflows to inf for values like -1e308 and 1e308, even though the answer is finite
    result = np.where(upper == lower, lower, lower * (1.0 - frac) + upper * frac)

    result = np.where(n_valid == 0, np.nan, result)
    if not skipna:
        result = np.where(n_valid < m, np.nan, result)

    # (..., k) -> (k, ...) like NumPy, then tidy the shape
    result = np.moveaxis(result, -1, 0)
    if axis is None:
        result = result[:, 0]
    if q_arr.ndim == 0:
        result = result[0]
    return _finish(result)


def median(x, axis=None, skipna=True):
    """
    Median, i.e. the 0.5 quantile (reuses quantile).

    Parameters:
    ----------
    - x: array-like of any shape, numeric
    - axis: None or int, the axis to reduce over
    - skipna: bool, if True (default) NaN values are ignored

    Returns:
    -------
    - float if axis is None, otherwise np.ndarray of shape x.shape with `axis` removed

    Raises:
    ------
    - ValueError: same conditions as quantile

    Complexity:
    ----------
    - Time complexity: O(n log m) (see quantile)
    - Space complexity: O(n)
    """
    return quantile(x, 0.5, axis=axis, skipna=skipna)


def histogram(x, bins=10, value_range=None):
    """
    Count how many values fall into each of `bins` equal-width bins.

    Parameters:
    ----------
    - x: array-like of any shape, numeric (flattened)
    - bins: int >= 1, number of bins (default 10)
    - value_range: (low, high) or None. Values outside it are ignored.
    Defaults to (min of x, max of x)

    Returns:
    -------
    - counts: np.ndarray of shape (bins,), integer counts per bin
    - edges: np.ndarray of shape (bins + 1,), the bin edges

    Raises:
    ------
    - ValueError: if x is empty or non-numeric, bins is not a positive integer,
    value_range is invalid, or the range cannot be inferred (all NaN or infinite values)

    Complexity:
    ----------
    - Time complexity: O(n + bins)
    - Space complexity: O(n + bins)

    Note:
    ----
    - Bins are [edge_i, edge_i+1) except the last, which also includes its right edge.
    - NaN values are ignored.
    - If all values are equal the range is widened to (value - 0.5, value + 0.5),
    the same convention as NumPy.
    - Vectorised: each value's bin index is computed directly with floor and counted
    with np.bincount, then nudged by one where floating-point rounding put a value
    just across an edge.
    """
    x = _as_float_array(x).ravel()
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer. Got: {!r}".format(bins))

    x = x[~np.isnan(x)]

    if value_range is None:
        if x.size == 0:
            raise ValueError("Cannot infer a range: every value is NaN.")
        low, high = x.min(), x.max()
    else:
        try:
            low, high = (float(v) for v in value_range)
        except (TypeError, ValueError):
            raise ValueError("value_range must be a pair of numbers (low, high).")

    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError("The histogram range must be finite. Pass a finite value_range.")
    if low > high:
        raise ValueError("value_range must satisfy low <= high. Got ({}, {})".format(low, high))
    if low == high:
        low, high = low - 0.5, high + 0.5

    edges = np.linspace(low, high, bins + 1)
    inside = x[(x >= low) & (x <= high)]

    idx = np.floor((inside - low) / (high - low) * bins).astype(np.intp)
    idx = np.clip(idx, 0, bins - 1)
    # correct values that rounding placed one bin too far right or left
    idx[inside < edges[idx]] -= 1
    idx[(inside >= edges[idx + 1]) & (idx != bins - 1)] += 1

    counts = np.bincount(idx, minlength=bins)
    return counts, edges


# ---------------------------------------------------------------------
# Streaming statistics
# ---------------------------------------------------------------------

class RunningStats:
    """
    Streaming count, mean, variance, std, min and max over data that arrives in
    chunks (e.g. a CSV read piece by piece), without storing the data.

    Uses Welford's algorithm, generalised to whole batches (Chan et al.'s parallel
    update): each batch's mean and squared deviations are computed with NumPy, then
    merged into the running totals. For a batch of one value this is exactly
    Welford's update.

    Attributes:
    ----------
    - count: int, number of (non-NaN) values seen
    - mean: float, running mean (NaN before any data)
    - min, max: float, running extremes (NaN before any data)

    Complexity:
    ----------
    - Time complexity: O(b) per update for a batch of b values
    - Space complexity: O(1) state (O(b) temporarily per batch)

    Note:
    ----
    - All input is flattened into one stream; keep one RunningStats per column
    for per-feature statistics.
    - NaN values are skipped; infinite values raise ValueError.
    - Welford avoids the cancellation error of summing x and x^2 separately, so the
    variance stays accurate for large values with a small spread.
    """

    def __init__(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0  # sum of squared deviations from the mean
        self._min = np.inf
        self._max = -np.inf

    def update(self, values):
        """
        Add a batch of values (scalar or array of any shape).

        Parameters:
        ----------
        - values: number or array-like, numeric

        Returns:
        -------
        - self, so calls can be chained

        Raises:
        ------
        - ValueError: if values is non-numeric or contains infinite values

        Complexity:
        ----------
        - Time complexity: O(b) for b values
        - Space complexity: O(b)
        """
        batch = np.asarray(values)
        if batch.dtype.kind not in "biuf":
            raise ValueError("values must be numeric. Got dtype: {}".format(batch.dtype))
        batch = batch.astype(np.float64).ravel()
        batch = batch[~np.isnan(batch)]
        if not np.isfinite(batch).all():
            raise ValueError("values must be finite (NaN values are skipped).")
        if batch.size == 0:
            return self

        batch_mean = batch.mean()
        batch_m2 = np.sum((batch - batch_mean) ** 2)
        self._combine(batch.size, batch_mean, batch_m2, batch.min(), batch.max())
        return self

    def merge(self, other):
        """
        Combine another RunningStats into this one (e.g. results from two workers).

        Parameters:
        ----------
        - other: RunningStats

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if other is not a RunningStats

        Complexity:
        ----------
        - Time and space complexity: O(1)
        """
        if not isinstance(other, RunningStats):
            raise ValueError("Can only merge another RunningStats.")
        if other._count > 0:
            self._combine(other._count, other._mean, other._m2, other._min, other._max)
        return self

    def _combine(self, n_b, mean_b, m2_b, min_b, max_b):
        """Merge (count, mean, M2, min, max) of a batch into the running state."""
        n_a = self._count
        total = n_a + n_b
        delta = mean_b - self._mean
        self._mean = self._mean + delta * n_b / total
        self._m2 = self._m2 + m2_b + delta * delta * n_a * n_b / total
        self._count = total
        self._min = float(np.minimum(self._min, min_b))
        self._max = float(np.maximum(self._max, max_b))

    @property
    def count(self):
        """int: number of non-NaN values seen."""
        return self._count

    @property
    def mean(self):
        """float: running mean, NaN before any data."""
        return float(self._mean) if self._count else float("nan")

    @property
    def min(self):
        """float: smallest value seen, NaN before any data."""
        return self._min if self._count else float("nan")

    @property
    def max(self):
        """float: largest value seen, NaN before any data."""
        return self._max if self._count else float("nan")

    def variance(self, ddof=0):
        """
        Running variance.

        Parameters:
        ----------
        - ddof: int >= 0, the divisor is (count - ddof); 0 (default) population,
        1 sample variance

        Returns:
        -------
        - float, NaN if count - ddof <= 0

        Raises:
        ------
        - ValueError: if ddof is not a non-negative integer

        Complexity:
        ----------
        - Time and space complexity: O(1)
        """
        if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
            raise ValueError("ddof must be a non-negative integer. Got: {!r}".format(ddof))
        if self._count - ddof <= 0:
            return float("nan")
        return float(self._m2 / (self._count - ddof))

    def std(self, ddof=0):
        """
        Running standard deviation: the square root of variance(ddof).

        Parameters:
        ----------
        - ddof: int >= 0 (see variance)

        Returns:
        -------
        - float, NaN if count - ddof <= 0

        Complexity:
        ----------
        - Time and space complexity: O(1)
        """
        return float(np.sqrt(self.variance(ddof)))


class StreamingStats:
    """
    Per-feature streaming count, mean, variance, std, min and max.

    Feed 2-D chunks of shape (n_samples, n_features) with ``update_stats``; every result
    is an array of shape (n_features,), one value per column (axis 0 is reduced). The data
    itself is never stored.

    Algorithm: each chunk is summarised per column as (count, mean, M2, min, max), where
    M2 is the sum of squared deviations from the chunk mean, then merged into the running
    totals with Chan et al.'s parallel update (the batch form of Welford's algorithm),
    vectorised across columns:

        delta = mean_b - mean_a
        mean  = mean_a + delta * n_b / n
        M2    = M2_a + M2_b + delta**2 * n_a * n_b / n

    Attributes:
    ----------
    - n_features: int, or None before the first chunk
    - count: np.ndarray of int (n_features,), non-NaN values seen per column
    - mean, min, max: np.ndarray of float (n_features,), NaN for a column with no values yet

    Raises:
    ------
    - RuntimeError: reading count/mean/min/max/variance/std before any data has been seen

    Complexity:
    ----------
    - Time complexity: O(b * d) per chunk of b rows and d features
    - Space complexity: O(d) state; O(b * d) temporarily per chunk

    Note:
    ----
    - NaN values are skipped per column, so columns can have different counts.
    - Infinite values raise ValueError (they would turn the running M2 into NaN).
    - A column whose min equals its max reports a variance of exactly 0.0, so rounding in
    the running mean cannot leave a tiny non-zero variance for constant data.
    - Adapted from RunningStats (NumCompute, Assignment 2.1), which flattens all input
    into a single stream.
    """

    def __init__(self):
        self.n_features = None
        self._count = None
        self._mean = None
        self._m2 = None
        self._min = None
        self._max = None

    def _init_state(self, n_features):
        """Create zeroed per-column state for n_features columns."""
        self.n_features = n_features
        self._count = np.zeros(n_features, dtype=np.int64)
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)

    def update_stats(self, X_chunk):
        """
        Add one chunk of data.

        Parameters:
        ----------
        - X_chunk: array-like of shape (n_samples, n_features), numeric; NaN is skipped

        Returns:
        -------
        - self, so calls can be chained

        Raises:
        ------
        - ValueError: if X_chunk is not numeric, not 2-D, empty, or contains infinity
        - ValueError: if the number of features differs from earlier chunks

        Complexity:
        ----------
        - Time complexity: O(b * d)
        - Space complexity: O(b * d) for the masked temporaries
        """
        X = check_array(X_chunk, name="X", allow_nan=True)
        n_features = check_n_features(X, self.n_features)
        if self.n_features is None:
            self._init_state(n_features)

        present = ~np.isnan(X)
        n_b = present.sum(axis=0)
        values = np.where(present, X, 0.0)
        mean_b = np.divide(values.sum(axis=0), n_b,
                           out=np.zeros(n_features), where=n_b > 0)
        m2_b = (np.where(present, X - mean_b, 0.0) ** 2).sum(axis=0)
        min_b = np.where(present, X, np.inf).min(axis=0)
        max_b = np.where(present, X, -np.inf).max(axis=0)

        self._combine(n_b, mean_b, m2_b, min_b, max_b)
        return self

    def merge(self, other):
        """
        Combine another StreamingStats into this one.

        Parameters:
        ----------
        - other: StreamingStats with the same n_features (or one that has seen no data)

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if other is not a StreamingStats, or n_features differ

        Complexity:
        ----------
        - Time and space complexity: O(d)
        """
        if not isinstance(other, StreamingStats):
            raise ValueError("Can only merge another StreamingStats.")
        if other.n_features is None:
            return self
        if self.n_features is None:
            self._init_state(other.n_features)
        elif other.n_features != self.n_features:
            raise ValueError(f"Cannot merge stats with {other.n_features} features into "
                             f"stats with {self.n_features} features.")
        self._combine(other._count, other._mean, other._m2, other._min, other._max)
        return self

    def _combine(self, n_b, mean_b, m2_b, min_b, max_b):
        """Merge per-column (count, mean, M2, min, max) arrays into the running state."""
        n_a = self._count
        total = n_a + n_b
        # n_b / total, with 0 where a column has still seen nothing (avoids 0/0).
        frac_b = np.divide(n_b, total, out=np.zeros(self.n_features), where=total > 0)
        delta = mean_b - self._mean
        self._mean = self._mean + delta * frac_b
        self._m2 = self._m2 + m2_b + delta ** 2 * n_a * frac_b
        self._count = total
        self._min = np.minimum(self._min, min_b)
        self._max = np.maximum(self._max, max_b)

    def _check_seen(self):
        if self.n_features is None:
            raise RuntimeError("StreamingStats has seen no data yet; call update_stats first.")

    def _nan_if_empty(self, values):
        return np.where(self._count > 0, values, np.nan)

    @property
    def count(self):
        """np.ndarray of int (n_features,): non-NaN values seen per column."""
        self._check_seen()
        return self._count.copy()

    @property
    def mean(self):
        """np.ndarray (n_features,): running mean, NaN for columns with no values."""
        self._check_seen()
        return self._nan_if_empty(self._mean)

    @property
    def min(self):
        """np.ndarray (n_features,): smallest value seen, NaN for columns with no values."""
        self._check_seen()
        return self._nan_if_empty(self._min)

    @property
    def max(self):
        """np.ndarray (n_features,): largest value seen, NaN for columns with no values."""
        self._check_seen()
        return self._nan_if_empty(self._max)

    def variance(self, ddof=0):
        """
        Per-feature running variance.

        Parameters:
        ----------
        - ddof: int >= 0, the divisor is (count - ddof); 0 (default) population,
        1 sample variance

        Returns:
        -------
        - np.ndarray (n_features,), NaN where count - ddof <= 0, exactly 0.0 where a
        column's min equals its max

        Raises:
        ------
        - ValueError: if ddof is not a non-negative integer
        - RuntimeError: if no data has been seen

        Complexity:
        ----------
        - Time and space complexity: O(d)
        """
        if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
            raise ValueError(f"ddof must be a non-negative integer. Got: {ddof!r}")
        self._check_seen()
        denom = self._count - ddof
        valid = denom > 0
        var = np.divide(self._m2, denom, out=np.full(self.n_features, np.nan), where=valid)
        var[valid & (self._min == self._max)] = 0.0
        return var

    def std(self, ddof=0):
        """
        Per-feature running standard deviation: sqrt(variance(ddof)).

        Parameters:
        ----------
        - ddof: int >= 0 (see variance)

        Returns:
        -------
        - np.ndarray (n_features,), NaN where count - ddof <= 0

        Complexity:
        ----------
        - Time and space complexity: O(d)
        """
        return np.sqrt(self.variance(ddof))


class StreamingQuantiles:
    """
    Approximate per-feature quantiles of a stream, using a uniform random sample of rows.

    Stub: not implemented yet (tests in tests/test_stats.py define the behaviour).

    Parameters:
    ----------
    - k: int >= 1, reservoir size in rows (default 1000); memory is O(k * n_features)
    - seed: int, seed for the random generator (default 0)

    Attributes:
    ----------
    - n_features: int, or None before the first chunk
    - n_seen: int, rows seen so far
    - sample: np.ndarray (min(n_seen, k), n_features), copy of the current reservoir
    """

    def __init__(self, k=1000, seed=0):
        raise NotImplementedError("StreamingQuantiles is not implemented yet")

    def update_stats(self, X_chunk):
        """Add one chunk of shape (n_samples, n_features); returns self."""
        raise NotImplementedError("StreamingQuantiles is not implemented yet")

    def quantile(self, q):
        """Per-feature quantiles: shape (n_features,) for scalar q, (len(q), n_features)."""
        raise NotImplementedError("StreamingQuantiles is not implemented yet")
