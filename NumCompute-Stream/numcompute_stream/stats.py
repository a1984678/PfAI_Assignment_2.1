"""Streaming statistics.

Streaming API (NumCompute-Stream)
---------------------------------
Every class takes 2-D chunks of shape (n_samples, n_features) through
``update_stats(X_chunk)`` and reports one result per feature (axis 0 is reduced).
NaN is skipped per column; infinity raises ValueError; reading a result before any
data raises RuntimeError.

- ``StreamingStats``: count, mean, variance, std, min, max (+ ``merge``), using a
  vectorised Chan/Welford merge, so memory is O(n_features)
- ``StreamingQuantiles``: approximate quantiles from a reservoir sample of k rows
  (exact while at most k rows have been seen)
- ``StreamingHistogram``: fixed-bin histograms per feature, optionally over a sliding
  window of the most recent chunks

Baseline (adapted from NumCompute, Assignment 2.1)
--------------------------------------------------
Batch functions that need all the data at once, kept for non-streaming use:
- Basic descriptive statistics: mean, median, std, min, max
- Histogram
- Quantiles (with NaN handling)
- Axis-wise stats with clear dimension/shape behaviour
- ``RunningStats``: Welford over a single flattened stream (StreamingStats is the
  per-feature version)
"""

import warnings
from collections import deque

import numpy as np

from ._validation import check_array, check_n_features, check_positive_int


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


def _power_of_two_below(x):
    """Largest power of two <= x, elementwise (0 where x is 0). Exact, never overflows.

    np.frexp writes x = m * 2**e with 0.5 <= m < 1, so 2**(e - 1) <= x < 2**e.
    """
    _, exponent = np.frexp(x)
    return np.where(x > 0, np.ldexp(1.0, exponent - 1), 0.0)


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
    - Any magnitude works (1e-300 to 1e308): mean and M2 are stored per column in units
    of a power of two near the largest |value| seen, rescaled exactly when larger values
    arrive, so sums cannot overflow and squares cannot underflow.
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
        self._unit = None

    def _init_state(self, n_features):
        """Create zeroed per-column state for n_features columns."""
        self.n_features = n_features
        self._count = np.zeros(n_features, dtype=np.int64)
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)
        # Per-column power of two that mean and M2 are stored in (0 = only zeros seen).
        self._unit = np.zeros(n_features)

    def _effective_unit(self, unit):
        """The unit to divide by: 1 where no non-zero value has been seen yet."""
        return np.where(unit > 0, unit, 1.0)

    def _rescale_factor(self, old_unit, new_unit):
        """Power-of-two factor converting state stored in old_unit into new_unit (<= 1).

        Where old_unit is 0 only zeros were seen, so the stored state is 0 and any factor
        works; 1 is used so the factor (and its square) can never overflow.
        """
        return np.where(old_unit > 0, old_unit / self._effective_unit(new_unit), 1.0)

    def _grow_unit(self, new_unit):
        """Raise the per-column unit to at least new_unit, rescaling stored mean and M2."""
        new_unit = np.maximum(self._unit, new_unit)
        r = self._rescale_factor(self._unit, new_unit)
        self._mean = self._mean * r
        self._m2 = self._m2 * r * r
        self._unit = new_unit

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

        # Work in units of a power of two near each column's largest |value|, so sums of
        # values near 1e308 cannot overflow and squares of values near 1e-200 cannot
        # underflow. Dividing by a power of two only shifts the exponent: it is exact.
        abs_max = np.where(present, np.abs(X), 0.0).max(axis=0)
        self._grow_unit(_power_of_two_below(abs_max))
        Xs = X / self._effective_unit(self._unit)

        values = np.where(present, Xs, 0.0)
        mean_b = np.divide(values.sum(axis=0), n_b,
                           out=np.zeros(n_features), where=n_b > 0)
        m2_b = (np.where(present, Xs - mean_b, 0.0) ** 2).sum(axis=0)
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
        self._grow_unit(other._unit)
        r = self._rescale_factor(other._unit, self._unit)
        self._combine(other._count, other._mean * r, other._m2 * r * r,
                      other._min, other._max)
        return self

    def _combine(self, n_b, mean_b, m2_b, min_b, max_b):
        """Merge per-column (count, mean, M2, min, max) arrays into the running state.

        mean_b and m2_b must already be in the current unit; min_b and max_b are raw values.
        """
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
        return self._nan_if_empty(self._mean * self._effective_unit(self._unit))

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

    def _scaled_variance(self, ddof):
        """Variance in the stored unit: NaN where count - ddof <= 0, 0 where min == max."""
        if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
            raise ValueError(f"ddof must be a non-negative integer. Got: {ddof!r}")
        self._check_seen()
        denom = self._count - ddof
        valid = denom > 0
        var = np.divide(self._m2, denom, out=np.full(self.n_features, np.nan), where=valid)
        var[valid & (self._min == self._max)] = 0.0
        return var

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
        column's min equals its max. inf where the true variance exceeds the float64
        range (spread above ~1e154); std stays finite there.

        Raises:
        ------
        - ValueError: if ddof is not a non-negative integer
        - RuntimeError: if no data has been seen

        Complexity:
        ----------
        - Time and space complexity: O(d)
        """
        unit = self._effective_unit(self._unit)
        with np.errstate(over="ignore"):  # a variance beyond float64 is reported as inf
            return self._scaled_variance(ddof) * unit * unit

    def std(self, ddof=0):
        """
        Per-feature running standard deviation: sqrt(variance(ddof)).

        Computed as sqrt(scaled variance) * unit, so it stays finite (and accurate) even
        when the variance itself would overflow or underflow.

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
        return np.sqrt(self._scaled_variance(ddof)) * self._effective_unit(self._unit)


class StreamingQuantiles:
    """
    Approximate per-feature quantiles of a stream, using a uniform random sample of rows.

    Keeps a reservoir of at most k rows (reservoir sampling, Algorithm R, vectorised per
    chunk): the i-th row of the stream (0-based) replaces a random reservoir slot with
    probability k / (i + 1), so at any time every row seen so far is equally likely to be
    in the sample. Quantiles are then computed exactly on the sample.

    Parameters:
    ----------
    - k: int >= 1, reservoir size in rows (default 1000); memory is O(k * n_features)
    - seed: int, seed for the random generator (default 0), so results are reproducible

    Attributes:
    ----------
    - n_features: int, or None before the first chunk
    - n_seen: int, rows seen so far
    - sample: np.ndarray (min(n_seen, k), n_features), copy of the current reservoir

    Raises:
    ------
    - ValueError: if k is not a positive integer

    Complexity:
    ----------
    - Time complexity: O(b * d + b log b) per chunk of b rows (the log term from removing
    duplicate slots); O(k * d log k) per quantile call
    - Space complexity: O(k * d)

    Note:
    ----
    - Exact while n_seen <= k (the reservoir then holds every row). Beyond that the result
    is an estimate whose rank error shrinks roughly as 1 / sqrt(k), whatever the data's
    distribution (chosen over a histogram, which fails on heavy-tailed data).
    - Whole rows are sampled, so all columns share one reservoir; NaN is skipped per
    column when computing quantiles. A column with many NaNs therefore has fewer real
    values in the sample and a less accurate estimate.
    """

    def __init__(self, k=1000, seed=0):
        self.k = check_positive_int(k, "k")
        self._rng = np.random.default_rng(seed)
        self.n_features = None
        self.n_seen = 0
        self._buf = None
        self._size = 0  # filled rows of the reservoir, min(n_seen, k)

    def update_stats(self, X_chunk):
        """
        Add one chunk of data to the reservoir.

        Parameters:
        ----------
        - X_chunk: array-like of shape (n_samples, n_features), numeric; NaN allowed

        Returns:
        -------
        - self, so calls can be chained

        Raises:
        ------
        - ValueError: if X_chunk is not numeric, not 2-D, empty, or contains infinity
        - ValueError: if the number of features differs from earlier chunks

        Complexity:
        ----------
        - Time complexity: O(b * d + b log b)
        - Space complexity: O(b * d) temporarily
        """
        X = check_array(X_chunk, name="X", allow_nan=True)
        self.n_features = check_n_features(X, self.n_features)
        if self._buf is None:
            self._buf = np.empty((self.k, self.n_features))

        # Fill phase: while the reservoir has room, every row is kept.
        n_fill = np.minimum(self.k - self._size, X.shape[0])
        self._buf[self._size:self._size + n_fill] = X[:n_fill]
        self._size += n_fill

        # Replacement phase: row with global index i picks a slot j uniformly in [0, i];
        # it is kept (at slot j) only if j < k, i.e. with probability k / (i + 1).
        rest = X[n_fill:]
        if rest.shape[0]:
            index = self.n_seen + n_fill + np.arange(rest.shape[0])
            slot = self._rng.integers(0, index + 1)
            kept = slot < self.k
            slot, rows = slot[kept], rest[kept]
            # Several rows in one chunk can pick the same slot. Processed one at a time, the
            # last of them would win, so keep only each slot's last occurrence (NumPy does
            # not guarantee which write wins for repeated indices in one assignment).
            _, first_in_reversed = np.unique(slot[::-1], return_index=True)
            last = slot.size - 1 - first_in_reversed
            self._buf[slot[last]] = rows[last]

        self.n_seen += X.shape[0]
        return self

    @property
    def sample(self):
        """np.ndarray (min(n_seen, k), n_features): copy of the current reservoir."""
        self._check_seen()
        return self._buf[:self._size].copy()

    def _check_seen(self):
        if self.n_features is None:
            raise RuntimeError("StreamingQuantiles has seen no data yet; "
                               "call update_stats first.")

    def quantile(self, q):
        """
        Per-feature quantiles of the sampled data (linear interpolation, as np.quantile).

        Parameters:
        ----------
        - q: float or array-like of floats in [0, 1]

        Returns:
        -------
        - np.ndarray of shape (n_features,) for scalar q, or (len(q), n_features) for an
        array of q (the layout of np.quantile(X, q, axis=0)). NaN for a column with no
        non-NaN values in the sample.

        Raises:
        ------
        - ValueError: if any q is outside [0, 1] (or NaN)
        - RuntimeError: if no data has been seen

        Complexity:
        ----------
        - Time complexity: O(k * d log k)
        - Space complexity: O(k * d)
        """
        q = np.asarray(q, dtype=np.float64)
        if not np.all((q >= 0) & (q <= 1)):
            raise ValueError(f"q must be in [0, 1]. Got: {q}")
        self._check_seen()

        data = self._buf[:self._size]
        has_values = ~np.isnan(data).all(axis=0)
        out = np.full(q.shape + (self.n_features,), np.nan)
        # All-NaN columns are left as NaN; nanquantile would warn on them.
        out[..., has_values] = np.nanquantile(data[:, has_values], q, axis=0)
        return out


class StreamingHistogram:
    """
    Per-feature histogram over a stream, with an optional sliding window of recent chunks.

    Parameters:
    ----------
    - bins: int >= 1, equal-width bins per feature
    - value_range: (low, high) shared by every feature, or a sequence of one (low, high)
    pair per feature; required, because a range inferred from the data breaks on outliers.
    A pair with low == high is widened to (low - 0.5, high + 0.5), as in ``histogram``.
    - window: int >= 1 or None, number of most recent chunks counted (default None, all)

    Attributes:
    ----------
    - n_features: int, or None before the first chunk
    - counts: np.ndarray of int (n_features, bins), values per bin (within the window)
    - edges: np.ndarray (n_features, bins + 1), bin edges per feature
    - n_out_of_range: np.ndarray of int (n_features,), finite values outside the range
    (within the window), so data drifting out of the chosen range is visible

    Raises:
    ------
    - ValueError: if bins or window is not a positive integer, or value_range is not a
    pair / sequence of pairs of finite numbers with low <= high

    Complexity:
    ----------
    - Time complexity: O(b * d + d * bins) per chunk of b rows and d features
    - Space complexity: O(d * bins) without a window, O(window * d * bins) with one

    Note:
    ----
    - Bins are [edge_i, edge_i+1) except the last, which also includes ``high``, matching
    np.histogram (including its correction for floating-point rounding at the edges).
    - NaN values are skipped: neither binned nor counted as out of range.
    - The window counts chunks, not rows: each chunk's counts are kept and subtracted
    exactly when the chunk leaves the window.
    - Adapted from ``histogram`` (NumCompute, Assignment 2.1), vectorised over features.
    """

    def __init__(self, bins, value_range, window=None):
        self.bins = check_positive_int(bins, "bins")
        self.window = None if window is None else check_positive_int(window, "window")

        try:
            ranges = np.asarray(value_range, dtype=np.float64)
        except (TypeError, ValueError):
            ranges = None
        if ranges is None or ranges.shape[-1:] != (2,) or ranges.ndim > 2:
            raise ValueError("value_range must be a pair (low, high) or a sequence of one "
                             f"pair per feature. Got: {value_range!r}")
        if not np.isfinite(ranges).all():
            raise ValueError("value_range must be finite.")
        low, high = ranges[..., 0], ranges[..., 1]
        if np.any(low > high):
            raise ValueError(f"value_range must satisfy low <= high. Got: {value_range!r}")
        equal = low == high
        self._low = np.where(equal, low - 0.5, low)
        self._high = np.where(equal, high + 0.5, high)
        self._per_feature = ranges.ndim == 2

        self.n_features = None
        self._edges = None
        self._counts = None
        self._out = None
        self._recent = deque()  # (counts, out) per chunk, only used with a window

    def _init_state(self, n_features):
        """Broadcast the range to every feature and build the edges, on the first chunk."""
        if self._per_feature and self._low.shape[0] != n_features:
            raise ValueError(f"value_range has {self._low.shape[0]} pairs but X has "
                             f"{n_features} features.")
        self.n_features = n_features
        self._low = np.broadcast_to(self._low, (n_features,)).copy()
        self._high = np.broadcast_to(self._high, (n_features,)).copy()
        self._edges = np.linspace(self._low, self._high, self.bins + 1, axis=1)
        self._counts = np.zeros((n_features, self.bins), dtype=np.int64)
        self._out = np.zeros(n_features, dtype=np.int64)

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
        - ValueError: if the number of features differs from earlier chunks, or (first
        chunk) from the number of per-feature value_range pairs

        Complexity:
        ----------
        - Time complexity: O(b * d + d * bins)
        - Space complexity: O(b * d) temporarily
        """
        X = check_array(X_chunk, name="X", allow_nan=True)
        n_features = check_n_features(X, self.n_features)
        if self.n_features is None:
            self._init_state(n_features)

        counts, out = self._count_chunk(X)
        if self.window is not None:
            if len(self._recent) == self.window:
                old_counts, old_out = self._recent.popleft()
                self._counts -= old_counts
                self._out -= old_out
            self._recent.append((counts, out))
        self._counts += counts
        self._out += out
        return self

    def _count_chunk(self, X):
        """Return (counts (d, bins), out_of_range (d,)) for one validated chunk."""
        finite = ~np.isnan(X)
        inside = finite & (X >= self._low) & (X <= self._high)
        out = (finite & ~inside).sum(axis=0)

        # Bin index straight from the value; NaN/outside values are replaced by low first,
        # so the float -> int cast never sees NaN (which would warn).
        x = np.where(inside, X, self._low)
        idx = np.floor((x - self._low) / (self._high - self._low) * self.bins)
        idx = np.clip(idx.astype(np.intp), 0, self.bins - 1)

        # Same rounding correction as np.histogram: nudge values that the division put
        # one bin too far left or right, comparing against the actual edges.
        cols = np.arange(self.n_features)
        idx[x < self._edges[cols, idx]] -= 1
        idx[(x >= self._edges[cols, idx + 1]) & (idx != self.bins - 1)] += 1

        # Give each column its own block of bins, then count everything in one bincount.
        flat = idx + cols * self.bins
        counts = np.bincount(flat[inside], minlength=self.n_features * self.bins)
        return counts.reshape(self.n_features, self.bins), out

    def _check_seen(self):
        if self.n_features is None:
            raise RuntimeError("StreamingHistogram has seen no data yet; "
                               "call update_stats first.")

    @property
    def counts(self):
        """np.ndarray of int (n_features, bins): values per bin within the window."""
        self._check_seen()
        return self._counts.copy()

    @property
    def edges(self):
        """np.ndarray (n_features, bins + 1): bin edges per feature."""
        self._check_seen()
        return self._edges.copy()

    @property
    def n_out_of_range(self):
        """np.ndarray of int (n_features,): finite values outside the range (in window)."""
        self._check_seen()
        return self._out.copy()
