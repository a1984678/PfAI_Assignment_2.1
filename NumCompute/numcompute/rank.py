"""
rank.py

- rank(data, method='average'|'dense'|'ordinal') — handle ties
- percentile(data, q, interpolation='linear'|'lower'|'higher'|'midpoint')
"""

import numpy as np

_RANK_METHODS = ("average", "dense", "ordinal")
_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint")


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

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


def _to_last_axis(arr, axis):
    """
    Put the axis to work on last (flattening first if axis is None).

    Parameters:
    ----------
    - arr: np.ndarray
    - axis: None or a validated non-negative int

    Returns:
    -------
    - np.ndarray view whose last axis is the one to work along

    Complexity:
    ----------
    - Time and space complexity: O(1) (a view), O(n) if flattening needs a copy
    """
    if axis is None:
        return arr.reshape(-1)
    return np.moveaxis(arr, axis, -1)


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------

def rank(data, method="average", axis=None):
    """
    Rank values from 1 (smallest) upwards, with a choice of how to rank ties.

    Parameters:
    ----------
    - data: array-like of any shape (numbers or strings)
    - method: str, how tied values are ranked:
      - 'average' (default): each tie gets the mean of the ranks it covers,
      e.g. [10, 20, 20, 30] -> [1, 2.5, 2.5, 4]
      - 'dense': ties share a rank and the next value gets the next integer,
      e.g. [10, 20, 20, 30] -> [1, 2, 2, 3]
      - 'ordinal': every value gets a distinct rank, ties in order of appearance,
      e.g. [10, 20, 20, 30] -> [1, 2, 3, 4]
    - axis: None (default, rank the flattened array) or int (rank along that axis,
    each slice separately)

    Returns:
    -------
    - ranks: np.ndarray. Shape (n,) if axis is None, otherwise the shape of data.
    dtype float64 for 'average', int64 for 'dense' and 'ordinal'

    Raises:
    ------
    - ValueError: if data is empty, cannot be ordered, contains NaN, or method/axis
    is invalid

    Complexity:
    ----------
    - Time complexity: O(n log m) where m is the length of the ranked axis (one sort)
    - Space complexity: O(n)

    Note:
    ----
    - One stable argsort per slice; everything after it is vectorised:
      - a value starts a new tie group if it differs from the one before it (sorted)
      - dense rank = running count of group starts (cumsum)
      - average rank = midpoint of the first and last position of its group, found
      with a running maximum (group start) and a reversed running minimum (group end)
      - ranks are written back to the original positions with np.put_along_axis
    - NaN is rejected because it has no place in an order; remove or fill it first.
    """
    if method not in _RANK_METHODS:
        raise ValueError("method must be 'average', 'dense' or 'ordinal'. Got: {!r}"
                         .format(method))
    arr = np.asarray(data)
    if arr.dtype.kind not in "biufUS":
        raise ValueError("data must be numbers or strings. Got dtype: {}".format(arr.dtype))
    if arr.size == 0:
        raise ValueError("Input must not be empty.")
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError("data must not contain NaN (it cannot be ranked).")
    axis = _check_axis(axis, arr.ndim)

    work = _to_last_axis(arr, axis)                     # shape (..., m)
    m = work.shape[-1]
    order = np.argsort(work, axis=-1, kind="stable")
    sorted_vals = np.take_along_axis(work, order, axis=-1)

    pos = np.broadcast_to(np.arange(m), work.shape)     # 0..m-1 in sorted order
    starts = np.ones(work.shape, dtype=bool)
    starts[..., 1:] = sorted_vals[..., 1:] != sorted_vals[..., :-1]

    if method == "ordinal":
        sorted_ranks = pos + 1
    elif method == "dense":
        sorted_ranks = np.cumsum(starts, axis=-1)
    else:
        ends = np.ones(work.shape, dtype=bool)
        ends[..., :-1] = starts[..., 1:]
        group_first = np.maximum.accumulate(np.where(starts, pos, 0), axis=-1)
        group_last = np.flip(np.minimum.accumulate(
            np.flip(np.where(ends, pos, m), axis=-1), axis=-1), axis=-1)
        sorted_ranks = (group_first + group_last) / 2.0 + 1.0

    dtype = np.float64 if method == "average" else np.int64
    ranks = np.empty(work.shape, dtype=dtype)
    np.put_along_axis(ranks, order, sorted_ranks, axis=-1)

    if axis is None:
        return ranks
    return np.moveaxis(ranks, -1, axis)


# ---------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------

def percentile(data, q, interpolation="linear", axis=None):
    """
    The q-th percentile(s) of the data, ignoring NaN.

    Parameters:
    ----------
    - data: array-like of any shape, numeric
    - q: float or 1D array-like of floats, each in [0, 100] (50 is the median)
    - interpolation: str, what to return when the percentile falls between two sorted
    values a (below) and b (above), at fraction f of the way from a to b:
      - 'linear' (default): a + f * (b - a)
      - 'lower': a
      - 'higher': b
      - 'midpoint': (a + b) / 2
    - axis: None (default, flatten) or int, the axis to compute along

    Returns:
    -------
    - scalar q: float if axis is None, else np.ndarray of data.shape with `axis` removed
    - 1D q of length k: np.ndarray with an extra leading axis of length k,
    i.e. shape (k,) or (k, *data.shape without axis) (same convention as NumPy)

    Raises:
    ------
    - ValueError: if data is empty or non-numeric, q is not in [0, 100],
    or interpolation/axis is invalid

    Complexity:
    ----------
    - Time complexity: O(n log m) where m is the length of the axis (one sort per slice)
    - Space complexity: O(n) for the sorted copy

    Note:
    ----
    - For a slice with N valid (non-NaN) values the position is q / 100 * (N - 1).
    a and b are the sorted values at floor and ceil of that position.
    - NaN values sort to the end, so counting them per slice tells us where the valid
    values stop. A slice that is entirely NaN gives NaN.
    - Matches np.nanpercentile with the same method name.
    """
    if interpolation not in _INTERPOLATIONS:
        raise ValueError("interpolation must be one of {}. Got: {!r}"
                         .format(", ".join(_INTERPOLATIONS), interpolation))
    arr = np.asarray(data)
    if arr.dtype.kind not in "biuf":
        raise ValueError("data must be numeric. Got dtype: {}".format(arr.dtype))
    if arr.size == 0:
        raise ValueError("Input must not be empty.")
    arr = arr.astype(np.float64)
    axis = _check_axis(axis, arr.ndim)

    try:
        q_arr = np.asarray(q, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("q must be a number or a 1D array of numbers. Got: {!r}".format(q))
    if q_arr.ndim > 1 or q_arr.size == 0:
        raise ValueError("q must be a number or a non-empty 1D array.")
    if np.isnan(q_arr).any() or (q_arr < 0).any() or (q_arr > 100).any():
        raise ValueError("q values must be between 0 and 100.")

    work = _to_last_axis(arr, axis)
    if axis is None:
        work = work.reshape(1, -1)                       # one slice: shape (1, m)
    sorted_data = np.sort(work, axis=-1)
    m = sorted_data.shape[-1]

    n_valid = np.sum(~np.isnan(sorted_data), axis=-1, keepdims=True)   # (..., 1)
    pos = q_arr.ravel() / 100.0 * (n_valid - 1)                         # (..., k)
    lower_idx = np.clip(np.floor(pos).astype(np.intp), 0, m - 1)
    upper_idx = np.clip(np.ceil(pos).astype(np.intp), 0, m - 1)
    below = np.take_along_axis(sorted_data, lower_idx, axis=-1)
    above = np.take_along_axis(sorted_data, upper_idx, axis=-1)

    if interpolation == "lower":
        result = below
    elif interpolation == "higher":
        result = above
    elif interpolation == "midpoint":
        result = np.where(above == below, below, (below + above) / 2.0)
    else:
        frac = pos - np.floor(pos)
        result = np.where(above == below, below, below + frac * (above - below))
    result = np.where(n_valid == 0, np.nan, result)

    # (..., k) -> (k, ...) like NumPy, then tidy the shape
    result = np.moveaxis(result, -1, 0)
    if axis is None:
        result = result[:, 0]
    if q_arr.ndim == 0:
        result = result[0]
    if np.ndim(result) == 0:
        return float(result)
    return result
