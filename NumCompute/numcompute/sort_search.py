"""
sort_search.py

- Sorting:
  - Stable sort wrapper (np.sort(kind='stable'))
  - Multi-key sort (sort by multiple columns)
- Top-k / Partial Sort:
  - topk(values, k, largest=True, return_indices=True) using np.argpartition
  - quickselect for educational purposes
- Searching:
  - binary_search(sorted_array, x) returning insertion index and existence boolean
"""

import numpy as np

_SORTABLE_KINDS = "biufUSmM"  # bool, ints, floats, strings, dates
_NUMERIC_KINDS = "biuf"


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _as_sortable(values, name="values"):
    """
    Convert input to an array whose values can be ordered.

    Parameters:
    ----------
    - values: array-like
    - name: str, the argument name used in error messages

    Returns:
    -------
    - np.ndarray (no copy if values is already an array)

    Raises:
    ------
    - ValueError: if the values cannot be ordered (e.g. object or complex arrays)

    Complexity:
    ----------
    - Time and space complexity: O(n) if a conversion is needed, else O(1)
    """
    arr = np.asarray(values)
    if arr.dtype.kind not in _SORTABLE_KINDS:
        raise ValueError("{} must be numbers, strings or dates. Got dtype: {}"
                         .format(name, arr.dtype))
    return arr


def _as_numeric_1d(values, name="values"):
    """
    Validate a 1D numeric array without NaN (used by topk and quickselect).

    Parameters:
    ----------
    - values: array-like of shape (n,)
    - name: str, the argument name used in error messages

    Returns:
    -------
    - np.ndarray of shape (n,)

    Raises:
    ------
    - ValueError: if values is not 1D, not numeric, or contains NaN

    Complexity:
    ----------
    - Time complexity: O(n) for the NaN check
    - Space complexity: O(1) extra (O(n) if a conversion is needed)
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("{} must be 1-dimensional. Got shape: {}".format(name, arr.shape))
    if arr.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError("{} must be numeric. Got dtype: {}".format(name, arr.dtype))
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError("{} must not contain NaN (the order of NaN is undefined)."
                         .format(name))
    return arr


def _check_k(k, upper, name="k"):
    """
    Validate an integer k with 0 <= k <= upper.

    Parameters:
    ----------
    - k: the value to check
    - upper: int, largest allowed value
    - name: str, the argument name used in error messages

    Returns:
    -------
    - int

    Raises:
    ------
    - ValueError: if k is not an integer or is outside [0, upper]

    Complexity:
    ----------
    - Time and space complexity: O(1)
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("{} must be an integer. Got: {!r}".format(name, k))
    if not 0 <= k <= upper:
        raise ValueError("{} must be between 0 and {}. Got: {}".format(name, upper, k))
    return int(k)


def _stable_argsort(values, axis=-1, descending=False):
    """
    Stable argsort along an axis, in ascending or descending order.

    Parameters:
    ----------
    - values: np.ndarray
    - axis: int, the axis to sort along
    - descending: bool

    Returns:
    -------
    - np.ndarray of indices, same shape as values

    Complexity:
    ----------
    - Time complexity: O(n log m) where m is the length of the axis
    - Space complexity: O(n)

    Note:
    ----
    - Reversing an ascending sort would put tied values in reverse order. Instead the
    array is flipped, sorted stably, and flipped back, so equal values keep their
    original order in both directions. This works for every sortable dtype,
    including strings (negating the values would not).
    """
    if not descending:
        return np.argsort(values, axis=axis, kind="stable")
    m = values.shape[axis]
    flipped = np.flip(values, axis=axis)
    order = np.argsort(flipped, axis=axis, kind="stable")
    return (m - 1) - np.flip(order, axis=axis)


# ---------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------

def stable_sort(values, axis=-1, descending=False, return_indices=False):
    """
    Sort values with a stable algorithm: equal values keep their original order.

    Parameters:
    ----------
    - values: array-like of any shape (numbers, strings or dates)
    - axis: int or None, the axis to sort along (default -1, the last axis).
    None flattens the array first
    - descending: bool, largest first if True (default False)
    - return_indices: bool, also return the indices that sort the array (default False)

    Returns:
    -------
    - sorted: np.ndarray, same shape as values (1D if axis is None)
    - indices: np.ndarray of the same shape, only if return_indices is True,
    such that np.take_along_axis(values, indices, axis) == sorted

    Raises:
    ------
    - ValueError: if values cannot be ordered, is 0-dimensional, or axis is invalid

    Complexity:
    ----------
    - Time complexity: O(n log m) where m is the length of the sorted axis
    - Space complexity: O(n)

    Note:
    ----
    - NaN values go last in ascending order and first in descending order.
    - The input is never modified.
    """
    arr = _as_sortable(values)
    if axis is None:
        arr = arr.ravel()
        axis = 0
    if arr.ndim == 0:
        raise ValueError("values must have at least 1 dimension.")
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)) \
            or not -arr.ndim <= axis < arr.ndim:
        raise ValueError("axis {!r} is invalid for input with {} dimension(s)."
                         .format(axis, arr.ndim))

    order = _stable_argsort(arr, axis=axis, descending=descending)
    result = np.take_along_axis(arr, order, axis=axis)
    if return_indices:
        return result, order
    return result


def multikey_sort(X, keys, ascending=True, return_indices=False):
    """
    Sort the rows of a 2D array by several columns: the first key decides, later keys
    break ties.

    Parameters:
    ----------
    - X: array-like of shape (n_rows, n_cols)
    - keys: int or sequence of ints, column indices in priority order
    (negative indices count from the end)
    - ascending: bool or sequence of bools (one per key), default True
    - return_indices: bool, also return the row order (default False)

    Returns:
    -------
    - X_sorted: np.ndarray of shape (n_rows, n_cols), the rows reordered
    - order: np.ndarray of shape (n_rows,), only if return_indices is True,
    such that X[order] == X_sorted

    Raises:
    ------
    - ValueError: if X is not 2D or cannot be ordered, keys is empty or out of range,
    or ascending does not match the number of keys

    Complexity:
    ----------
    - Time complexity: O(p * n log n) for p keys and n rows
    - Space complexity: O(p * n)

    Note:
    ----
    - Uses np.lexsort, which is stable: rows equal on every key keep their order.
    - np.lexsort treats its LAST key as the primary one, so the keys are passed in
    reverse.
    - A descending key is replaced by its negated rank codes (from np.unique), which
    works for strings too.
    """
    X = _as_sortable(X, name="X")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional. Got shape: {}".format(X.shape))

    keys = [keys] if isinstance(keys, (int, np.integer)) else list(keys)
    if not keys:
        raise ValueError("keys must name at least one column.")
    n_cols = X.shape[1]
    for key in keys:
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)) \
                or not -n_cols <= key < n_cols:
            raise ValueError("Column key {!r} is invalid for {} columns.".format(key, n_cols))

    if isinstance(ascending, (bool, np.bool_)):
        ascending = [bool(ascending)] * len(keys)
    else:
        ascending = list(ascending)
        if len(ascending) != len(keys):
            raise ValueError("ascending must be a bool or have one entry per key. "
                             "Got {} entries for {} keys.".format(len(ascending), len(keys)))

    columns = []
    for key, asc in zip(keys, ascending):
        col = X[:, key]
        if not asc:
            col = -np.unique(col, return_inverse=True)[1]
        columns.append(col)

    order = np.lexsort(columns[::-1])
    if return_indices:
        return X[order], order
    return X[order]


# ---------------------------------------------------------------------
# Top-k / partial sort
# ---------------------------------------------------------------------

def topk(values, k, largest=True, return_indices=True):
    """
    The k largest (or smallest) values, without sorting the whole array.

    Parameters:
    ----------
    - values: array-like of shape (n,), numeric
    - k: int, 0 <= k <= n
    - largest: bool, the k largest if True (default), else the k smallest
    - return_indices: bool, also return their positions in values (default True)

    Returns:
    -------
    - top_values: np.ndarray of shape (k,), best first
    (largest first if largest=True, smallest first otherwise)
    - indices: np.ndarray of shape (k,), only if return_indices is True,
    such that values[indices] == top_values

    Raises:
    ------
    - ValueError: if values is not 1D numeric, contains NaN, or k is not in [0, n]

    Complexity:
    ----------
    - Time complexity: O(n + k log k): np.argpartition finds the k-th value in O(n)
    on average, then only the k selected values are sorted
    - Space complexity: O(n)

    Note:
    ----
    - Ties are broken by position: among equal values the earlier index comes first,
    and if a tie straddles the cut-off the earliest ones are kept. np.argpartition
    alone would pick arbitrarily, so the result would not be reproducible.
    """
    arr = _as_numeric_1d(values)
    n = arr.size
    k = _check_k(k, n)

    if k == 0:
        top_idx = np.empty(0, dtype=np.intp)
    else:
        # value at the cut-off: the k-th largest (or smallest)
        pivot_pos = n - k if largest else k - 1
        threshold = arr[np.argpartition(arr, pivot_pos)[pivot_pos]]

        better = np.flatnonzero(arr > threshold if largest else arr < threshold)
        ties = np.flatnonzero(arr == threshold)[:k - better.size]
        chosen = np.sort(np.concatenate([better, ties]))  # index order
        top_idx = chosen[_stable_argsort(arr[chosen], descending=largest)]

    if return_indices:
        return arr[top_idx], top_idx
    return arr[top_idx]


def quickselect(values, k, seed=0):
    """
    The k-th smallest value (k = 0 is the minimum) using quickselect.
    Included for educational purposes; use np.partition in real code.

    Parameters:
    ----------
    - values: array-like of shape (n,), numeric
    - k: int, 0 <= k < n, the 0-based rank of the value wanted
    - seed: int, seed for the random pivot choice (default 0, for reproducibility)

    Returns:
    -------
    - the k-th smallest value, as a Python number

    Raises:
    ------
    - ValueError: if values is empty, not 1D numeric, contains NaN, or k is not in [0, n)

    Complexity:
    ----------
    - Time complexity: O(n) on average (n + n/2 + n/4 + ...), O(n^2) in the very
    unlikely worst case where every pivot is the smallest or largest value
    - Space complexity: O(n) for the partitions

    Note:
    ----
    - Each round picks a random pivot and splits the values into three groups:
    smaller than, equal to, and larger than the pivot. Only the group that contains
    the k-th value is kept, so on average half the data is thrown away per round.
    - The split into groups is vectorised with boolean masks; the loop runs over
    rounds (about log n of them), not over values.
    - Keeping an "equal" group means arrays with many duplicates still finish quickly.
    """
    arr = _as_numeric_1d(values)
    if arr.size == 0:
        raise ValueError("values must not be empty.")
    k = _check_k(k, arr.size - 1)
    rng = np.random.default_rng(seed)

    while True:
        pivot = arr[rng.integers(arr.size)]
        smaller = arr[arr < pivot]
        n_equal = np.count_nonzero(arr == pivot)

        if k < smaller.size:
            arr = smaller                       # answer is among the smaller values
        elif k < smaller.size + n_equal:
            return pivot.item()                 # answer is the pivot itself
        else:
            k -= smaller.size + n_equal         # skip the smaller and equal values
            arr = arr[arr > pivot]


# ---------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------

def binary_search(sorted_array, x, check_sorted=False):
    """
    Find where x is, or would be inserted, in an ascending sorted 1D array.

    Parameters:
    ----------
    - sorted_array: array-like of shape (n,), sorted in ascending order
    - x: a value, or array-like of values to look up all at once
    - check_sorted: bool, verify the array is sorted first (default False, since the
    check costs O(n) and binary search is meant to be O(log n))

    Returns:
    -------
    - index: int (or np.ndarray of ints for array x), the leftmost position where x
    could be inserted keeping the order (0 to n)
    - found: bool (or np.ndarray of bools), True if x is in the array
    (then sorted_array[index] == x)

    Raises:
    ------
    - ValueError: if sorted_array is not 1D or cannot be ordered, x cannot be compared
    with it, or check_sorted is True and the array is not sorted

    Complexity:
    ----------
    - Time complexity: O(log n) per value of x (O(m log n) for m values)
    - Space complexity: O(m)

    Note:
    ----
    - Uses np.searchsorted (NumPy's binary search) with side='left', so for duplicates
    the index of the first copy is returned.
    - An empty array gives index 0 and found False.
    - NaN is never "found" (NaN != NaN). NaN values may sit at the end of the
    sorted array, which is where np.sort puts them.
    """
    arr = _as_sortable(sorted_array, name="sorted_array")
    if arr.ndim != 1:
        raise ValueError("sorted_array must be 1-dimensional. Got shape: {}"
                         .format(arr.shape))

    if check_sorted and arr.size > 1:
        in_order = arr[:-1] <= arr[1:]
        if arr.dtype.kind == "f":
            in_order |= np.isnan(arr[1:])
        if not np.all(in_order):
            raise ValueError("sorted_array is not sorted in ascending order.")

    # NumPy would silently compare numbers with strings as text, so check the kinds
    x_kind = np.asarray(x).dtype.kind
    groups = ("biuf", "US", "m", "M")
    same_group = any(arr.dtype.kind in g and x_kind in g for g in groups)
    if arr.size and not same_group:
        raise ValueError("x (dtype kind {!r}) cannot be compared with sorted_array (dtype {})."
                         .format(x_kind, arr.dtype))
    try:
        index = np.searchsorted(arr, x, side="left")
    except (TypeError, ValueError):
        raise ValueError("x cannot be compared with sorted_array (dtype {})."
                         .format(arr.dtype))

    if arr.size == 0:
        found = np.zeros(np.shape(index), dtype=bool)
    else:
        found = (index < arr.size) & (arr[np.minimum(index, arr.size - 1)] == x)

    if np.ndim(index) == 0:
        return int(index), bool(found)
    return index, found

