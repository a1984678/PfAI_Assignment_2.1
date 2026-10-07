"""
utils.py

Shared numerical helpers (NumPy only).

- pairwise_distances(X, Y, metric) — euclidean / manhattan / cosine distance matrix
- sigmoid, relu, leaky_relu, tanh, softmax, log_softmax — activations
- logsumexp(x, axis, keepdims) — log(sum(exp(x))) without overflow
- row_topk(M, k, largest) — indices of the k best entries in every row
- batch_indices(n, batch_size, ...), batches(*arrays, batch_size, ...) — mini-batching
"""

import numpy as np

_METRICS = ("euclidean", "manhattan", "cosine")


# ---------------------------------------------------------------------
# Validation helpers (private)
# ---------------------------------------------------------------------

def _as_float(data, name):
    """Numeric array-like -> float64 array. Rejects empty and non-numeric input."""
    arr = np.asarray(data)
    if arr.dtype.kind not in "biuf":
        raise ValueError("{} must be numeric. Got dtype: {}".format(name, arr.dtype))
    if arr.size == 0:
        raise ValueError("Input must not be empty.")
    return arr.astype(np.float64)


def _check_axis(axis, ndim):
    """Validate axis (None or int, negatives allowed). Returns None or a non-negative int."""
    if axis is None:
        return None
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be None or an integer. Got: {!r}".format(axis))
    if not -ndim <= axis < ndim:
        raise ValueError("axis {} is out of bounds for input with {} dimension(s)."
                         .format(axis, ndim))
    return int(axis) % ndim


def _as_matrix(data, name):
    """Numeric 2D float64 array without NaN/inf."""
    arr = _as_float(data, name)
    if arr.ndim != 2:
        raise ValueError("{} must be 2-dimensional. Got shape: {}".format(name, arr.shape))
    if not np.isfinite(arr).all():
        raise ValueError("{} must not contain NaN or infinity.".format(name))
    return arr


def _check_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("{} must be a positive integer. Got: {!r}".format(name, value))
    return int(value)


# ---------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------

def pairwise_distances(X, Y=None, metric="euclidean"):
    """
    Distance between every row of X and every row of Y.

    Parameters:
    ----------
    - X: array-like, shape (n, d)
    - Y: array-like, shape (m, d), or None (default) to compare X with itself
    - metric: 'euclidean' (default), 'manhattan' or 'cosine'
      - cosine distance = 1 - cosine similarity; a zero vector has distance 1 to everything

    Returns:
    -------
    - D: np.ndarray, shape (n, m), D[i, j] = distance(X[i], Y[j])

    Raises:
    ------
    - ValueError: if inputs are not numeric 2D arrays without NaN/inf, the number of
    columns differs, or metric is unknown

    Complexity:
    ----------
    - Time complexity: O(n * m * d)
    - Space complexity: O(n * m)

    Note:
    ----
    - euclidean uses |x|^2 + |y|^2 - 2 x.y so the heavy work is one matrix product
    (no (n, m, d) intermediate array); tiny negative values from rounding are clipped to 0
    and the diagonal of a self-comparison is exactly 0.
    - manhattan loops over the d columns (not over rows), adding |x_j - y_j| each time,
    which keeps memory at O(n * m).
    """
    if metric not in _METRICS:
        raise ValueError("metric must be one of {}. Got: {!r}"
                         .format(", ".join(_METRICS), metric))
    A = _as_matrix(X, "X")
    same = Y is None
    B = A if same else _as_matrix(Y, "Y")
    if A.shape[1] != B.shape[1]:
        raise ValueError("X and Y must have the same number of columns. Got {} and {}."
                         .format(A.shape[1], B.shape[1]))

    if metric == "euclidean":
        sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
        D = np.sqrt(np.maximum(sq, 0.0))
    elif metric == "manhattan":
        D = np.zeros((A.shape[0], B.shape[0]))
        for j in range(A.shape[1]):
            D += np.abs(A[:, j][:, None] - B[:, j][None, :])
    else:
        na = np.linalg.norm(A, axis=1)
        nb = np.linalg.norm(B, axis=1)
        denom = na[:, None] * nb[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, (A @ B.T) / denom, 0.0)
        D = 1.0 - np.clip(sim, -1.0, 1.0)
    if same:
        np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------

def sigmoid(x):
    """
    Logistic function 1 / (1 + exp(-x)), computed so exp never overflows.

    Parameters:
    ----------
    - x: array-like of numbers, any shape

    Returns:
    -------
    - np.ndarray (or float for a scalar) with values in [0, 1], same shape as x

    Raises:
    ------
    - ValueError: if x is non-numeric or empty

    Complexity:
    ----------
    - Time and space complexity: O(n)

    Note:
    ----
    - exp(-|x|) is always <= 1, and the two halves of the formula are joined with np.where:
    x >= 0: 1 / (1 + e), x < 0: e / (1 + e), where e = exp(-|x|).
    """
    arr = _as_float(x, "x")
    e = np.exp(-np.abs(arr))
    out = np.where(arr >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return float(out) if out.ndim == 0 else out


def relu(x):
    """
    max(0, x) element-wise.

    Parameters:
    ----------
    - x: array-like of numbers, any shape

    Returns:
    -------
    - np.ndarray (float for a scalar), same shape as x

    Raises:
    ------
    - ValueError: if x is non-numeric or empty

    Complexity:
    ----------
    - Time and space complexity: O(n)
    """
    out = np.maximum(_as_float(x, "x"), 0.0)
    return float(out) if out.ndim == 0 else out


def leaky_relu(x, alpha=0.01):
    """
    x for x > 0, alpha * x otherwise.

    Parameters:
    ----------
    - x: array-like of numbers, any shape
    - alpha: float >= 0, slope for negative inputs

    Returns:
    -------
    - np.ndarray (float for a scalar), same shape as x

    Raises:
    ------
    - ValueError: if x is non-numeric or empty, or alpha is negative or not a number

    Complexity:
    ----------
    - Time and space complexity: O(n)
    """
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or alpha < 0:
        raise ValueError("alpha must be a number >= 0. Got: {!r}".format(alpha))
    arr = _as_float(x, "x")
    out = np.where(arr > 0, arr, alpha * arr)
    return float(out) if out.ndim == 0 else out


def tanh(x):
    """
    Hyperbolic tangent, output in [-1, 1].

    Parameters:
    ----------
    - x: array-like of numbers, any shape

    Returns:
    -------
    - np.ndarray (float for a scalar), same shape as x

    Raises:
    ------
    - ValueError: if x is non-numeric or empty

    Complexity:
    ----------
    - Time and space complexity: O(n)
    """
    out = np.tanh(_as_float(x, "x"))
    return float(out) if out.ndim == 0 else out


def logsumexp(x, axis=None, keepdims=False):
    """
    log(sum(exp(x))) without overflow or underflow.

    Parameters:
    ----------
    - x: array-like of numbers (-inf allowed)
    - axis: None (default, all values) or int
    - keepdims: bool, keep the reduced axis with length 1

    Returns:
    -------
    - float if axis is None and keepdims is False, else np.ndarray

    Raises:
    ------
    - ValueError: if x is non-numeric, empty or contains NaN, or axis is invalid

    Complexity:
    ----------
    - Time and space complexity: O(n)

    Note:
    ----
    - Subtracting the maximum m first gives m + log(sum(exp(x - m))). exp(1000) overflows
    on its own, but exp(1000 - 1000) = 1 does not.
    - If every value along the axis is -inf the answer is -inf.
    """
    arr = _as_float(x, "x")
    if np.isnan(arr).any():
        raise ValueError("x must not contain NaN.")
    axis = _check_axis(axis, arr.ndim)
    m = np.max(arr, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(arr - m_safe), axis=axis, keepdims=True)) + m_safe
    if not keepdims:
        out = out.reshape(()) if axis is None else np.squeeze(out, axis=axis)
    return float(out) if np.ndim(out) == 0 else out


def softmax(x, axis=-1):
    """
    exp(x) / sum(exp(x)) along an axis, so the values sum to 1.

    Parameters:
    ----------
    - x: array-like of numbers, at least 1D
    - axis: int, default -1 (last axis)

    Returns:
    -------
    - np.ndarray, same shape as x, non-negative, summing to 1 along axis

    Raises:
    ------
    - ValueError: if x is non-numeric, empty or contains NaN/inf, or axis is invalid

    Complexity:
    ----------
    - Time and space complexity: O(n)

    Note:
    ----
    - Computed as exp(x - logsumexp(x)), so large inputs such as 1000 do not overflow.
    """
    arr = _as_float(x, "x")
    if arr.ndim == 0:
        raise ValueError("x must be at least 1-dimensional.")
    if not np.isfinite(arr).all():
        raise ValueError("x must not contain NaN or infinity.")
    _check_axis(axis, arr.ndim)
    return np.exp(arr - logsumexp(arr, axis=axis, keepdims=True))


def log_softmax(x, axis=-1):
    """
    log(softmax(x)) computed directly, which stays accurate when probabilities are tiny.

    Parameters:
    ----------
    - x: array-like of numbers, at least 1D
    - axis: int, default -1 (last axis)

    Returns:
    -------
    - np.ndarray, same shape as x, all values <= 0

    Raises:
    ------
    - ValueError: same as softmax

    Complexity:
    ----------
    - Time and space complexity: O(n)
    """
    arr = _as_float(x, "x")
    if arr.ndim == 0:
        raise ValueError("x must be at least 1-dimensional.")
    if not np.isfinite(arr).all():
        raise ValueError("x must not contain NaN or infinity.")
    _check_axis(axis, arr.ndim)
    return arr - logsumexp(arr, axis=axis, keepdims=True)


# ---------------------------------------------------------------------
# Top-k helper
# ---------------------------------------------------------------------

def row_topk(M, k, largest=True):
    """
    Column indices of the k largest (or smallest) entries in every row.

    Parameters:
    ----------
    - M: array-like, shape (n, m), numeric, no NaN
    - k: int, 1 <= k <= m
    - largest: bool, True (default) for the k largest, False for the k smallest

    Returns:
    -------
    - idx: np.ndarray of int, shape (n, k), best first in each row

    Raises:
    ------
    - ValueError: if M is not a numeric 2D array without NaN, or k is out of range

    Complexity:
    ----------
    - Time complexity: O(n * m log m)
    - Space complexity: O(n * m)

    Note:
    ----
    - Uses a stable sort, so equal values are returned in column order and the result
    is the same on every run. Typical use: row_topk(pairwise_distances(Q, X), 5,
    largest=False) gives the 5 nearest neighbours of each query row.
    """
    arr = np.asarray(M)
    if arr.dtype.kind not in "biuf" or arr.ndim != 2:
        raise ValueError("M must be a numeric 2-dimensional array.")
    if arr.size == 0:
        raise ValueError("Input must not be empty.")
    if np.isnan(arr.astype(np.float64)).any():
        raise ValueError("M must not contain NaN.")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) \
            or not 1 <= k <= arr.shape[1]:
        raise ValueError("k must be an integer between 1 and {}. Got: {!r}"
                         .format(arr.shape[1], k))
    key = -arr.astype(np.float64) if largest else arr
    return np.argsort(key, axis=1, kind="stable")[:, :int(k)]


# ---------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------

def batch_indices(n, batch_size, shuffle=False, seed=0, drop_last=False):
    """
    Yield index arrays that split range(n) into mini-batches.

    Parameters:
    ----------
    - n: int >= 1, number of samples
    - batch_size: int >= 1
    - shuffle: bool, shuffle the order first (default False)
    - seed: int, used only when shuffle=True, so runs are repeatable
    - drop_last: bool, drop a final batch smaller than batch_size

    Yields:
    ------
    - np.ndarray of int, length batch_size (the last may be shorter)

    Raises:
    ------
    - ValueError: if n or batch_size is not a positive integer

    Complexity:
    ----------
    - Time complexity: O(n) in total
    - Space complexity: O(n) if shuffle, O(batch_size) otherwise
    """
    n = _check_positive_int(n, "n")
    batch_size = _check_positive_int(batch_size, "batch_size")
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        chunk = order[start:start + batch_size]
        if drop_last and chunk.size < batch_size:
            return
        yield chunk


def batches(*arrays, batch_size, shuffle=False, seed=0, drop_last=False):
    """
    Yield aligned mini-batches from one or more arrays (e.g. X and y).

    Parameters:
    ----------
    - arrays: one or more array-likes with the same length along axis 0
    - batch_size, shuffle, seed, drop_last: see batch_indices

    Yields:
    ------
    - tuple of np.ndarray, one slice per input array (same rows in each)

    Raises:
    ------
    - ValueError: if no array is given or the lengths differ

    Complexity:
    ----------
    - Time complexity: O(n) in total
    - Space complexity: O(batch_size) per batch
    """
    if not arrays:
        raise ValueError("At least one array is required.")
    arrs = [np.asarray(a) for a in arrays]
    if any(a.ndim == 0 for a in arrs):
        raise ValueError("Arrays must be at least 1-dimensional.")
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs):
        raise ValueError("All arrays must have the same length. Got: {}"
                         .format([a.shape[0] for a in arrs]))
    for idx in batch_indices(n, batch_size, shuffle, seed, drop_last):
        yield tuple(a[idx] for a in arrs)