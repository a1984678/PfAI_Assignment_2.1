"""
optim.py
"""
import numpy as np

_METHODS = ("central", "forward")


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _check_point(x, h, method):
    """
    Validate the evaluation point, step size and method shared by grad and jacobian.

    Parameters:
    ----------
    - x: array-like, the point to differentiate at (scalar or any shape)
    - h: float, the finite-difference step size
    - method: str, 'central' or 'forward'

    Returns:
    -------
    - x: np.ndarray of dtype float64, a fresh copy (the caller's array is never changed)
    - h: float

    Raises:
    ------
    - ValueError: if method is invalid, h is not a positive finite number,
    or x is empty, non-numeric or not finite

    Complexity:
    ----------
    - Time complexity: O(n) where n is the number of elements in x
    - Space complexity: O(n) for the copy
    """
    if method not in _METHODS:
        raise ValueError("method must be 'central' or 'forward'. Got: {!r}".format(method))

    try:
        h = float(h)
    except (TypeError, ValueError):
        raise ValueError("h must be a positive finite number. Got: {!r}".format(h))
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number. Got: {!r}".format(h))

    x = np.array(x)  # np.array (not asarray) always copies
    if x.dtype.kind not in "biuf":
        raise ValueError("x must be numeric. Got dtype: {}".format(x.dtype))
    if x.size == 0:
        raise ValueError("x must not be empty.")

    # integer x must become float, otherwise x + 1e-5 would be truncated back to x
    x = x.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values.")

    return x, h


def _eval(func, point, expected_size=None):
    """
    Call func at point and return its output as a flat float64 array.

    Parameters:
    ----------
    - func: callable, the user's function
    - point: np.ndarray, where to evaluate it
    - expected_size: int or None, the output size every call must have

    Returns:
    -------
    - np.ndarray of shape (m,), dtype float64

    Raises:
    ------
    - ValueError: if the output is non-numeric, empty, or changes size between calls

    Complexity:
    ----------
    - Time complexity: O(cost of func + m)
    - Space complexity: O(m)
    """
    try:
        out = np.asarray(func(point), dtype=np.float64).ravel()
    except (TypeError, ValueError):
        raise ValueError("The function must return numeric values.")

    if out.size == 0:
        raise ValueError("The function returned an empty output.")
    if expected_size is not None and out.size != expected_size:
        raise ValueError("The function returned {} values, expected {}."
                         .format(out.size, expected_size))
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def grad(f, x, h=1e-5, method="central"):
    """
    Estimate the gradient of a scalar function f at x with finite differences.

    Parameters:
    ----------
    - f: callable, takes an array shaped like x and returns a single number
    - x: array-like of any shape (a scalar is allowed), the point to evaluate at
    - h: float, step size (default 1e-5)
    - method: str, 'central' (default) or 'forward'
      - central: (f(x + h*e_i) - f(x - h*e_i)) / (2h), error shrinks like h^2
      - forward: (f(x + h*e_i) - f(x)) / h, error shrinks like h

    Returns:
    -------
    - g: np.ndarray with the same shape as x (a float if x is a scalar),
    where g[i] is the estimated partial derivative df/dx_i

    Raises:
    ------
    - ValueError: if x, h or method are invalid (see _check_point)
    - ValueError: if f does not return exactly one number

    Complexity:
    ----------
    - Time complexity: O(n * C) where n is the number of elements in x and C is the
    cost of one call to f (central: 2n + 1 calls, forward: n + 1 calls)
    - Space complexity: O(n)

    Note:
    ----
    - The loop over coordinates is necessary: f is a black box that takes one point,
    so each partial derivative needs its own call(s) to f.
    - x is copied and converted to float, so integer input works and the caller's
    array is never modified.
    """
    x, h = _check_point(x, h, method)
    flat = x.ravel()
    n = flat.size

    def f_at(point_flat):
        return _eval(f, point_flat.reshape(x.shape), expected_size=1)[0]

    # check once that f returns a single number (also gives f(x) for forward)
    f0 = _eval(f, x)
    if f0.size != 1:
        raise ValueError("f must return a single number. Got {} values.".format(f0.size))
    f0 = f0[0]

    g = np.empty(n, dtype=np.float64)
    for i in range(n):
        step = np.zeros(n)
        step[i] = h
        if method == "central":
            g[i] = (f_at(flat + step) - f_at(flat - step)) / (2.0 * h)
        else:
            g[i] = (f_at(flat + step) - f0) / h

    if x.ndim == 0:
        return float(g[0])
    return g.reshape(x.shape)


def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian matrix of a vector function F at x with finite differences.

    Parameters:
    ----------
    - F: callable, takes an array shaped like x and returns m numbers (any shape,
    flattened to (m,))
    - x: array-like of any shape (n elements in total), the point to evaluate at
    - h: float, step size (default 1e-5)
    - method: str, 'central' (default) or 'forward'

    Returns:
    -------
    - J: np.ndarray of shape (m, n) where J[i, j] is the estimated dF_i/dx_j
    (x is flattened, so j indexes x.ravel())

    Raises:
    ------
    - ValueError: if x, h or method are invalid (see _check_point)
    - ValueError: if F returns non-numeric or empty output, or its size changes

    Complexity:
    ----------
    - Time complexity: O(n * C) where C is the cost of one call to F
    (central: 2n + 1 calls, forward: n + 1 calls)
    - Space complexity: O(m * n) for the Jacobian

    Note:
    ----
    - A scalar-valued F gives shape (1, n), i.e. the gradient as a row vector.
    - The loop over the n input coordinates is necessary for the same reason as in grad;
    each iteration fills a whole column of J at once.
    """
    x, h = _check_point(x, h, method)
    flat = x.ravel()
    n = flat.size

    F0 = _eval(F, x)
    m = F0.size
    J = np.empty((m, n), dtype=np.float64)

    for j in range(n):
        step = np.zeros(n)
        step[j] = h
        F_plus = _eval(F, (flat + step).reshape(x.shape), expected_size=m)
        if method == "central":
            F_minus = _eval(F, (flat - step).reshape(x.shape), expected_size=m)
            J[:, j] = (F_plus - F_minus) / (2.0 * h)
        else:
            J[:, j] = (F_plus - F0) / h

    return J

