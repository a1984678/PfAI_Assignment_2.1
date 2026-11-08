"""Decision tree implemented from scratch with streaming support.

``DecisionTreeClassifier``: depth-limited, Gini or entropy-based.

- ``.fit(X, y)``: trains the tree from scratch on the given data
- ``.partial_fit(X_chunk, y_chunk)`` supports streaming via buffered retraining:

  - Each call appends the chunk to an internal data buffer
  - Once the buffer holds at least ``buffer_size`` samples, the tree is
    retrained on the buffered data and the buffer is cleared
  - Between retrains, ``.predict()`` uses the most recently trained tree

- Support config: ``max_depth``, ``min_samples_split``, ``max_features``,
  ``buffer_size`` (plus ``criterion`` and ``seed``)

Streaming API
-------------
- ``partial_fit(X_chunk, y_chunk)``: buffer, retrain when full (see above)
- ``predict(X)`` / ``predict_proba(X)``: labels / leaf class fractions, shape
  (n_samples,) / (n_samples, n_classes); ``classes_`` holds every label seen so far
- ``n_buffered_``, ``n_retrains_``: buffer state for logging

How it works
------------
Same rules as the course tree (``classification-tree.html``, Week 10): weighted gain,
a split needs gain > 0, ties go to the lowest feature then the lowest threshold, a
majority tie goes to the smallest class. Numeric features use threshold questions
``X[:, f] <= t``. The best split over all features and thresholds is found at once
(sort, cumulative class counts), the tree is stored as flat arrays, and prediction
moves every row down one level per step. Python loops remain only once per node
(build) and once per tree level (predict). NaN is rejected: impute upstream.

Example
-------
>>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
>>> for X_chunk, y_chunk in stream:
...     tree.partial_fit(X_chunk, y_chunk)
"""

import numpy as np

from ._validation import check_array, check_n_features, check_positive_int, check_X_y
from .pipeline import Estimator

_CRITERIA = ("gini", "entropy")


def _impurity(counts, criterion):
    """
    Gini (1 - sum p^2) or entropy (-sum p log2 p) from class counts.

    Parameters:
    ----------
    - counts: np.ndarray of shape (..., n_classes), class counts of one or more groups
    - criterion: "gini" or "entropy"

    Returns:
    -------
    - np.ndarray of shape (...), impurity per group; an empty group has impurity 0
    """
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros(counts.shape), where=n > 0)
    if criterion == "gini":
        return 1.0 - (p * p).sum(axis=-1)
    logp = np.log2(p, out=np.zeros(p.shape), where=p > 0)
    return -(p * logp).sum(axis=-1)


def _best_split(X, y, n_classes, criterion):
    """
    Best threshold question over all features and thresholds at once (no Python loops).

    Sort every feature once; the one-hot labels in sorted order, summed cumulatively, give
    the left class counts of every split position, and right = total - left. Only
    positions between two different values are real thresholds.

    Parameters:
    ----------
    - X: np.ndarray of shape (n, d), float64, n >= 2
    - y: np.ndarray of shape (n,), int class indices in [0, n_classes)
    - n_classes: int
    - criterion: "gini" or "entropy"

    Returns:
    -------
    - (feature, threshold) with feature an index into X's columns, or None if no
    question lowers the impurity (gain must be > 0). Ties: lowest feature, then lowest
    threshold (as in the course loop).

    Complexity:
    ----------
    - Time: O(d * n log n) for the sort plus O(n * d * k) for the counts
    - Space: O(n * d * k) int64 for the cumulative counts (e.g. 5000 x 50 x 10 -> 20 MB)
    """
    n = X.shape[0]
    order = np.argsort(X, axis=0, kind="stable")                    # (n, d)
    xs = np.take_along_axis(X, order, axis=0)
    onehot = np.eye(n_classes, dtype=np.int64)[y[order]]            # (n, d, k)
    left = np.cumsum(onehot, axis=0)[:-1]                           # rows 0..i go left
    total = left[-1] + onehot[-1]                                   # (d, k)
    n_left = np.arange(1, n)[:, None]
    after = (n_left * _impurity(left, criterion)
             + (n - n_left) * _impurity(total - left, criterion)) / n
    gain = _impurity(total[0], criterion) - after                   # (n - 1, d)
    gain = np.where(xs[:-1] < xs[1:], gain, -np.inf)

    flat = gain.T.ravel()                    # feature-major: first maximum = lowest feature
    i = int(np.argmax(flat))
    if not flat[i] > 0:
        return None
    f, pos = divmod(i, n - 1)
    a, b = xs[pos, f], xs[pos + 1, f]
    t = a / 2 + b / 2                        # no overflow near 1e308
    return f, (a if t >= b else t)           # rounding up to b would not separate a and b


class DecisionTreeClassifier(Estimator):
    """
    Classification tree with numeric threshold questions ``X[:, f] <= t``.

    Same conventions as the course tree (classification-tree.html): the gain is the
    impurity before minus the size-weighted impurity after; a split must have gain > 0;
    ties go to the lowest feature index, then the lowest threshold; a leaf predicts the
    majority class, a tie going to the smallest class. Thresholds are midpoints between
    neighbouring distinct values.

    Parameters:
    ----------
    - max_depth: int >= 1 or None, maximum number of splits on any path (None = until
    pure or no split helps; default 5)
    - min_samples_split: int >= 2, a node with fewer samples is a leaf (default 2)
    - max_features: int >= 1 or None, number of features tried at each node, chosen at
    random with ``seed`` (None = all; default None)
    - buffer_size: int >= 1, samples collected by partial_fit before retraining
    (default 500)
    - criterion: "gini" or "entropy" (default "gini")
    - seed: int, seed for the max_features choice (default 0)

    Attributes (after fit):
    ----------
    - classes_: np.ndarray of shape (n_classes,), sorted labels seen in training
    - n_features_in_: int
    - feature_, threshold_, left_, right_: np.ndarray of shape (n_nodes,), the tree in
    preorder; a leaf has feature -1, threshold NaN and children -1
    - value_: np.ndarray of shape (n_nodes, n_classes), training class counts per node
    - n_nodes_, depth_: int
    - n_buffered_: int, rows waiting in the partial_fit buffer (always available)
    - n_retrains_: int, trees trained by partial_fit since construction or the last fit

    Memory: the buffer holds at most buffer_size + chunk_size - 1 rows.

    Raises:
    ------
    - ValueError: for invalid hyperparameters (at construction)
    """

    supports_partial_fit = True

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 buffer_size=500, criterion="gini", seed=0):
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}. Got: {criterion!r}")
        if max_depth is not None:
            max_depth = check_positive_int(max_depth, "max_depth")
        if (isinstance(min_samples_split, bool)
                or not isinstance(min_samples_split, (int, np.integer))
                or min_samples_split < 2):
            raise ValueError(f"min_samples_split must be at least 2. "
                             f"Got: {min_samples_split!r}")
        if max_features is not None:
            max_features = check_positive_int(max_features, "max_features")
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.buffer_size = check_positive_int(buffer_size, "buffer_size")
        self.criterion = criterion
        self.seed = seed
        self._reset_stream()

    def _reset_stream(self):
        """Empty the partial_fit buffer and the retrain counter."""
        self._buffer_X, self._buffer_y = [], []
        self._n_retrains = 0

    @property
    def n_buffered_(self):
        return sum(len(chunk) for chunk in self._buffer_y)

    @property
    def n_retrains_(self):
        return self._n_retrains

    def _check_max_features(self, n_features):
        if self.max_features is not None and self.max_features > n_features:
            raise ValueError(f"max_features={self.max_features} is larger than the number "
                             f"of features ({n_features}).")

    def fit(self, X, y):
        """
        Build the tree from scratch on X, y.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), numeric, no NaN or infinity
        - y: array-like of shape (n_samples,), labels (numbers or strings)

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X or y is invalid, or max_features > n_features

        Any partial_fit buffer is cleared: fit always starts again.
        """
        X, y = check_X_y(X, y)
        self._check_max_features(X.shape[1])
        self._reset_stream()
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self._build(X, y)
        return self

    def _build(self, X, y):
        """
        Grow the tree with an explicit stack (no recursion limit), nodes in preorder.

        One loop step per node; the work inside a node (split search, row partition) is
        vectorised. Each node stores its class counts, so predict_proba needs no extra pass.
        """
        k = len(self.classes_)
        rng = np.random.default_rng(self.seed)       # fresh per fit: refit = same tree
        n_try = self.max_features or X.shape[1]
        feature, threshold, left, right, value, depth = [], [], [], [], [], []
        stack = [(np.arange(len(y)), 0, -1, None)]   # (rows, depth, parent, side)
        while stack:
            rows, d, parent, side = stack.pop()
            idx = len(feature)
            if parent >= 0:
                (left if side == "left" else right)[parent] = idx
            counts = np.bincount(y[rows], minlength=k)
            feature.append(-1)
            threshold.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(counts)
            depth.append(d)

            if ((self.max_depth is not None and d >= self.max_depth)
                    or len(rows) < self.min_samples_split
                    or np.count_nonzero(counts) == 1):
                continue
            cols = np.arange(X.shape[1])
            if n_try < len(cols):
                cols = np.sort(rng.choice(cols, n_try, replace=False))  # sorted: tie-break
            split = _best_split(X[np.ix_(rows, cols)], y[rows], k, self.criterion)
            if split is None:
                continue
            f, t = int(cols[split[0]]), split[1]
            feature[idx], threshold[idx] = f, t
            go_left = X[rows, f] <= t
            # push right first so the left subtree is built (and numbered) first
            stack.append((rows[~go_left], d + 1, idx, "right"))
            stack.append((rows[go_left], d + 1, idx, "left"))

        self.feature_ = np.array(feature, dtype=np.int64)
        self.threshold_ = np.array(threshold, dtype=np.float64)
        self.left_ = np.array(left, dtype=np.int64)
        self.right_ = np.array(right, dtype=np.int64)
        self.value_ = np.array(value, dtype=np.int64)
        self.n_nodes_ = len(feature)
        self.depth_ = max(depth)

    def _leaves(self, X):
        """
        Leaf index reached by every row: all rows move down one level per loop step, so the
        loop runs depth_ times, not n_samples times.
        """
        if not hasattr(self, "feature_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet; call fit, or "
                               f"partial_fit until the buffer is full ({self.n_buffered_}/"
                               f"{self.buffer_size} samples buffered).")
        X = check_array(X)
        check_n_features(X, self.n_features_in_)
        at = np.zeros(X.shape[0], dtype=np.int64)
        active = np.flatnonzero(self.feature_[at] >= 0)
        while active.size:
            nodes = at[active]
            go_left = X[active, self.feature_[nodes]] <= self.threshold_[nodes]
            at[active] = np.where(go_left, self.left_[nodes], self.right_[nodes])
            active = active[self.feature_[at[active]] >= 0]
        return at

    def partial_fit(self, X, y):
        """
        Add one chunk to the buffer; retrain once it holds at least buffer_size samples.

        The retrain uses all buffered rows (possibly more than buffer_size) and then clears
        the buffer, so each tree is trained on the latest buffer only. Between retrains,
        predict uses the most recent tree (or the tree from fit). classes_ keeps every label
        seen in the stream, so predict_proba always has the same columns; a class missing
        from the latest buffer gets probability 0.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), one chunk, no NaN or infinity;
        the first chunk (or fit) fixes n_features
        - y: array-like of shape (n_samples,), labels (numbers or strings)

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if the chunk is invalid or its feature count changes (the buffer is
        left unchanged)
        - TypeError: if string and numeric labels are mixed across chunks
        """
        # validate everything first, so a bad chunk leaves the state unchanged
        X, y = check_X_y(X, y)
        n_features = check_n_features(X, getattr(self, "n_features_in_", None))
        self._check_max_features(n_features)
        seen = self.classes_ if hasattr(self, "classes_") else (
            self._buffer_y[0] if self._buffer_y else None)
        if seen is not None and (seen.dtype.kind in "US") != (y.dtype.kind in "US"):
            # np.union1d / concatenate would silently turn the number 1 into the string "1"
            raise TypeError("Mixed label types across chunks, use all numbers or all strings.")

        self.n_features_in_ = n_features
        # copy: check_X_y may return the caller's own arrays, and a caller that reuses them
        # for the next chunk would otherwise change the buffered data
        self._buffer_X.append(X.copy())
        self._buffer_y.append(y.copy())
        if self.n_buffered_ >= self.buffer_size:
            X_all = np.concatenate(self._buffer_X)
            y_all = np.concatenate(self._buffer_y)
            classes = np.unique(y_all)
            if hasattr(self, "classes_"):
                classes = np.union1d(self.classes_, classes)   # keep earlier classes
            self.classes_ = classes
            self._build(X_all, np.searchsorted(classes, y_all))
            self._buffer_X, self._buffer_y = [], []
            self._n_retrains += 1
        return self

    def predict_proba(self, X):
        """
        Class probabilities: the training class fractions of the leaf each row reaches.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features_in_)

        Returns:
        -------
        - np.ndarray of shape (n_samples, n_classes), columns in classes_ order

        Raises:
        ------
        - RuntimeError: if the tree is not fitted yet
        - ValueError: if X is invalid or has the wrong number of features
        """
        leaves = self._leaves(X)            # raises first if not fitted
        counts = self.value_[leaves]
        return counts / counts.sum(axis=1, keepdims=True)    # every node has >= 1 sample

    def predict(self, X):
        """
        Predicted labels: the majority class of the leaf each row reaches.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features_in_)

        Returns:
        -------
        - np.ndarray of shape (n_samples,), values from classes_

        Raises:
        ------
        - RuntimeError: if the tree is not fitted yet
        - ValueError: if X is invalid or has the wrong number of features
        """
        leaves = self._leaves(X)
        return self.classes_[self.value_[leaves].argmax(axis=1)]   # tie -> smallest class
