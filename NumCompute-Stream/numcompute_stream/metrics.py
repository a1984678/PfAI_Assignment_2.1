"""Streaming classification metrics.

Streaming API (each class has ``update(y_true_chunk, y_pred_chunk) -> self``,
``reset() -> self`` and ``result()``; ``window=w`` scores only the last w chunks):

- ``StreamingConfusionMatrix(labels=None, window=None)``: counts accumulate over chunks;
  classes are added (in sorted order) as they first appear
- ``Accuracy``, ``Precision``, ``Recall``, ``F1``: scores computed from a streaming
  confusion matrix with the 2.1 formulas, so a streamed result equals the batch function
  on the same data. ``result()`` is NaN before any data; 0/0 per class is 0.0 (as in 2.1)

Batch functions (adapted from NumCompute, Assignment 2.1)
---------------------------------------------------------
- Classification: accuracy, precision, recall, f1, confusion_matrix
- Regression: mse(y_true, y_pred)
- roc_curve and auc for binary classification

Fully vectorised with NumPy (np.bincount for counting, no loops over samples).
"""
from collections import deque

import numpy as np

from ._validation import check_positive_int

_AVERAGES = ("binary", "macro")


# ---------------------------------------------------------------------

def _check_targets(y_true, y_pred):
    """
    Validate a pair of 1D label/target arrays. Every metric calls this first.

    Parameters:
    ----------
    - y_true: array-like of shape (n_samples,), the true labels or targets
    - y_pred: array-like of shape (n_samples,), the predicted labels, scores or targets

    Returns:
    -------
    - (y_true, y_pred): tuple of np.ndarray, both of shape (n_samples,)

    Raises:
    ------
    - ValueError: if either input is not 1-dimensional, the lengths differ,
    the inputs are empty, or a floating-point input contains NaN

    Complexity:
    ----------
    - Time complexity: O(n) where n is the number of samples
    - Space complexity: O(n) if the inputs need converting to arrays
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional. Got shapes: {} and {}"
                         .format(y_true.shape, y_pred.shape))

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Inputs must have the same length. Got {} and {}"
                         .format(y_true.shape[0], y_pred.shape[0]))

    if y_true.size == 0:
        raise ValueError("Inputs must not be empty.")

    for arr in (y_true, y_pred):
        if arr.dtype.kind == "f" and np.isnan(arr).any():
            raise ValueError("Inputs must not contain NaN values.")

    return y_true, y_pred


def _to_float(arr, name):
    """
    Convert an array to float64, raising a clear error for non-numeric data.

    Parameters:
    ----------
    - arr: np.ndarray, the array to convert
    - name: str, the argument name used in the error message

    Returns:
    -------
    - np.ndarray of dtype float64 with the same shape as arr

    Raises:
    ------
    - ValueError: if arr is not numeric or contains infinite values

    Complexity:
    ----------
    - Time complexity: O(n)
    - Space complexity: O(n) for the converted copy
    """
    if arr.dtype.kind not in "biuf":
        raise ValueError("{} must be numeric. Got dtype: {}".format(name, arr.dtype))

    arr = arr.astype(np.float64)

    if not np.isfinite(arr).all():
        raise ValueError("{} must contain only finite values.".format(name))

    return arr


def _label_indices(values, labels):
    """
    Map each value to the position of that value in labels (vectorised lookup).

    Parameters:
    ----------
    - values: np.ndarray of shape (n_samples,), the labels to look up
    - labels: np.ndarray of shape (n_labels,), the reference labels (any order)

    Returns:
    -------
    - np.ndarray of shape (n_samples,), integer positions into labels

    Raises:
    ------
    - ValueError: if any value does not appear in labels

    Complexity:
    ----------
    - Time complexity: O((n + k) log k) where n is the number of values and k the number of labels
    - Space complexity: O(n + k)
    """
    sorter = np.argsort(labels, kind="stable")
    positions = np.searchsorted(labels, values, sorter=sorter)
    positions = np.clip(positions, 0, labels.size - 1)
    indices = sorter[positions]

    if not np.all(labels[indices] == values):
        raise ValueError("Found labels that are not present in `labels`.")

    return indices


def _safe_divide(numerator, denominator):
    """
    Element-wise division that returns 0.0 wherever the denominator is 0.

    Parameters:
    ----------
    - numerator: np.ndarray
    - denominator: np.ndarray with the same shape as numerator

    Returns:
    -------
    - np.ndarray of dtype float64 with the same shape as the inputs

    Complexity:
    ----------
    - Time complexity: O(k) where k is the number of elements
    - Space complexity: O(k)
    """
    out = np.zeros(np.shape(numerator), dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _class_counts(y_true, y_pred):
    """
    Compute per-class true positives, false positives and false negatives once,
    from the confusion matrix, so precision/recall/f1 all share one computation.

    Parameters:
    ----------
    - y_true: np.ndarray of shape (n_samples,), validated true labels
    - y_pred: np.ndarray of shape (n_samples,), validated predicted labels

    Returns:
    -------
    - labels: np.ndarray of shape (n_classes,), sorted union of the labels
    - tp, fp, fn: np.ndarray of shape (n_classes,) each

    Complexity:
    ----------
    - Time complexity: O(n log n + k^2) where k is the number of classes
    - Space complexity: O(n + k^2)
    """
    labels = np.union1d(y_true, y_pred)
    tp, fp, fn = _counts_from_matrix(confusion_matrix(y_true, y_pred, labels=labels))
    return labels, tp, fp, fn


def _counts_from_matrix(cm):
    """
    Per-class true positives, false positives and false negatives of a confusion matrix.

    Parameters:
    ----------
    - cm: np.ndarray of shape (n_classes, n_classes), rows true and columns predicted

    Returns:
    -------
    - tp, fp, fn: np.ndarray of shape (n_classes,) each

    Complexity:
    ----------
    - Time and space complexity: O(k^2) where k is the number of classes
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp, fp, fn


def _precision_per_class(tp, fp, fn):
    """TP / (TP + FP) per class, 0.0 where nothing was predicted as that class."""
    return _safe_divide(tp, tp + fp)


def _recall_per_class(tp, fp, fn):
    """TP / (TP + FN) per class, 0.0 where the class never occurred."""
    return _safe_divide(tp, tp + fn)


def _f1_per_class(tp, fp, fn):
    """2PR / (P + R) per class, 0.0 where precision and recall are both 0."""
    p = _precision_per_class(tp, fp, fn)
    r = _recall_per_class(tp, fp, fn)
    return _safe_divide(2 * p * r, p + r)


def _check_average(average):
    """Raise ValueError unless average is 'binary' or 'macro'."""
    if average not in _AVERAGES:
        raise ValueError("average must be 'binary' or 'macro'. Got: {!r}".format(average))


def _check_labels(labels):
    """
    Convert a user-given label order to an array.

    Raises:
    ------
    - ValueError: if labels is empty, not 1-dimensional or has duplicates
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty 1-dimensional array.")
    if np.unique(labels).size != labels.size:
        raise ValueError("labels must not contain duplicates.")
    return labels


def _select(labels, per_class, pos_label, average):
    """
    Reduce a per-class score to the final value for the requested average.

    Parameters:
    ----------
    - labels: np.ndarray of shape (n_classes,)
    - per_class: np.ndarray of shape (n_classes,), the score for each class
    - pos_label: the label treated as positive when average='binary'
    - average: str, 'binary' or 'macro'

    Returns:
    -------
    - float, the selected score

    Raises:
    ------
    - ValueError: if average is not 'binary' or 'macro', if average='binary' is used
    with more than 2 classes, or if pos_label is not one of 2 present classes

    Complexity:
    ----------
    - Time complexity: O(k) where k is the number of classes
    - Space complexity: O(k)
    """
    _check_average(average)

    if average == "macro":
        return float(np.mean(per_class))

    if labels.size > 2:
        raise ValueError("average='binary' needs at most 2 classes, found {}. "
                         "Use average='macro'.".format(labels.size))

    match = labels == pos_label
    if not np.any(match):
        if labels.size == 2:
            raise ValueError("pos_label={!r} is not a valid label. Labels: {}"
                             .format(pos_label, labels.tolist()))
        return 0.0  # only one class present and it is the negative class

    return float(per_class[match][0])


# ---------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Count how often each true label was predicted as each label.

    Parameters:
    ----------
    - y_true: array-like of shape (n_samples,), true labels (numbers or strings)
    - y_pred: array-like of shape (n_samples,), predicted labels
    - labels: array-like of shape (n_classes,), optional. The row/column order.
    Defaults to the sorted union of y_true and y_pred.

    Returns:
    -------
    - cm: np.ndarray of shape (n_classes, n_classes) with integer counts.
    Rows are true labels and columns are predicted labels, both in `labels` order.

    Raises:
    ------
    - ValueError: if the inputs fail validation (see _check_targets)
    - ValueError: if labels is empty, not 1-dimensional or has duplicates
    - ValueError: if y_true or y_pred contains a label missing from labels

    Complexity:
    ----------
    - Time complexity: O(n log n + k^2) where n is the number of samples and k is the number
    of classes; if labels is given the sort is skipped and it is O(n log k + k^2)
    - Space complexity: O(n + k^2)

    Note:
    ----
    - Fully vectorised: each (true, pred) pair is turned into one index
    true_index * k + pred_index and counted with np.bincount.
    """
    y_true, y_pred = _check_targets(y_true, y_pred)

    if labels is None:
        labels = np.union1d(y_true, y_pred)
    else:
        labels = _check_labels(labels)

    n_classes = labels.size
    true_idx = _label_indices(y_true, labels)
    pred_idx = _label_indices(y_pred, labels)

    counts = np.bincount(true_idx * n_classes + pred_idx, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes)


def accuracy(y_true, y_pred):
    """
    Fraction of predictions that exactly match the true labels.

    Parameters:
    ----------
    - y_true: array-like of shape (n_samples,), true labels
    - y_pred: array-like of shape (n_samples,), predicted labels

    Returns:
    -------
    - float in [0, 1]

    Raises:
    ------
    - ValueError: if the inputs fail validation (see _check_targets)

    Complexity:
    ----------
    - Time complexity: O(n) where n is the number of samples
    - Space complexity: O(n) for the boolean comparison array
    """
    y_true, y_pred = _check_targets(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred, pos_label=1, average="binary"):
    """
    Of everything predicted positive, the fraction that really was positive:
    TP / (TP + FP).

    Parameters:
    ----------
    - y_true: array-like of shape (n_samples,), true labels
    - y_pred: array-like of shape (n_samples,), predicted labels
    - pos_label: the positive class when average='binary' (default 1)
    - average: str, 'binary' (score for pos_label) or 'macro'
    (unweighted mean of the per-class scores)

    Returns:
    -------
    - float in [0, 1]

    Raises:
    ------
    - ValueError: if the inputs fail validation, average is invalid,
    average='binary' is used with more than 2 classes, or pos_label is invalid

    Complexity:
    ----------
    - Time complexity: O(n log n + k^2) where k is the number of classes
    - Space complexity: O(n + k^2)

    Note:
    ----
    - If nothing is predicted positive (TP + FP = 0) the score is defined as 0.0.
    """
    y_true, y_pred = _check_targets(y_true, y_pred)
    labels, tp, fp, fn = _class_counts(y_true, y_pred)
    return _select(labels, _precision_per_class(tp, fp, fn), pos_label, average)


def recall(y_true, y_pred, pos_label=1, average="binary"):
    """
    Of everything that really was positive, the fraction the model found:
    TP / (TP + FN).

    Parameters:
    ----------
    - y_true: array-like of shape (n_samples,), true labels
    - y_pred: array-like of shape (n_samples,), predicted labels
    - pos_label: the positive class when average='binary' (default 1)
    - average: str, 'binary' or 'macro'

    Returns:
    -------
    - float in [0, 1]

    Raises:
    ------
    - ValueError: same conditions as precision

    Complexity:
    ----------
    - Time complexity: O(n log n + k^2) where k is the number of classes
    - Space complexity: O(n + k^2)

    Note:
    ----
    - If there are no true positives to find (TP + FN = 0) the score is defined as 0.0.
    """
    y_true, y_pred = _check_targets(y_true, y_pred)
    labels, tp, fp, fn = _class_counts(y_true, y_pred)
    return _select(labels, _recall_per_class(tp, fp, fn), pos_label, average)


def f1(y_true, y_pred, pos_label=1, average="binary"):
    """
    Harmonic mean of precision and recall: 2 * P * R / (P + R).

    Parameters:
    ----------
    - y_true: array-like of shape (n_samples,), true labels
    - y_pred: array-like of shape (n_samples,), predicted labels
    - pos_label: the positive class when average='binary' (default 1)
    - average: str, 'binary' or 'macro' (mean of the per-class F1 scores)

    Returns:
    -------
    - float in [0, 1]

    Raises:
    ------
    - ValueError: same conditions as precision

    Complexity:
    ----------
    - Time complexity: O(n log n + k^2) where k is the number of classes
    - Space complexity: O(n + k^2)

    Note:
    ----
    - If precision and recall are both 0 the score is defined as 0.0.
    """
    y_true, y_pred = _check_targets(y_true, y_pred)
    labels, tp, fp, fn = _class_counts(y_true, y_pred)
    return _select(labels, _f1_per_class(tp, fp, fn), pos_label, average)


# ---------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------

def mse(y_true, y_pred):
    """
    Mean squared error: the average of (y_true - y_pred) ** 2.

    Parameters:
    ----------
    - y_true: array-like of shape (n_samples,), true numeric targets
    - y_pred: array-like of shape (n_samples,), predicted numeric targets

    Returns:
    -------
    - float >= 0

    Raises:
    ------
    - ValueError: if the inputs fail validation, are not numeric, or contain inf

    Complexity:
    ----------
    - Time complexity: O(n) where n is the number of samples
    - Space complexity: O(n) for the difference array
    """
    y_true, y_pred = _check_targets(y_true, y_pred)
    y_true = _to_float(y_true, "y_true")
    y_pred = _to_float(y_pred, "y_pred")

    diff = y_true - y_pred
    return float(np.mean(diff * diff))


# ---------------------------------------------------------------------
# Bonus: ROC curve and AUC (binary classification)
# ---------------------------------------------------------------------

def roc_curve(y_true, y_score, pos_label=1):
    """
    Receiver operating characteristic curve for a binary classifier.

    Parameters:
    ----------
    - y_true: array-like of shape (n_samples,), true binary labels
    - y_score: array-like of shape (n_samples,), scores where higher means
    "more likely positive" (e.g. predicted probabilities)
    - pos_label: the positive class (default 1)

    Returns:
    -------
    - fpr: np.ndarray of shape (n_thresholds,), false positive rate at each threshold
    - tpr: np.ndarray of shape (n_thresholds,), true positive rate at each threshold
    - thresholds: np.ndarray of shape (n_thresholds,), decreasing; the first is inf
    so the curve starts at (0, 0)

    Raises:
    ------
    - ValueError: if the inputs fail validation, y_score is not numeric/finite,
    y_true has more than 2 classes, or y_true lacks positives or negatives

    Complexity:
    ----------
    - Time complexity: O(n log n) for the sort
    - Space complexity: O(n)

    Note:
    ----
    - Tied scores are grouped into one threshold, so ties give one diagonal step
    rather than an arbitrary staircase.
    """
    y_true, y_score = _check_targets(y_true, y_score)
    y_score = _to_float(y_score, "y_score")

    if np.unique(y_true).size > 2:
        raise ValueError("roc_curve is for binary classification; y_true has more than 2 classes.")

    is_pos = y_true == pos_label
    n_pos = int(is_pos.sum())
    n_neg = is_pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive (pos_label={!r}) and negative samples."
                         .format(pos_label))

    order = np.argsort(-y_score, kind="stable")
    sorted_scores = y_score[order]
    sorted_pos = is_pos[order]

    tps = np.cumsum(sorted_pos)
    fps = np.cumsum(~sorted_pos)

    # keep only the last position of each run of equal scores
    last_of_run = np.r_[np.diff(sorted_scores) != 0, True]

    tpr = np.r_[0.0, tps[last_of_run] / n_pos]
    fpr = np.r_[0.0, fps[last_of_run] / n_neg]
    thresholds = np.r_[np.inf, sorted_scores[last_of_run]]
    return fpr, tpr, thresholds


def auc(x, y):
    """
    Area under a curve using the trapezoidal rule (e.g. auc(fpr, tpr) for ROC AUC).

    Parameters:
    ----------
    - x: array-like of shape (n_points,), monotonic (increasing or decreasing)
    - y: array-like of shape (n_points,)

    Returns:
    -------
    - float, the area

    Raises:
    ------
    - ValueError: if the inputs fail validation, are not numeric/finite,
    have fewer than 2 points, or x is not monotonic

    Complexity:
    ----------
    - Time complexity: O(n) where n is the number of points
    - Space complexity: O(n)
    """
    x, y = _check_targets(x, y)
    x = _to_float(x, "x")
    y = _to_float(y, "y")

    if x.size < 2:
        raise ValueError("auc needs at least 2 points. Got {}".format(x.size))

    dx = np.diff(x)
    if np.all(dx >= 0):
        direction = 1.0
    elif np.all(dx <= 0):
        direction = -1.0
    else:
        raise ValueError("x must be monotonic (increasing or decreasing).")

    return float(direction * np.sum(dx * (y[1:] + y[:-1]) / 2.0))


# ---------------------------------------------------------------------
# Streaming metrics: update(y_true_chunk, y_pred_chunk) / reset() / result()
# ---------------------------------------------------------------------

class StreamingConfusionMatrix:
    """
    Confusion matrix accumulated over chunks, optionally over a rolling window.

    Parameters:
    ----------
    - labels: array-like of shape (n_classes,), optional. Fixed row/column order; a label
    outside it raises. If None, classes are added as they first appear and the result is
    in sorted label order.
    - window: int >= 1 or None. If given, only the last ``window`` chunks (update calls)
    are counted; None counts everything since the last reset.

    Attributes:
    ----------
    - labels_: np.ndarray of shape (n_classes,), the row/column labels of result(). Without
    fixed labels it holds every class seen since the last reset (sorted), including classes
    whose chunks have left the window (their counts are then 0).

    Raises:
    ------
    - ValueError: if labels is empty, not 1-D or has duplicates, or window is not a
    positive integer

    Complexity:
    ----------
    - Space complexity: O(k^2) for k classes, plus O(k^2) per chunk kept in the window

    Note:
    ----
    - Each chunk is counted with the vectorised 2.1 confusion_matrix (one np.bincount) and
    added into the running total. With a window, each chunk's small matrix is kept and
    subtracted when it leaves; integer counts make this exact.
    """

    def __init__(self, labels=None, window=None):
        self._fixed = None if labels is None else _check_labels(labels)
        self.window = None if window is None else check_positive_int(window, "window")
        self.reset()

    def reset(self):
        """Forget all counts (fixed labels are kept). Returns self."""
        self.labels_ = self._fixed
        k = 0 if self._fixed is None else self._fixed.size
        self._total = np.zeros((k, k), dtype=np.int64)
        self._chunks = deque()
        return self

    def update(self, y_true, y_pred):
        """
        Add one chunk of true and predicted labels.

        Parameters:
        ----------
        - y_true: array-like of shape (n_samples,), true labels (numbers or strings)
        - y_pred: array-like of shape (n_samples,), predicted labels

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if the chunk fails validation (see _check_targets), or contains a
        label outside a fixed ``labels``
        - TypeError: if a chunk's labels are strings and earlier labels numbers (or the
        other way round)

        Complexity:
        ----------
        - Time complexity: O(n log k + k^2) for n samples and k classes
        - Space complexity: O(k^2) per chunk kept in the window
        """
        y_true, y_pred = _check_targets(y_true, y_pred)
        if self._fixed is not None:
            chunk_labels = self._fixed
        else:
            chunk_labels = np.union1d(y_true, y_pred)
            self._add_labels(chunk_labels)
        chunk_cm = confusion_matrix(y_true, y_pred, labels=chunk_labels)

        self._add(chunk_labels, chunk_cm, sign=1)
        if self.window is not None:
            self._chunks.append((chunk_labels, chunk_cm))
            if len(self._chunks) > self.window:
                self._add(*self._chunks.popleft(), sign=-1)
        return self

    def _add_labels(self, chunk_labels):
        """Merge new labels into the sorted labels_, moving the counts to match."""
        if self.labels_ is None:
            self.labels_ = chunk_labels
            self._total = np.zeros((chunk_labels.size,) * 2, dtype=np.int64)
            return
        if (self.labels_.dtype.kind in "US") != (chunk_labels.dtype.kind in "US"):
            # np.union1d would silently turn the number 1 into the string "1"
            raise TypeError("Mixed label types across chunks, use all numbers or all strings.")
        labels = np.union1d(self.labels_, chunk_labels)
        if labels.size == self.labels_.size:
            return
        total = np.zeros((labels.size,) * 2, dtype=np.int64)
        old = np.searchsorted(labels, self.labels_)
        total[np.ix_(old, old)] = self._total
        self.labels_, self._total = labels, total

    def _add(self, chunk_labels, chunk_cm, sign):
        """Add (sign=1) or remove (sign=-1) one chunk's counts from the running total."""
        pos = _label_indices(chunk_labels, self.labels_)
        self._total[np.ix_(pos, pos)] += sign * chunk_cm

    def result(self):
        """
        Return the accumulated confusion matrix.

        Returns:
        -------
        - np.ndarray of int64, shape (n_classes, n_classes): rows are true labels, columns
        predicted labels, both in ``labels_`` order (a copy). Shape (0, 0) before any data
        when no labels were given.
        """
        return self._total.copy()


class _StreamingClassificationMetric:
    """
    Shared base for streaming Accuracy / Precision / Recall / F1.

    Each metric keeps a StreamingConfusionMatrix (with the same window) and computes its
    score from the counts with the 2.1 formulas, so a streamed result equals the 2.1 batch
    function on all the data seen (or on the chunks in the window). Subclasses implement
    _score(labels, cm), called only when at least one sample has been seen.

    result() is NaN while no samples have been seen.
    """

    def __init__(self, window=None):
        self._cm = StreamingConfusionMatrix(window=window)
        self.window = self._cm.window

    def update(self, y_true, y_pred):
        """
        Add one chunk; see StreamingConfusionMatrix.update. Returns self.
        """
        self._cm.update(y_true, y_pred)
        return self

    def reset(self):
        """Forget all data seen so far. Returns self."""
        self._cm.reset()
        return self

    def result(self):
        """
        Return the score over all data seen (or the window).

        Returns:
        -------
        - float in [0, 1], or NaN if no samples have been seen yet (or in the window)

        Raises:
        ------
        - ValueError: from Precision / Recall / F1 with average='binary' when more than
        2 classes are present
        """
        cm = self._cm.result()
        if cm.sum() == 0:
            return float("nan")
        return self._score(self._cm.labels_, cm)

    def _score(self, labels, cm):
        raise NotImplementedError


class Accuracy(_StreamingClassificationMetric):
    """
    Streaming accuracy: correct predictions / samples seen.

    Parameters:
    ----------
    - window: int >= 1 or None, number of most recent chunks to score (None = all)
    """

    def _score(self, labels, cm):
        return float(np.trace(cm) / cm.sum())


class _PerClassMetric(_StreamingClassificationMetric):
    """
    Shared base for Precision / Recall / F1 (pos_label and average as in 2.1).

    Parameters:
    ----------
    - pos_label: the positive class when average='binary' (default 1)
    - average: 'binary' (score of pos_label) or 'macro' (mean over the classes seen)
    - window: int >= 1 or None

    Raises:
    ------
    - ValueError: if average is not 'binary' or 'macro' (at construction), or, from
    result(), if average='binary' and more than 2 classes have been seen

    Note:
    ----
    - A class not seen yet does not take part in the macro average; a 0/0 score is 0.0,
    as in the 2.1 functions.
    - With a window, only classes present in the window count (as the batch function on
    those chunks would see them).
    """

    _per_class = None   # set by subclasses to one of the 2.1 per-class formulas

    def __init__(self, pos_label=1, average="binary", window=None):
        _check_average(average)
        super().__init__(window=window)
        self.pos_label = pos_label
        self.average = average

    def _score(self, labels, cm):
        # drop classes with no samples in the counted chunks (e.g. left the window)
        present = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
        cm = cm[np.ix_(present, present)]
        per_class = self._per_class(*_counts_from_matrix(cm))
        return _select(labels[present], per_class, self.pos_label, self.average)


class Precision(_PerClassMetric):
    """Streaming precision TP / (TP + FP); parameters as _PerClassMetric."""

    _per_class = staticmethod(_precision_per_class)


class Recall(_PerClassMetric):
    """Streaming recall TP / (TP + FN); parameters as _PerClassMetric."""

    _per_class = staticmethod(_recall_per_class)


class F1(_PerClassMetric):
    """Streaming F1 = 2PR / (P + R); parameters as _PerClassMetric."""

    _per_class = staticmethod(_f1_per_class)
