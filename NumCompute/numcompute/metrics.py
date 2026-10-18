"""
metrics.py

Classification:
- accuracy
- precision
- recall
- f1
- confusion_matrix

Regression:
- mse(y_true, y_pred)

Bonus:
- roc_curve and auc for binary classification
"""
import numpy as np

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
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return labels, tp, fp, fn


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
    if average not in _AVERAGES:
        raise ValueError("average must be 'binary' or 'macro'. Got: {!r}".format(average))

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
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("labels must be a non-empty 1-dimensional array.")
        if np.unique(labels).size != labels.size:
            raise ValueError("labels must not contain duplicates.")

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
    labels, tp, fp, _ = _class_counts(y_true, y_pred)
    return _select(labels, _safe_divide(tp, tp + fp), pos_label, average)


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
    labels, tp, _, fn = _class_counts(y_true, y_pred)
    return _select(labels, _safe_divide(tp, tp + fn), pos_label, average)


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
    p = _safe_divide(tp, tp + fp)
    r = _safe_divide(tp, tp + fn)
    return _select(labels, _safe_divide(2 * p * r, p + r), pos_label, average)


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
