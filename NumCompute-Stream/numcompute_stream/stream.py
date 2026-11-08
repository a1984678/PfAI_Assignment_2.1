"""Streaming training management.

- Implements ``StreamTrainer``, managing model + pipeline + logging
- Supports ``.fit_chunk(X, y)`` and ``.score_chunk(X, y)`` logic
- Logs per-chunk metrics, memory footprint, and cumulative accuracy

Streaming API
-------------
- ``StreamTrainer(model, metrics=None)``: ``score_chunk`` then ``fit_chunk`` on every chunk
  (test-then-train); ``history(key)`` returns one logged value over time as an array,
  ready for ``visualise.plot_metric_over_time``.
- ``memory_footprint(obj)``: bytes of NumPy array data held by a model or pipeline.
"""

import time
import types
from collections import deque

import numpy as np

from . import metrics as _metrics
from .metrics import Accuracy, StreamingAUC
from .pipeline import _supports_partial_fit

_SCORE_KEYS = ("accuracy", "cumulative_accuracy")
_FIT_KEYS = ("n_samples_seen", "fit_time", "model_bytes")
_SKIP = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
         types.MethodType)


def memory_footprint(obj):
    """
    Bytes of NumPy array data reachable from obj (a model, pipeline, list, dict, ...).

    Follows lists, tuples, sets, dicts, deques and object attributes. Each underlying
    array buffer is counted once: shared arrays and views of the same array add nothing
    extra, and a view counts its whole base buffer. Python scalars and strings count 0, so
    this measures data held in arrays (e.g. a tree's buffer), not the whole process.

    Parameters:
    ----------
    - obj: any object

    Returns:
    -------
    - int, number of bytes

    Complexity:
    ----------
    - Time: O(number of reachable objects); each is visited once, so cycles are safe
    """
    seen, buffers, total = set(), set(), 0
    stack = [obj]
    while stack:
        item = stack.pop()
        if id(item) in seen or isinstance(item, _SKIP):
            continue
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            base = item
            while isinstance(base.base, np.ndarray):     # a view: count the owner instead
                base = base.base
            if id(base) not in buffers:
                buffers.add(id(base))
                total += base.nbytes
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset, deque)):
            stack.extend(item)
        elif hasattr(item, "__dict__"):
            stack.extend(vars(item).values())
    return total


def _final_step(model):
    """The estimator at the end of a (possibly nested) Pipeline, or the model itself."""
    while hasattr(model, "steps") and model.steps:
        model = model.steps[-1][1]
    return model


class StreamTrainer:
    """
    Run a streaming model chunk by chunk and log what happens.

    Typical use is test-then-train (prequential): score each chunk with the current model,
    then train on it, so every score is on data the model has not seen yet.

    >>> trainer = StreamTrainer(pipe, metrics={"f1": F1(average="macro")})
    >>> for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=100):
    ...     trainer.score_chunk(X_chunk, y_chunk)
    ...     trainer.fit_chunk(X_chunk, y_chunk)
    >>> plot_metric_over_time(trainer.history("cumulative_accuracy"), ...)

    Parameters:
    ----------
    - model: Pipeline or estimator that supports partial_fit (and predict; predict_proba
    for a StreamingAUC metric)
    - metrics: dict of name -> streaming metric (update / result returning a number), e.g.
    F1(average="macro"), StreamingAUC(pos_label=1), Accuracy(window=5). Updated on every
    scored chunk and logged by name. A StreamingAUC gets the predict_proba column of its
    pos_label (all zeros while that class has not been seen). Default None.

    Logged per call (read with history(key)):
    ----------
    - score_chunk: "accuracy" (this chunk), "cumulative_accuracy", and every metric name
    - fit_chunk: "n_samples_seen", "fit_time" (seconds), "model_bytes" (memory_footprint)

    Attributes:
    ----------
    - model, metrics
    - last_y_true_, last_y_pred_: labels and predictions of the latest scored chunk (None
    until the model can predict)

    Raises:
    ------
    - TypeError: if the model does not support partial_fit, or a metric lacks update/result
    - ValueError: if a metric name is one of the logged names above
    """

    def __init__(self, model, metrics=None):
        if not _supports_partial_fit(model):
            raise TypeError(f"{type(model).__name__} does not support partial_fit, so it "
                            "can't be trained on a stream.")
        metrics = dict(metrics or {})
        for name, metric in metrics.items():
            if name in _SCORE_KEYS + _FIT_KEYS:
                raise ValueError(f"Metric name {name!r} is reserved: StreamTrainer already "
                                 f"logs {_SCORE_KEYS + _FIT_KEYS}.")
            if not (hasattr(metric, "update") and hasattr(metric, "result")):
                raise TypeError(f"Metric {name!r} must have update and result methods.")
        self.model = model
        self.metrics = metrics
        self._cumulative = Accuracy()
        self._score_log = {key: [] for key in (*_SCORE_KEYS, *metrics)}
        self._fit_log = {key: [] for key in _FIT_KEYS}
        self._n_seen = 0
        self.last_y_true_ = self.last_y_pred_ = None

    def fit_chunk(self, X, y):
        """
        Train the model on one chunk (model.partial_fit) and log time and memory.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,)

        Returns:
        -------
        - self

        Raises:
        ------
        - anything model.partial_fit raises (nothing is logged then)
        """
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        fit_time = time.perf_counter() - start
        self._n_seen += len(y)
        for key, value in zip(_FIT_KEYS, (self._n_seen, fit_time,
                                          memory_footprint(self.model))):
            self._fit_log[key].append(value)
        return self

    def _proba_of(self, X, pos_label):
        """predict_proba column of pos_label, or zeros if the model has not seen it."""
        classes = list(_final_step(self.model).classes_)
        if pos_label not in classes:
            return np.zeros(len(X))
        return self.model.predict_proba(X)[:, classes.index(pos_label)]

    def score_chunk(self, X, y):
        """
        Score the current model on one chunk, update the metrics and log them.

        If the model cannot predict yet (it raises RuntimeError, e.g. the tree's buffer is
        not full), the row is logged with NaN and no metric is updated.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,)

        Returns:
        -------
        - dict, the logged row: name -> float

        Raises:
        ------
        - anything other than RuntimeError that the model raises (e.g. wrong feature count)
        """
        try:
            y_pred = self.model.predict(X)
        except RuntimeError:                       # not fitted yet: log a gap, not an error
            row = dict.fromkeys(self._score_log, np.nan)
        else:
            y = np.asarray(y)
            self._cumulative.update(y, y_pred)
            row = {"accuracy": float(_metrics.accuracy(y, y_pred)),
                   "cumulative_accuracy": float(self._cumulative.result())}
            for name, metric in self.metrics.items():
                if isinstance(metric, StreamingAUC):
                    metric.update(y, self._proba_of(X, metric.pos_label))
                else:
                    metric.update(y, y_pred)
                row[name] = float(metric.result())
            self.last_y_true_, self.last_y_pred_ = y, y_pred
        for key, value in row.items():
            self._score_log[key].append(value)
        return row

    def history(self, key):
        """
        One logged value over time, oldest first.

        Parameters:
        ----------
        - key: str, a name logged by score_chunk or fit_chunk

        Returns:
        -------
        - np.ndarray of shape (n_calls,), float64 (empty before the first call)

        Raises:
        ------
        - KeyError: if key is not a logged name
        """
        for log in (self._score_log, self._fit_log):
            if key in log:
                return np.array(log[key], dtype=np.float64)
        raise KeyError(f"{key!r} is not logged. Available: "
                       f"{[*self._score_log, *self._fit_log]}")
