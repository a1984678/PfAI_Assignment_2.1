"""Streaming training management.

- Implements ``StreamTrainer``, managing model + pipeline + logging
- Supports ``.fit_chunk(X, y)`` and ``.score_chunk(X, y)`` logic
- Logs per-chunk metrics, memory footprint, and cumulative accuracy
"""


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
    """
    raise NotImplementedError


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
    - metrics: dict of name -> streaming metric (update / result), e.g. F1(average="macro"),
    StreamingAUC(pos_label=1), Accuracy(window=5). Updated on every scored chunk and logged
    by name. A StreamingAUC gets the predict_proba column of its pos_label. Default None.

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
        self.model = model
        self.metrics = metrics

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
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
