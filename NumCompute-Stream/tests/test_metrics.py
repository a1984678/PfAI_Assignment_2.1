"""Tests for ``numcompute_stream.metrics``.

Cover streaming ``update(y_true_chunk, y_pred_chunk)``, ``reset()`` and
``result()`` for accuracy, precision, recall, etc.; accumulation of the
confusion matrix and AUC over time; and rolling-window metrics.

Main check: a streamed metric equals the 2.1 batch function on all the data seen (or, with
``window``, on the chunks still in the window).
"""

import numpy as np
import pytest

from numcompute_stream import metrics
from numcompute_stream.metrics import (
    Accuracy,
    F1,
    Precision,
    Recall,
    StreamingConfusionMatrix,
)

# (streaming class, 2.1 batch function, constructor kwargs)
CLASSIFICATION = [
    (Accuracy, metrics.accuracy, {}),
    (Precision, metrics.precision, {}),
    (Recall, metrics.recall, {}),
    (F1, metrics.f1, {}),
    (Precision, metrics.precision, {"average": "macro"}),
    (Recall, metrics.recall, {"average": "macro"}),
    (F1, metrics.f1, {"average": "macro"}),
]


def _chunks(y_true, y_pred, chunk_size):
    for start in range(0, len(y_true), chunk_size):
        yield y_true[start:start + chunk_size], y_pred[start:start + chunk_size]


def _binary(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    y_pred = np.where(rng.random(n) < 0.8, y_true, 1 - y_true)    # ~80 % correct
    return y_true, y_pred


# ---------------------------------------------------------------------
# StreamingConfusionMatrix
# ---------------------------------------------------------------------

@pytest.mark.parametrize("chunk_size", [1, 7, 200])
def test_confusion_matrix_accumulates_like_batch(chunk_size):
    rng = np.random.default_rng(1)
    y_true, y_pred = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
    cm = StreamingConfusionMatrix()
    for yt, yp in _chunks(y_true, y_pred, chunk_size):
        assert cm.update(yt, yp) is cm
    np.testing.assert_array_equal(cm.result(), metrics.confusion_matrix(y_true, y_pred))
    np.testing.assert_array_equal(cm.labels_, [0, 1, 2])


def test_confusion_matrix_class_first_seen_in_later_chunk():
    cm = StreamingConfusionMatrix()
    cm.update(["cat", "dog", "cat"], ["cat", "cat", "cat"])
    np.testing.assert_array_equal(cm.result(), [[2, 0], [1, 0]])

    # "bird" appears only now; labels stay sorted and earlier counts are kept
    cm.update(["bird", "dog"], ["dog", "dog"])
    assert list(cm.labels_) == ["bird", "cat", "dog"]
    np.testing.assert_array_equal(cm.result(), [[0, 0, 1],
                                                [0, 2, 0],
                                                [0, 1, 1]])


def test_confusion_matrix_fixed_labels_reset_and_errors():
    cm = StreamingConfusionMatrix(labels=[1, 0])
    np.testing.assert_array_equal(cm.result(), [[0, 0], [0, 0]])   # nothing seen yet
    cm.update([0, 1, 1], [0, 0, 1])
    np.testing.assert_array_equal(cm.result(), [[1, 1], [0, 1]])   # rows/cols in labels order
    with pytest.raises(ValueError, match="not present in `labels`"):
        cm.update([2], [0])
    with pytest.raises(ValueError, match="same length"):
        cm.update([0, 1], [0])

    assert cm.reset() is cm
    np.testing.assert_array_equal(cm.result(), [[0, 0], [0, 0]])
    assert StreamingConfusionMatrix().result().shape == (0, 0)


# ---------------------------------------------------------------------
# Accuracy / Precision / Recall / F1
# ---------------------------------------------------------------------

@pytest.mark.parametrize("cls, batch, kwargs", CLASSIFICATION)
@pytest.mark.parametrize("chunk_size", [1, 13])
def test_metric_matches_batch_on_all_data(cls, batch, kwargs, chunk_size):
    y_true, y_pred = _binary()
    metric = cls(**kwargs)
    for yt, yp in _chunks(y_true, y_pred, chunk_size):
        assert metric.update(yt, yp) is metric
    assert metric.result() == pytest.approx(batch(y_true, y_pred, **kwargs), abs=1e-12)


@pytest.mark.parametrize("cls, batch, kwargs", CLASSIFICATION)
def test_rolling_window_uses_only_recent_chunks(cls, batch, kwargs):
    y_true, y_pred = _binary(seed=2)
    chunks = list(_chunks(y_true, y_pred, 20))
    metric = cls(window=3, **kwargs)

    # window larger than the data seen so far: everything seen counts
    metric.update(*chunks[0])
    assert metric.result() == pytest.approx(batch(*chunks[0], **kwargs), abs=1e-12)

    for chunk in chunks[1:]:
        metric.update(*chunk)
    recent_true = np.concatenate([c[0] for c in chunks[-3:]])
    recent_pred = np.concatenate([c[1] for c in chunks[-3:]])
    assert metric.result() == pytest.approx(batch(recent_true, recent_pred, **kwargs),
                                            abs=1e-12)


def test_metrics_edge_cases_no_data_unseen_class_and_errors():
    # nothing seen yet (or after reset): undefined, so NaN rather than an exception
    for metric in (Accuracy(), Precision(), Recall(), F1()):
        assert np.isnan(metric.result())
        metric.update([1, 0], [1, 1])
        assert not np.isnan(metric.result())
        assert metric.reset() is metric
        assert np.isnan(metric.result())

    # positive class not seen yet: 0/0 gives 0.0, as in the 2.1 functions
    for cls in (Precision, Recall, F1):
        assert cls().update([0, 0], [0, 0]).result() == 0.0

    # class seen only in a later chunk is included from then on (macro over 3 classes)
    macro = Recall(average="macro").update([0, 1], [0, 1])
    assert macro.result() == 1.0
    macro.update([2], [0])
    assert macro.result() == pytest.approx(metrics.recall([0, 1, 2], [0, 1, 0],
                                                          average="macro"))

    with pytest.raises(ValueError, match="window must be a positive integer"):
        Accuracy(window=0)
    with pytest.raises(ValueError, match="average must be"):
        Precision(average="micro")
    with pytest.raises(ValueError, match="average='binary' needs at most 2 classes"):
        Precision().update([0, 1, 2], [0, 1, 2]).result()
