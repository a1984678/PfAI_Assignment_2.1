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
    StreamingAUC,
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


# ---------------------------------------------------------------------
# StreamingAUC (score histogram over [0, 1], see the AUC spike)
# ---------------------------------------------------------------------

def _batch_auc(y_true, y_score, pos_label=1):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    return metrics.auc(fpr, tpr)


def _leaf_scores(n=300, seed=3):
    """Tree-like scores: few distinct probabilities (many ties), incl. exactly 0 and 1."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    leaves = np.array([0.0, 0.2, 0.25, 0.5, 0.75, 0.9, 1.0])
    shift = np.where(y == 1, 2, 0)
    return y, leaves[np.clip(rng.integers(0, 5, n) + shift, 0, 6)]


@pytest.mark.parametrize("chunk_size", [1, 50])
def test_auc_exact_when_scores_fall_in_distinct_bins(chunk_size):
    y, s = _leaf_scores()
    auc = StreamingAUC()
    for yt, ys in _chunks(y, s, chunk_size):
        assert auc.update(yt, ys) is auc
    assert auc.result() == pytest.approx(_batch_auc(y, s), abs=1e-12)

    # string labels with pos_label, and a rolling window of the last 2 chunks
    labels = np.where(y == 1, "yes", "no")
    windowed = StreamingAUC(pos_label="yes", window=2)
    chunks = list(_chunks(labels, s, 50))
    for yt, ys in chunks:
        windowed.update(yt, ys)
    recent_y = np.concatenate([c[0] for c in chunks[-2:]])
    recent_s = np.concatenate([c[1] for c in chunks[-2:]])
    assert windowed.result() == pytest.approx(_batch_auc(recent_y, recent_s, "yes"),
                                              abs=1e-12)


def test_auc_continuous_scores_within_reported_error_bound():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 2000)
    s = 1 / (1 + np.exp(-(0.3 * rng.normal(size=2000) + 0.2 * (2 * y - 1))))  # crowded
    coarse = StreamingAUC(bins=20)
    for yt, ys in _chunks(y, s, 100):
        coarse.update(yt, ys)
    exact = _batch_auc(y, s)
    assert 0 < coarse.max_error() < 0.5
    assert abs(coarse.result() - exact) <= coarse.max_error()

    fine = StreamingAUC(bins=1000).update(y, s)          # more bins -> tighter bound
    assert fine.max_error() < coarse.max_error()
    assert abs(fine.result() - exact) <= fine.max_error()


def test_auc_edge_cases_undefined_and_invalid_input():
    auc = StreamingAUC()
    assert np.isnan(auc.result()) and np.isnan(auc.max_error())    # nothing seen yet
    auc.update([1, 1], [0.2, 0.9])
    assert np.isnan(auc.result())                                    # no negatives yet
    auc.update([0], [0.1])
    assert auc.result() == 1.0
    assert np.isnan(auc.reset().result())

    with pytest.raises(ValueError, match=r"y_score must be in \[0, 1\]"):
        auc.update([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError, match="more than 2 classes"):
        auc.update([0, 1], [0.1, 0.2]).update([2], [0.3])
    with pytest.raises(ValueError, match="same length"):
        auc.update([0, 1], [0.5])
    with pytest.raises(ValueError, match="pos_label=1 is not a valid label"):
        StreamingAUC().update(["no", "yes"], [0.2, 0.8])   # would otherwise be NaN forever
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        StreamingAUC(bins=0)
    with pytest.raises(ValueError, match="window must be a positive integer"):
        StreamingAUC(window=-1)
