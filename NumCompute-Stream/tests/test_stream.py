"""Tests for ``numcompute_stream.stream.StreamTrainer``.

Cover ``.fit_chunk(X, y)`` and ``.score_chunk(X, y)``, and logging of
per-chunk metrics, memory footprint, and cumulative accuracy.
"""

import numpy as np
import pytest

from numcompute_stream import metrics
from numcompute_stream.io import iter_chunks
from numcompute_stream.metrics import F1, Accuracy, StreamingAUC
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer, memory_footprint
from numcompute_stream.tree import DecisionTreeClassifier


def _chunks(n=600, chunk_size=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > X[:, 1]).astype(int)
    return list(iter_chunks(X, y, chunk_size=chunk_size))


def _model(max_depth=3, buffer_size=100):
    return Pipeline([("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=max_depth,
                                                      buffer_size=buffer_size))])


def test_test_then_train_logs_chunk_and_cumulative_metrics():
    chunks = _chunks()
    trainer = StreamTrainer(_model(), metrics={"f1": F1(average="macro"),
                                               "auc": StreamingAUC(pos_label=1),
                                               "rolling_accuracy": Accuracy(window=3)})
    ref, ref_auc = _model(), StreamingAUC(pos_label=1)     # same model, driven by hand
    scored = []
    for i, (X_chunk, y_chunk) in enumerate(chunks):
        row = trainer.score_chunk(X_chunk, y_chunk)        # score first, then train
        if i < 2:                                          # no tree until 100 rows buffered
            assert np.isnan(row["accuracy"]) and np.isnan(row["cumulative_accuracy"])
        else:
            pred = ref.predict(X_chunk)
            ref_auc.update(y_chunk, ref.predict_proba(X_chunk)[:, 1])
            scored.append((y_chunk, pred))
            assert row["accuracy"] == metrics.accuracy(y_chunk, pred)
            np.testing.assert_array_equal(trainer.last_y_pred_, pred)
            np.testing.assert_array_equal(trainer.last_y_true_, y_chunk)
        assert trainer.fit_chunk(X_chunk, y_chunk) is trainer
        ref.partial_fit(X_chunk, y_chunk)

    y_true = np.concatenate([t for t, _ in scored])
    y_pred = np.concatenate([p for _, p in scored])
    assert row["cumulative_accuracy"] == pytest.approx(metrics.accuracy(y_true, y_pred))
    assert row["f1"] == pytest.approx(metrics.f1(y_true, y_pred, average="macro"))
    assert row["auc"] == ref_auc.result()                  # fed the P(y=1) column
    recent = scored[-3:]
    assert row["rolling_accuracy"] == pytest.approx(metrics.accuracy(
        np.concatenate([t for t, _ in recent]), np.concatenate([p for _, p in recent])))

    history = trainer.history("cumulative_accuracy")      # one value per scored chunk
    assert history.shape == (12,) and np.isnan(history[:2]).all()
    assert history[-1] == row["cumulative_accuracy"]


def test_fit_log_records_samples_time_and_memory():
    trainer = StreamTrainer(_model(max_depth=2, buffer_size=200))
    for X_chunk, y_chunk in _chunks()[:4]:
        trainer.fit_chunk(X_chunk, y_chunk)

    np.testing.assert_array_equal(trainer.history("n_samples_seen"), [50, 100, 150, 200])
    fit_time = trainer.history("fit_time")
    assert fit_time.shape == (4,) and np.all(fit_time >= 0)

    # the tree's buffer grows by one scaled chunk (50x3 float64 + 50 int64 labels) per
    # call, then the 4th chunk fills it (200 rows): retrain, buffer cleared, memory drops
    model_bytes = trainer.history("model_bytes")
    np.testing.assert_array_equal(np.diff(model_bytes[:3]), [1600, 1600])
    assert model_bytes[3] < model_bytes[2]


def test_memory_footprint_counts_each_array_buffer_once():
    a, b = np.zeros(10), np.ones((2, 3), dtype=np.int32)

    class Box:
        pass

    box = Box()
    box.data = [a, a, {"b": b}, (5, "text")]     # shared array counted once
    box.view = a[:5]                             # a view owns no memory of its own
    box.me = box                                 # cycles are fine
    assert memory_footprint(box) == 80 + 24
    assert memory_footprint(a[2:]) == 80         # the whole underlying buffer
    assert memory_footprint(None) == 0


def test_score_before_training_and_invalid_setup():
    X_chunk, y_chunk = _chunks()[0]
    trainer = StreamTrainer(DecisionTreeClassifier())
    row = trainer.score_chunk(X_chunk, y_chunk)            # no tree yet: logged, not raised
    assert np.isnan(row["accuracy"]) and np.isnan(row["cumulative_accuracy"])
    assert trainer.last_y_pred_ is None
    assert trainer.history("accuracy").shape == (1,)
    assert trainer.history("fit_time").shape == (0,)

    trainer.model.fit(X_chunk, y_chunk)                    # fitted outside the trainer
    with pytest.raises(ValueError, match="features"):     # real errors still propagate
        trainer.score_chunk(X_chunk[:, :2], y_chunk)
    with pytest.raises(KeyError, match="nope"):
        trainer.history("nope")

    with pytest.raises(TypeError, match="does not support partial_fit"):
        StreamTrainer(object())
    with pytest.raises(TypeError, match="update and result"):
        StreamTrainer(DecisionTreeClassifier(), metrics={"x": object()})
    with pytest.raises(ValueError, match="reserved"):
        StreamTrainer(DecisionTreeClassifier(), metrics={"accuracy": Accuracy()})
