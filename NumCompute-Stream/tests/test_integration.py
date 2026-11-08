"""End-to-end streaming tests.

Load CSV data, split into chunks, train a pipeline incrementally with
``.partial_fit()`` and evaluate with streaming metrics.

These tests join modules that each have their own unit tests, so they were written after
the modules: they check the pieces fit together (CSV text cells, string labels, NaN through
the imputer, AUC column lookup, saving plots) rather than drive a design.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from numcompute_stream import metrics  # noqa: E402
from numcompute_stream.io import iter_chunks, load_csv  # noqa: E402
from numcompute_stream.metrics import F1, StreamingAUC  # noqa: E402
from numcompute_stream.pipeline import Pipeline  # noqa: E402
from numcompute_stream.preprocessing import Imputer, StandardScaler  # noqa: E402
from numcompute_stream.stream import StreamTrainer  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier  # noqa: E402
from numcompute_stream.visualise import (  # noqa: E402
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _write_csv(path, n, seed=0, missing=0.05):
    """Header + 3 numeric features (some cells blank) + a text label column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    labels = np.where(X[:, 0] + X[:, 1] > 0, "high", "low")
    cells = np.char.mod("%.6f", X).astype(object)
    cells[rng.random(X.shape) < missing] = ""                 # missing values in the file
    lines = ["f0,f1,f2,label"] + [",".join([*row, lab]) for row, lab in zip(cells, labels)]
    path.write_text("\n".join(lines) + "\n")
    return path


def _load(path):
    data = load_csv(path, skip_header=1, dtype=object)       # text label column -> object
    return data[:, :3].astype(np.float64), data[:, 3].astype(str)


def _trainer(buffer_size=300):
    pipe = Pipeline([("impute", Imputer()), ("scale", StandardScaler()),
                     ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size))])
    return StreamTrainer(pipe, metrics={"auc": StreamingAUC(pos_label="high"),
                                        "f1": F1(pos_label="high")})


def test_csv_to_pipeline_to_metrics_and_plots(tmp_path):
    X, y = _load(_write_csv(tmp_path / "stream.csv", n=1500))
    assert X.shape == (1500, 3) and np.isnan(X).any()        # blanks arrive as NaN

    trainer, scored_true, scored_pred = _trainer(), [], []
    for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=100):
        trainer.score_chunk(X_chunk, y_chunk)
        if trainer.last_y_pred_ is not None:
            scored_true.append(trainer.last_y_true_)
            scored_pred.append(trainer.last_y_pred_)
        trainer.fit_chunk(X_chunk, y_chunk)

    tree = trainer.model.steps[-1][1]
    assert trainer.history("n_samples_seen")[-1] == 1500 and tree.n_retrains_ == 5
    assert list(tree.classes_) == ["high", "low"]

    # nothing lost between modules: the logged cumulative accuracy is the batch accuracy
    # of exactly the predictions made chunk by chunk
    y_true, y_pred = np.concatenate(scored_true), np.concatenate(scored_pred)
    accuracy = trainer.history("cumulative_accuracy")
    assert accuracy[-1] == pytest.approx(metrics.accuracy(y_true, y_pred))
    assert accuracy[-1] > 0.85                                # learnable: the tree learns it
    assert 0.85 < trainer.history("auc")[-1] <= 1.0
    assert trainer.history("f1")[-1] == pytest.approx(
        metrics.f1(y_true, y_pred, pos_label="high"))

    plot_metric_over_time(accuracy, "Cumulative accuracy", "Accuracy",
                          save_path=tmp_path / "acc.png")
    plot_predictions_vs_ground_truth(trainer.last_y_true_, trainer.last_y_pred_,
                                     save_path=tmp_path / "cm.png")
    assert (tmp_path / "acc.png").stat().st_size > 0 and (tmp_path / "cm.png").exists()


def test_stream_is_reproducible_and_memory_bounded(tmp_path):
    X, y = _load(_write_csv(tmp_path / "long.csv", n=4500, seed=1))

    def run(n_rows, shuffle):
        trainer = _trainer(buffer_size=300)
        for X_chunk, y_chunk in iter_chunks(X[:n_rows], y[:n_rows], chunk_size=100,
                                            shuffle=shuffle, seed=7):
            trainer.score_chunk(X_chunk, y_chunk)
            trainer.fit_chunk(X_chunk, y_chunk)
        return trainer

    # same seeds -> identical logs, also with a shuffled stream
    first, second = run(4500, shuffle=True), run(4500, shuffle=True)
    for key in ("accuracy", "auc", "model_bytes"):
        np.testing.assert_array_equal(first.history(key), second.history(key))

    # memory is bounded by buffer_size, not by how much has streamed past
    short, long = run(1500, shuffle=False), run(4500, shuffle=False)
    assert long.history("model_bytes").max() <= 1.1 * short.history("model_bytes").max()

    compare_models(short.history("model_bytes"), long.history("model_bytes"),
                   labels=("1500 rows", "4500 rows"), ylabel="bytes",
                   save_path=tmp_path / "memory.png")
    assert (tmp_path / "memory.png").exists()
