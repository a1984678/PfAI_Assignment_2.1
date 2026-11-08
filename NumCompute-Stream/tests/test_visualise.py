"""Tests for ``numcompute_stream.visualise``.

Cover ``plot_metric_over_time``, ``compare_models`` and
``plot_predictions_vs_ground_truth``, including saving to file and inline
display options.
"""

import matplotlib
matplotlib.use("Agg")                     # no window: works headless and in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from numcompute_stream import metrics  # noqa: E402
from numcompute_stream.visualise import (  # noqa: E402
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")                      # matplotlib warns after 20 open figures


def test_plot_metric_over_time_values_options_and_errors(tmp_path, monkeypatch):
    values = np.array([np.nan, np.nan, 0.7, 0.8, 0.85])   # NaN before the first tree
    path = tmp_path / "acc.png"
    ax = plot_metric_over_time(values, "Accuracy over time", "Accuracy", save_path=path)

    line, = ax.get_lines()
    np.testing.assert_array_equal(line.get_xdata(), np.arange(5))
    np.testing.assert_array_equal(line.get_ydata(), values)   # NaN kept: drawn as a gap
    assert (ax.get_title(), ax.get_ylabel(), ax.get_xlabel()) == (
        "Accuracy over time", "Accuracy", "Chunk")
    assert path.stat().st_size > 0

    # custom x (e.g. samples seen), drawing into a given Axes, and show
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    _, given = plt.subplots()
    ax = plot_metric_over_time(values, "t", "y", x=[50, 100, 150, 200, 250],
                               xlabel="Samples seen", ax=given, show=True)
    assert ax is given and shown == [True]
    np.testing.assert_array_equal(ax.get_lines()[0].get_xdata(), [50, 100, 150, 200, 250])

    with pytest.raises(ValueError, match="1-dimensional"):
        plot_metric_over_time([[0.5, 0.6]], "t", "y")
    with pytest.raises(ValueError, match="same length"):
        plot_metric_over_time(values, "t", "y", x=[1, 2])


def test_compare_models_two_series_with_labels(tmp_path):
    small, large = [0.6, 0.7, 0.75, 0.8], [np.nan, 0.65, 0.8]     # lengths may differ
    ax = compare_models(small, large, labels=("buffer 100", "buffer 500"),
                        save_path=tmp_path / "cmp.png")
    lines = ax.get_lines()
    assert [ln.get_label() for ln in lines] == ["buffer 100", "buffer 500"]
    np.testing.assert_array_equal(lines[1].get_ydata(), large)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["buffer 100", "buffer 500"]
    assert (tmp_path / "cmp.png").exists()

    with pytest.raises(ValueError, match="labels must be two"):
        compare_models(small, large, labels=("only one",))


def test_plot_predictions_vs_ground_truth_confusion_and_samples():
    y_true = np.array(["cat", "dog", "dog", "bird", "cat", "dog"])
    y_pred = np.array(["cat", "cat", "dog", "bird", "dog", "dog"])

    ax = plot_predictions_vs_ground_truth(y_true, y_pred)       # default: confusion matrix
    np.testing.assert_array_equal(ax.images[0].get_array(),
                                  metrics.confusion_matrix(y_true, y_pred))
    assert [t.get_text() for t in ax.get_xticklabels()] == ["bird", "cat", "dog"]
    assert sorted(t.get_text() for t in ax.texts) == sorted("100011012")  # count per cell

    # per-sample view: true labels and predictions, the 2 mistakes highlighted
    ax = plot_predictions_vs_ground_truth(y_true, y_pred, kind="samples")
    by_label = {c.get_label(): c for c in ax.collections}
    assert len(by_label["wrong"].get_offsets()) == 2
    assert [t.get_text() for t in ax.get_yticklabels()] == ["bird", "cat", "dog"]

    with pytest.raises(ValueError, match="same length"):
        plot_predictions_vs_ground_truth(y_true, y_pred[:3])
    with pytest.raises(ValueError, match="kind must be"):
        plot_predictions_vs_ground_truth(y_true, y_pred, kind="pie")
