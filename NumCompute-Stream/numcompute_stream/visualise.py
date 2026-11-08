"""Reusable plotting functions using matplotlib.

Must be usable across scripts, demos, or pipeline logs.

Required plots:

- ``plot_metric_over_time(metric_values, title, ylabel)``: Plot a metric
  (e.g., accuracy) across chunks
- ``compare_models(metric1, metric2, labels)``: Compare two models on
  streaming metrics (e.g., trees with different ``buffer_size`` values)
- ``plot_predictions_vs_ground_truth(y_true, y_pred)``: Visualise
  predictions vs. actuals on latest chunk

Should support options for saving to file or inline display.

Common options (all three functions):

- ``ax``: draw into an existing matplotlib Axes (e.g. a subplot grid); default a new figure
- ``save_path``: also save the figure to this file (format from the extension, e.g. .png)
- ``show``: call ``plt.show()`` (scripts); in a notebook the returned plot displays inline
- Every function returns the Axes, so the caller can adjust or inspect the plot.
"""

import matplotlib.pyplot as plt
import numpy as np

from .metrics import confusion_matrix

_KINDS = ("confusion", "samples")


def _series(values, name):
    """A metric series as a 1-D float array (NaN kept)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional (n_chunks,). Got shape: {arr.shape}")
    return arr


def _axes(ax):
    """The given Axes, or a new figure's Axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    return ax


def _finish(ax, title, save_path, show):
    """Set the title, then save and/or show the figure the Axes belongs to."""
    ax.set_title(title)
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric", *,
                          x=None, xlabel="Chunk", ax=None, save_path=None, show=False):
    """
    Line plot of one streaming metric, one point per chunk.

    Parameters:
    ----------
    - metric_values: array-like of shape (n_chunks,); NaN (e.g. before the first tree) is
    drawn as a gap
    - title, ylabel, xlabel: str
    - x: array-like of shape (n_chunks,), x positions (default 0..n_chunks-1, the chunk
    index), e.g. trainer.history("n_samples_seen")
    - ax, save_path, show: see the module docstring

    Returns:
    -------
    - matplotlib.axes.Axes

    Raises:
    ------
    - ValueError: if metric_values is not 1-D, or x has a different length
    """
    values = _series(metric_values, "metric_values")
    xs = np.arange(len(values)) if x is None else _series(x, "x")
    if len(xs) != len(values):
        raise ValueError(f"x and metric_values must have the same length. "
                         f"Got {len(xs)} and {len(values)}.")
    ax = _axes(ax)
    ax.plot(xs, values, marker="o", markersize=3)    # markers keep lone points visible
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _finish(ax, title, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), *,
                   title="Model comparison", ylabel="Metric", xlabel="Chunk", ax=None,
                   save_path=None, show=False):
    """
    Two streaming metric series on one plot, e.g. trees with different buffer_size.

    Parameters:
    ----------
    - metric1, metric2: array-like of shape (n1,) and (n2,); lengths may differ
    - labels: two str, legend entries for metric1 and metric2
    - title, ylabel, xlabel: str
    - ax, save_path, show: see the module docstring

    Returns:
    -------
    - matplotlib.axes.Axes

    Raises:
    ------
    - ValueError: if a series is not 1-D or labels is not two names
    """
    if isinstance(labels, str) or len(labels) != 2:
        raise ValueError(f"labels must be two names, one per series. Got: {labels!r}")
    series = (_series(metric1, "metric1"), _series(metric2, "metric2"))
    ax = _axes(ax)
    for values, label in zip(series, labels):
        ax.plot(np.arange(len(values)), values, marker="o", markersize=3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(ax, title, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, kind="confusion",
                                     title="Predictions vs ground truth", ax=None,
                                     save_path=None, show=False):
    """
    Compare predictions with the true labels, e.g. of the latest chunk.

    Parameters:
    ----------
    - y_true, y_pred: array-like of shape (n_samples,), labels (numbers or strings)
    - kind: "confusion" (default) for a confusion-matrix heatmap with counts (rows = true,
    columns = predicted, labels sorted), or "samples" for true labels and predictions per
    sample with the wrong ones highlighted (readable up to a few hundred samples; use
    "confusion" for larger chunks)
    - title: str
    - ax, save_path, show: see the module docstring

    Returns:
    -------
    - matplotlib.axes.Axes

    Raises:
    ------
    - ValueError: if y_true and y_pred differ in length, or kind is unknown
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}. Got: {kind!r}")
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D with the same length. "
                         f"Got shapes {y_true.shape} and {y_pred.shape}.")
    labels = np.unique(np.concatenate([y_true, y_pred]))
    names = [str(label) for label in labels]
    ticks = np.arange(len(labels))
    ax = _axes(ax)

    if kind == "confusion":
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        ax.imshow(cm, cmap="Blues")
        # one text per cell: a loop over n_classes^2 cells, not over samples
        dark = cm > cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if dark[i, j] else "black")
        ax.set_xticks(ticks, labels=names)
        ax.set_yticks(ticks, labels=names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    else:
        index = np.arange(len(y_true))
        true_pos = np.searchsorted(labels, y_true)
        pred_pos = np.searchsorted(labels, y_pred)
        wrong = true_pos != pred_pos
        ax.scatter(index, true_pos, marker="o", s=40, facecolors="none",
                   edgecolors="tab:blue", label="true")
        ax.scatter(index, pred_pos, marker="x", s=25, color="tab:orange", label="predicted")
        ax.scatter(index[wrong], pred_pos[wrong], marker="s", s=90, facecolors="none",
                   edgecolors="tab:red", label="wrong")
        ax.set_yticks(ticks, labels=names)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Label")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))   # outside: never covers points

    accuracy = np.mean(y_true == y_pred) if len(y_true) else np.nan
    return _finish(ax, f"{title}\naccuracy {accuracy:.3f} on {len(y_true)} samples",
                   save_path, show)
