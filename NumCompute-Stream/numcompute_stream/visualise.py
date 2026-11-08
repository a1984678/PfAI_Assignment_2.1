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
    raise NotImplementedError


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
    raise NotImplementedError


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
    sample with the wrong ones highlighted
    - title: str
    - ax, save_path, show: see the module docstring

    Returns:
    -------
    - matplotlib.axes.Axes

    Raises:
    ------
    - ValueError: if y_true and y_pred differ in length, or kind is unknown
    """
    raise NotImplementedError
