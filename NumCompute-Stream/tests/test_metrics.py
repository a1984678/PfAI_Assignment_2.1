"""Tests for ``numcompute_stream.metrics``.

Cover streaming ``update(y_true_chunk, y_pred_chunk)``, ``reset()`` and
``result()`` for accuracy, precision, recall, etc.; accumulation of the
confusion matrix and AUC over time; and rolling-window metrics.
"""
