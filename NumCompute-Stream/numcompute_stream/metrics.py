"""Streaming classification metrics.

- All classification metrics (``accuracy``, ``precision``, ``recall``, etc.)
  must support:

  - Streaming updates via ``update(y_true_chunk, y_pred_chunk)``
  - ``reset()``, ``result()`` methods

- Confusion matrix and AUC must accumulate over time
- Include support for rolling-window metrics
"""
