"""Tests for ``numcompute_stream.tree.DecisionTreeClassifier``.

Cover ``.fit()``, ``.predict()`` and ``.partial_fit()`` buffered retraining:
chunks are appended to the buffer, the tree retrains once the buffer holds at
least ``buffer_size`` samples and the buffer is cleared, and ``.predict()``
uses the most recent tree between retrains. Also cover Gini vs. entropy,
``max_depth``, ``min_samples_split``, ``max_features``, and edge cases such as
NaNs, ties, and zero-variance chunks.
"""
