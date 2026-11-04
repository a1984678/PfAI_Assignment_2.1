"""Tests for ``numcompute_stream.preprocessing``.

Cover ``.partial_fit()`` for ``StandardScaler`` (running mean/var, e.g.
Welford or EMA), ``Imputer`` (on-the-fly missing-value estimates) and
``OneHotEncoder`` (incremental category expansion), including edge cases such
as zero-variance chunks and all-NaN columns.
"""
