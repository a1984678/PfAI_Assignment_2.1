"""Tests for ``numcompute_stream.stats``.

Cover chunk-based mean, variance, quantiles and histograms (optionally with
sliding window), and the ``update_stats(X_chunk)`` API, checking streaming
results agree with full-batch results.
"""
