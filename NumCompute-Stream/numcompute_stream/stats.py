"""Streaming statistics.

- All statistical functions redesigned for streaming
- Implement chunk-based versions of:

  - Mean, variance, quantiles
  - Histograms (optionally with sliding window)

- Add ``update_stats(X_chunk)`` API
"""
