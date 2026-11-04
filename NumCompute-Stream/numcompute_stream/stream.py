"""Streaming training management.

- Implements ``StreamTrainer``, managing model + pipeline + logging
- Supports ``.fit_chunk(X, y)`` and ``.score_chunk(X, y)`` logic
- Logs per-chunk metrics, memory footprint, and cumulative accuracy
"""
