"""End-to-end streaming tests.

Load CSV data, split into chunks, train a pipeline incrementally with
``.partial_fit()`` and evaluate with streaming metrics.
"""
