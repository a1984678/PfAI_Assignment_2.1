"""Chained streaming pipeline.

- Must support ``partial_fit()`` for models and transformers
- Support incremental transformation + prediction in chained pipeline

Example
-------
>>> pipe = Pipeline([
...     ('scale', StandardScaler()),
...     ('model', DecisionTreeClassifier(max_depth=5, buffer_size=500))
... ])
>>> pipe.partial_fit(X_chunk, y_chunk)
"""
