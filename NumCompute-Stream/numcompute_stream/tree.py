"""Decision tree implemented from scratch with streaming support.

``DecisionTreeClassifier``: depth-limited, Gini or entropy-based.

- ``.fit(X, y)``: trains the tree from scratch on the given data
- ``.partial_fit(X_chunk, y_chunk)`` supports streaming via buffered retraining:

  - Each call appends the chunk to an internal data buffer
  - Once the buffer holds at least ``buffer_size`` samples, the tree is
    retrained on the buffered data and the buffer is cleared
  - Between retrains, ``.predict()`` uses the most recently trained tree

- Support config: ``max_depth``, ``min_samples_split``, ``max_features``,
  ``buffer_size``

Example
-------
>>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
>>> for X_chunk, y_chunk in stream:
...     tree.partial_fit(X_chunk, y_chunk)

See ``classification-tree.html`` (Canvas) for details on building the tree.
"""
