"""NumCompute-Stream: a modularised, decision tree-based streaming ML framework.

Extends NumCompute with streaming learning support. All components (models,
preprocessing, metrics, statistics) support incremental updates as new data
arrives in chunks, via ``.partial_fit()`` or ``.update()``.

Only NumPy and matplotlib may be used (no scikit-learn, pandas, PyTorch, etc.).

Modules
-------
io            : CSV loading (custom I/O pipeline)
stream        : StreamTrainer (model + pipeline + logging)
tree          : DecisionTreeClassifier with buffered retraining
preprocessing : Streaming scalers, imputer and one-hot encoder
stats         : Chunk-based statistics
metrics       : Streaming classification metrics
pipeline      : Chained pipeline supporting partial_fit
visualise     : Reusable matplotlib plotting functions

Baseline modules carried over from NumCompute (Assignment 2.1)
--------------------------------------------------------------
rank, sort_search, optim, utils, benchmarking
"""
