## Module Changes & New Components

All components must now support **streaming settings**, with support for incremental `.partial_fit()` and per-chunk updates.
### `stream.py`
- Implements `StreamTrainer`, managing model + pipeline + logging
- Supports `.fit_chunk(X, y)` and `.score_chunk(X, y)` logic
- Logs per-chunk metrics, memory footprint, and cumulative accuracy

### `tree.py`
- `DecisionTreeClassifier`: depth-limited, Gini or entropy-based
- `.fit(X, y)`: trains the tree from scratch on the given data
- `.partial_fit(X_chunk, y_chunk)` supports streaming via **buffered retraining**:
  - Each call appends the chunk to an internal data buffer
  - Once the buffer holds at least `buffer_size` samples, the tree is retrained on the buffered data and the buffer is cleared
  - Between retrains, `.predict()` uses the most recently trained tree
- Support config: `max_depth`, `min_samples_split`, `max_features`, `buffer_size`
  ```python
  tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
  for X_chunk, y_chunk in stream:
      tree.partial_fit(X_chunk, y_chunk)
  ```

### `preprocessing.py`
- Scalers (e.g., `StandardScaler`) must support `.partial_fit()`:
  - Maintain running mean/var (e.g., Welford or EMA)
- `Imputer` updates missing-value estimates on the fly
- `OneHotEncoder` expands categories incrementally

### `stats.py`
- All statistical functions redesigned for streaming
- Implement chunk-based versions of:
  - Mean, variance, quantiles
  - Histograms (optionally with sliding window)
- Add `update_stats(X_chunk)` API

### `metrics.py`
- All classification metrics (`accuracy`, `precision`, `recall`, etc.) must support:
  - Streaming updates via `update(y_true_chunk, y_pred_chunk)`
  - `reset()`, `result()` methods
- Confusion matrix and AUC must accumulate over time
- Include support for rolling-window metrics

### `pipeline.py`
- Must support `partial_fit()` for models and transformers
- Support incremental transformation + prediction in chained pipeline
  ```python
  pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', DecisionTreeClassifier(max_depth=5, buffer_size=500))
  ])
  pipe.partial_fit(X_chunk, y_chunk)
  ```
### `visualise.py`
- Provide reusable plotting functions using `matplotlib`
- Must be usable across scripts, demos, or pipeline logs
- Required plots:
  - `plot_metric_over_time(metric_values, title, ylabel)`: Plot a metric (e.g., accuracy) across chunks
  - `compare_models(metric1, metric2, labels)`: Compare two models on streaming metrics (e.g., trees with different `buffer_size` values)
  - `plot_predictions_vs_ground_truth(y_true, y_pred)`: Visualise predictions vs. actuals on latest chunk
- Should support options for saving to file or inline display
