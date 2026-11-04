# NumCompute-Stream

A modularised, decision tree-based streaming machine learning framework,
extending NumCompute. All components (models, preprocessing, metrics,
statistics) support incremental updates as data arrives in chunks.
Built using only NumPy and matplotlib.

## Installation

## Usage

## Modules

- `stream.py` – `StreamTrainer`: model + pipeline + logging, `.fit_chunk()` / `.score_chunk()`
- `tree.py` – `DecisionTreeClassifier` with buffered retraining via `.partial_fit()`
- `preprocessing.py` – streaming `StandardScaler`, `Imputer`, `OneHotEncoder`
- `stats.py` – chunk-based mean, variance, quantiles, histograms; `update_stats()`
- `metrics.py` – streaming metrics with `update()`, `reset()`, `result()`
- `pipeline.py` – chained `Pipeline` supporting `partial_fit()`
- `visualise.py` – matplotlib plots for metrics over time, model comparison, predictions
- `io.py` – CSV loading and chunking

## Demo

See `demo/stream_demo.ipynb`.

## Testing

## Benchmarks

See `benchmark/results.md`.
