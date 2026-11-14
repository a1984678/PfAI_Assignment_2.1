# NumCompute-Stream

A streaming machine learning framework built on a decision tree, extending NumCompute
(Assignment 2.1). Every component learns from data in chunks through `partial_fit()` or
`update()`. The tree uses buffered retraining: it rebuilds itself each time `buffer_size` rows
have arrived. Uses only NumPy and matplotlib.

## Installation

From the `NumCompute-Stream` folder (Python 3.12):

```bash
pip install -e .
```

## Usage

```python
from numcompute_stream.io import load_csv, iter_chunks
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.metrics import StreamingAUC
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.visualise import plot_metric_over_time

data = load_csv("data.csv", skip_header=1)       # last column = label
X, y = data[:, :-1], data[:, -1]

pipe = Pipeline([
    ("impute", Imputer(strategy="mean")),
    ("scale", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])
trainer = StreamTrainer(pipe, metrics={"auc": StreamingAUC(pos_label=1)})

for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=100):
    trainer.score_chunk(X_chunk, y_chunk)    # test on the new chunk
    trainer.fit_chunk(X_chunk, y_chunk)      # then train on it (pipe.partial_fit)

plot_metric_over_time(trainer.history("cumulative_accuracy"), "Accuracy", "Accuracy")
```

Until the first `buffer_size` rows have arrived the tree has no model, so `predict` raises
`RuntimeError` and `StreamTrainer` logs NaN for those chunks.

## Modules

- `io`: `load_csv` reads a CSV file, and `iter_chunks` splits data into chunks.
- `stats`: `StreamingStats`, `StreamingQuantiles` and `StreamingHistogram`, all updated with
  `update_stats(X_chunk)`.
- `preprocessing`: `StandardScaler`, `MinMaxScaler`, `Imputer` and `OneHotEncoder`, all with
  `partial_fit`.
- `metrics`: `Accuracy`, `Precision`, `Recall`, `F1`, `StreamingConfusionMatrix` and
  `StreamingAUC`. Each has `update`, `result` and `reset`, and `window=` gives rolling metrics.
- `tree`: `DecisionTreeClassifier` with `fit`, `partial_fit`, `predict` and `predict_proba`.
- `pipeline`: `Pipeline` with `partial_fit`, `predict` and `predict_proba`.
- `stream`: `StreamTrainer` logs per-chunk metrics, cumulative accuracy and memory footprint.
- `visualise`: `plot_metric_over_time`, `compare_models` and
  `plot_predictions_vs_ground_truth`. Each takes `save_path=` and `show=`.

The batch functions and the `rank`, `sort_search`, `optim`, `utils` and `benchmarking` modules
are kept from the 2.1 group code.

## Demo

Open `demo/stream_demo.ipynb` and run all cells. It loads `demo/data/sensor_stream.csv`,
streams it in chunks through two pipelines with different `buffer_size` values, and plots the
results.

## Tests and benchmarks

```bash
pytest                       # all tests
flake8                       # style check
python benchmark/run_all.py  # writes benchmark/results.md (~35 s)
```

The streaming tests are in `tests/test_*.py`, and the 2.1 tests are in `tests/baseline/`. The
benchmarks compare loop-based and vectorised versions, and different `buffer_size` values. The
results are in `benchmark/results.md`.
