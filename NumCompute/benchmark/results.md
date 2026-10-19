# NumCompute benchmark results

Produced by `python benchmark/run_all.py`. Both the timings and the speed-up ratios depend
on the machine, the Python and NumPy versions, and the input sizes used here.


## NumCompute benchmarks: metrics, optim, stats

- python: 3.14.7
- numpy: 2.5.2
- platform: Windows-11-10.0.26200-SP0
- processor: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- seed: 5004, median of 5 runs

### 1) Speed: plain-Python loop vs numcompute vs NumPy

| Function | Input | Python loop | numcompute | NumPy | Speed-up vs loop |
|---|---|---|---|---|---|
| metrics.confusion_matrix | n=200,000, 5 classes | 10.05 ms | 8.42 ms | - | 1.2x |
| metrics.f1 (macro) | n=200,000, 5 classes | 40.12 ms | 8.65 ms | - | 4.6x |
| metrics.roc_curve + auc | n=3,000, with ties | 48.91 ms | 180.0 µs | - | 271.7x |
| stats.histogram | n=500,000, 20 bins | 52.55 ms | 13.76 ms | 7.96 ms | 3.8x |
| stats.median (axis=0) | 2,000 x 200 | 30.28 ms | 3.85 ms | 3.80 ms | 7.9x |
| stats.RunningStats (100 chunks) | n=500,000 | 35.07 ms | 1.87 ms | 1.25 ms | 18.8x |
| optim.grad (forward) | dim=200, same calls to f | 1.50 ms | 563.0 µs | - | 2.7x |

All versions were checked to give the same answer before timing. The loop for
confusion_matrix and f1 assumes the labels are the integers 0 to 4, while numcompute
also validates the input and finds the labels itself (a sort), so a small speed-up or a
slowdown is possible there. The AUC loop is O(n^2) and is timed once; the ROC version
sorts instead, so its speed-up comes from a better algorithm as well as from NumPy.
optim.grad is an implementation comparison: both versions loop over the coordinates.

### 2) Numerical stability

| Case | Result | True value |
|---|---|---|
| Variance, x = 1e9 + N(0,1): sum-of-squares formula | -256.000000 | 1.000830 |
| Variance, x = 1e9 + N(0,1): RunningStats (Welford) | 1.000830 | 1.000830 |

### 3) Finite-difference error for d/dx sin(x) at x = 1 (exact = cos 1)

| h | central error | forward error |
|---|---|---|
| 1e-01 | 9.00e-04 | 4.29e-02 |
| 1e-02 | 9.00e-06 | 4.22e-03 |
| 1e-03 | 9.01e-08 | 4.21e-04 |
| 1e-04 | 9.00e-10 | 4.21e-05 |
| 1e-05 | 1.11e-11 | 4.21e-06 |
| 1e-06 | 2.77e-11 | 4.21e-07 |
| 1e-07 | 1.94e-10 | 4.18e-08 |
| 1e-08 | 2.58e-09 | 2.97e-09 |
| 1e-09 | 2.97e-09 | 5.25e-08 |
| 1e-10 | 5.85e-08 | 5.85e-08 |

## NumCompute benchmarks: rank, sort_search, utils

- python: 3.14.7
- numpy: 2.5.2
- platform: Windows-11-10.0.26200-SP0
- processor: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- seed: 5004, median of 5 runs, quick: False

| Benchmark | Size | Python loop | numcompute | Speed-up |
|---|---|---|---|---|
| `rank` (average ties) | n = 200,000 | 107.90 ms | 18.53 ms | 5.8x |
| `percentile` (axis=0) | 50,000 x 20 | 114.94 ms | 15.03 ms | 7.6x |
| `multikey_sort` (asc + desc keys) vs `sorted` with key | n = 200,000 | 90.72 ms | 22.10 ms | 4.1x |
| `topk` (k = 10) vs `heapq.nlargest` | n = 2,000,000 | 50.81 ms | 10.46 ms | 4.9x |
| `binary_search` vs `bisect` loop | 200,000 queries | 62.86 ms | 18.75 ms | 3.4x |
| `pairwise_distances` (euclidean) | 300 x 8 | 89.70 ms | 5.45 ms | 16.5x |
| `softmax` (rows, loop over rows) | 50,000 x 50 | 126.35 ms | 26.64 ms | 4.7x |

Every pair of versions was checked to give the same answer before timing. The loop
baselines for `multikey_sort`, `topk` and `binary_search` use Python's own `sorted`,
`heapq` and `bisect`, which are fast C-backed tools, so those speed-ups are modest.
The `softmax` baseline still uses NumPy on every row, so only the row loop is removed.
The `pairwise_distances` loop is plain Python on lists.

### quickselect (written for teaching, so no speed-up is expected)

Finding the median of n = 1,000,000 values:

```
quickselect (ours)      25.768 ms  2.8x slower
np.sort then index       9.056 ms  (baseline)
np.partition             3.251 ms  2.8x faster
```

`quickselect` is slower than `np.sort` because NumPy's sort is compiled C, while the
partition loop in `quickselect` runs in Python. It is included to show the algorithm.
For real work, NumPy's partition functions are the right tool: `topk` uses
`np.argpartition`, and `np.partition` is the matching function for values.
