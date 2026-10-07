# Benchmark results: rank, sort_search, utils

Produced by `python benchmark/bench_rank_sort_utils.py` (seed 5004, median of 5 runs after one
warm-up). Every pair of versions was checked to give the same answer before timing.
Timings depend on the machine, so compare the ratios, not the milliseconds.

Environment: Python 3.13, NumPy 2.5, Linux x86_64.

| Benchmark | Size | Python loop | numcompute | Speed-up |
|---|---|---|---|---|
| `rank` (average ties) | n = 200,000 | 238.25 ms | 29.06 ms | 8.2x |
| `percentile` (axis=0) | 50,000 x 20 | 207.09 ms | 22.31 ms | 9.3x |
| `multikey_sort` (asc + desc keys) vs `sorted` with key | n = 200,000 | 147.82 ms | 32.73 ms | 4.5x |
| `topk` (k = 10) vs `heapq.nlargest` | n = 2,000,000 | 114.97 ms | 8.13 ms | 14.1x |
| `binary_search` vs `bisect` loop | 200,000 queries | 120.83 ms | 32.74 ms | 3.7x |
| `pairwise_distances` (euclidean) | 300 x 8 | 170.84 ms | 0.59 ms | 288x |
| `softmax` (rows) | 50,000 x 50 | 179.96 ms | 54.38 ms | 3.3x |

## quickselect (written for teaching, so no speed-up is expected)

Finding the median of 1,000,000 values:

| Method | Time | Relative |
|---|---|---|
| `quickselect` (ours) | 37.23 ms | 4.3x slower than sorting |
| `np.sort` then index | 8.68 ms | baseline |
| `np.partition` | 2.50 ms | 3.5x faster than sorting |

`quickselect` is slower than `np.sort` because NumPy's sort is compiled C, while quickselect here
runs its partition loop in Python. It is included to show the algorithm. For real work,
`np.partition` (used inside `topk`) is the right tool.
