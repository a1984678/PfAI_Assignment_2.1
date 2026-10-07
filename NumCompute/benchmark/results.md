NumCompute benchmarks: metrics, optim, stats
Python 3.13.16 | NumPy 2.5.3 | Linux-6.18.44-fc-v77-x86_64-with-glibc2.39 | seed 5004 | median of 5 runs

1) Speed: plain-Python loop vs numcompute (vectorised) vs NumPy
| Function | Input | Python loop | numcompute | NumPy | Speed-up vs loop |
|---|---|---|---|---|---|
| metrics.confusion_matrix | n=200,000, 5 classes | 10.85 ms | 14.43 ms | - | 1x |
| metrics.f1 (macro) | n=200,000, 5 classes | 58.59 ms | 15.16 ms | - | 4x |
| metrics.roc_curve + auc | n=3,000, with ties | 73.25 ms | 292.8 µs | - | 250x |
| stats.histogram | n=500,000, 20 bins | 52.89 ms | 11.47 ms | 4.18 ms | 5x |
| stats.median (axis=0) | 2,000 x 200 | 49.91 ms | 3.18 ms | 5.52 ms | 16x |
| stats.RunningStats (100 chunks) | n=500,000 | 45.81 ms | 2.68 ms | 939.4 µs | 17x |
| optim.grad (forward) | dim=200 | 1.83 ms | 790.6 µs | - | 2x |

(The AUC loop is timed once: it is O(n^2). All versions were checked to agree.)

2) Numerical stability
| Case | Result | True value |
|---|---|---|
| Variance, x = 1e9 + N(0,1): sum-of-squares formula | -256.000000 | 1.000830 |
| Variance, x = 1e9 + N(0,1): RunningStats (Welford) | 1.000830 | 1.000830 |

3) Finite-difference error for d/dx sin(x) at x = 1 (exact = cos 1)
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
