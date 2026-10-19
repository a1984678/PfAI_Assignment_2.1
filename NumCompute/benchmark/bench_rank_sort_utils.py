"""
bench_rank_sort_utils.py

Reproducible benchmarks for numcompute.rank, numcompute.sort_search and numcompute.utils.

Each benchmark compares a loop version against the numcompute version, first checks that
both give the same answer (benchmarking.compare_loop_vs_vectorised raises if they differ),
then reports the median time. The timing itself comes from numcompute.benchmarking.

Run from the NumCompute folder:

    python benchmark/bench_rank_sort_utils.py
    python benchmark/bench_rank_sort_utils.py --quick      (smaller sizes)

Reproducibility: fixed seed (SEED), median of REPEATS runs after one warm-up run,
and Python / NumPy / platform details are printed with the results.
"""

import argparse
import bisect
import heapq
import math
import os
import sys

import numpy as np

# make `import numcompute` work when run as `python benchmark/...` from NumCompute/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numcompute import benchmarking, rank, sort_search, utils  # noqa: E402

SEED = 5004
REPEATS = 5


# ---------------------------------------------------------------------
# Plain-Python baselines
# ---------------------------------------------------------------------

def loop_rank_average(values):
    """Average ranks with one pass over the sorted order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return np.array(ranks)


def loop_column_percentile(x2d, q):
    """Linear-interpolation percentile of every column, one column at a time."""
    out = []
    for col in range(x2d.shape[1]):
        s = sorted(x2d[:, col].tolist())
        pos = q / 100.0 * (len(s) - 1)
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        out.append(s[lo] + (pos - lo) * (s[hi] - s[lo]))
    return np.array(out)


def loop_multikey_order(a, b):
    """Order rows by a ascending then b descending, using Python's sorted with a key."""
    return np.array(sorted(range(len(a)), key=lambda i: (a[i], -b[i])))


def loop_topk_values(values, k):
    return np.array(heapq.nlargest(k, values.tolist()))


def loop_binary_search(sorted_list, queries):
    return np.array([bisect.bisect_left(sorted_list, q) for q in queries])


def loop_pairwise_euclidean(X):
    """Plain Python on lists: a double loop over rows and a sum over the columns."""
    rows = X.tolist()
    return np.array([[math.sqrt(sum((a - b) ** 2 for a, b in zip(ri, rj))) for rj in rows]
                     for ri in rows])


def loop_softmax_rows(X):
    """One row at a time, but each row still uses NumPy (a fair loop-over-rows baseline)."""
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        e = np.exp(X[i] - X[i].max())
        out[i] = e / e.sum()
    return out


# ---------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------

def fmt_speedup(ratio):
    """'4.2x' or '0.8x (slower)': one decimal, so a slowdown is not rounded to 1x."""
    return "{:,.1f}x".format(ratio) + ("" if ratio >= 1 else " (slower)")


def run(quick):
    rng = np.random.default_rng(SEED)
    n_rank = 20_000 if quick else 200_000
    n_rows = 2_000 if quick else 50_000
    n_top = 200_000 if quick else 2_000_000
    n_dist = 80 if quick else 300
    cases = []

    x = rng.integers(0, n_rank // 10, n_rank).astype(float)
    cases.append(("`rank` (average ties) | n = {:,}".format(n_rank),
                  lambda: loop_rank_average(x.tolist()), lambda: rank.rank(x)))

    X = rng.normal(size=(n_rows, 20))
    cases.append(("`percentile` (axis=0) | {:,} x 20".format(n_rows),
                  lambda: loop_column_percentile(X, 37.5),
                  lambda: rank.percentile(X, 37.5, axis=0)))

    keys = rng.integers(0, 50, size=(n_rank, 2))
    cases.append(("`multikey_sort` (asc + desc keys) vs `sorted` with key | n = {:,}"
                  .format(n_rank),
                  lambda: loop_multikey_order(keys[:, 0].tolist(), keys[:, 1].tolist()),
                  lambda: sort_search.multikey_sort(keys, [0, 1], ascending=[True, False],
                                                    return_indices=True)[1]))

    v = rng.normal(size=n_top)
    cases.append(("`topk` (k = 10) vs `heapq.nlargest` | n = {:,}".format(n_top),
                  lambda: loop_topk_values(v, 10),
                  lambda: sort_search.topk(v, 10, return_indices=False)))

    s = np.sort(rng.normal(size=n_rank))
    s_list, queries = s.tolist(), rng.normal(size=n_rank)
    cases.append(("`binary_search` vs `bisect` loop | {:,} queries".format(n_rank),
                  lambda: loop_binary_search(s_list, queries.tolist()),
                  lambda: sort_search.binary_search(s, queries)[0]))

    P = rng.normal(size=(n_dist, 8))
    cases.append(("`pairwise_distances` (euclidean) | {} x 8".format(n_dist),
                  lambda: loop_pairwise_euclidean(P),
                  lambda: utils.pairwise_distances(P)))

    S = rng.normal(size=(n_rows, 50))
    cases.append(("`softmax` (rows, loop over rows) | {:,} x 50".format(n_rows),
                  lambda: loop_softmax_rows(S), lambda: utils.softmax(S, axis=1)))

    print("| Benchmark | Size | Python loop | numcompute | Speed-up |")
    print("|---|---|---|---|---|")
    for label, loop_f, vec_f in cases:
        out = benchmarking.compare_loop_vs_vectorised(loop_f, vec_f, repeats=REPEATS)
        print("| {} | {:.2f} ms | {:.2f} ms | {} |".format(
            label, out["loop"] * 1e3, out["vectorised"] * 1e3, fmt_speedup(out["speedup"])))
    print()
    print("Every pair of versions was checked to give the same answer before timing. The loop")
    print("baselines for `multikey_sort`, `topk` and `binary_search` use Python's own `sorted`,")
    print("`heapq` and `bisect`, which are fast C-backed tools, so those speed-ups are modest.")
    print("The `softmax` baseline still uses NumPy on every row, so only the row loop is removed.")
    print("The `pairwise_distances` loop is plain Python on lists.")


def run_quickselect(quick):
    """quickselect is for teaching: compare honestly with np.sort and np.partition."""
    n = 100_000 if quick else 1_000_000
    v = np.random.default_rng(SEED).normal(size=n)
    k = n // 2
    if sort_search.quickselect(v, k) != np.sort(v)[k]:
        raise RuntimeError("quickselect gives a different answer from np.sort")
    results = benchmarking.compare({
        "quickselect (ours)": lambda: sort_search.quickselect(v, k),
        "np.sort then index": lambda: np.sort(v)[k],
        "np.partition": lambda: np.partition(v, k)[k],
    }, repeats=REPEATS)
    print()
    print("### quickselect (written for teaching, so no speed-up is expected)")
    print()
    print("Finding the median of n = {:,} values:".format(n))
    print()
    print("```")
    print(benchmarking.report(results, baseline="np.sort then index"))
    print("```")
    print()
    print("`quickselect` is slower than `np.sort` because NumPy's sort is compiled C, while the")
    print("partition loop in `quickselect` runs in Python. It is included to show the algorithm.")
    print("For real work, NumPy's partition functions are the right tool: `topk` uses")
    print("`np.argpartition`, and `np.partition` is the matching function for values.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="use smaller sizes")
    args = parser.parse_args()

    print("## NumCompute benchmarks: rank, sort_search, utils")
    print()
    for key, value in benchmarking.environment_info().items():
        print("- {}: {}".format(key, value))
    print("- seed: {}, median of {} runs, quick: {}".format(SEED, REPEATS, args.quick))
    print()
    run(args.quick)
    run_quickselect(args.quick)


if __name__ == "__main__":
    main()
