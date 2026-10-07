"""
bench_metrics_optim_stats.py

Reproducible benchmarks for numcompute.metrics, numcompute.optim and numcompute.stats.

Each benchmark compares a plain-Python loop baseline against the vectorised
numcompute version (and NumPy's own function where one exists), checks that all
versions give the same answer, then reports the median time over several runs.

Run from the NumCompute folder:

    python benchmark/bench_metrics_optim_stats.py
    python benchmark/bench_metrics_optim_stats.py --quick      (smaller sizes, ~5 s)

Reproducibility:
- every input is generated from a fixed seed (SEED below)
- each timing is the median of REPEATS runs, after one warm-up run
- Python, NumPy and platform versions are printed with the results
"""

import argparse
import os
import platform
import sys
import time

import numpy as np

# make `import numcompute` work when run as `python benchmark/...` from NumCompute/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numcompute import metrics, optim, stats  # noqa: E402

SEED = 5004
REPEATS = 5


# ---------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------

def time_it(func, repeats=REPEATS):
    """
    Median wall-clock time of func() in seconds, after one warm-up call.

    Parameters:
    ----------
    - func: callable with no arguments
    - repeats: int, number of timed runs

    Returns:
    -------
    - (median_seconds, result of the last call)
    """
    result = func()  # warm-up (also gives us the result to check)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), result


def fmt_time(seconds):
    """Format seconds as ms or µs for the results table."""
    if seconds >= 1e-3:
        return "{:.2f} ms".format(seconds * 1e3)
    return "{:.1f} µs".format(seconds * 1e6)


# ---------------------------------------------------------------------
# Loop baselines (deliberately plain Python, used only for comparison)
# ---------------------------------------------------------------------

def loop_confusion_matrix(y_true, y_pred, n_classes):
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t][p] += 1
    return np.array(cm)


def loop_macro_f1(y_true, y_pred, n_classes):
    scores = []
    yt, yp = y_true.tolist(), y_pred.tolist()
    for c in range(n_classes):
        tp = fp = fn = 0
        for t, p in zip(yt, yp):
            if p == c and t == c:
                tp += 1
            elif p == c:
                fp += 1
            elif t == c:
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / n_classes


def loop_auc(y_true, y_score):
    """O(n^2) definition: share of (positive, negative) pairs ranked correctly."""
    pos = [s for s, t in zip(y_score.tolist(), y_true.tolist()) if t == 1]
    neg = [s for s, t in zip(y_score.tolist(), y_true.tolist()) if t == 0]
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0 if p > n else 0.5 if p == n else 0.0
    return total / (len(pos) * len(neg))


def loop_histogram(x, bins):
    low, high = min(x), max(x)
    width = (high - low) / bins
    counts = [0] * bins
    for v in x:
        i = int((v - low) / width)
        counts[min(i, bins - 1)] += 1
    return np.array(counts)


def loop_column_medians(x2d):
    out = []
    for j in range(x2d.shape[1]):
        col = sorted(x2d[:, j].tolist())
        m = len(col)
        mid = m // 2
        out.append(col[mid] if m % 2 else (col[mid - 1] + col[mid]) / 2)
    return np.array(out)


def loop_welford(values):
    """One value at a time, textbook Welford."""
    count, mean, m2 = 0, 0.0, 0.0
    for v in values.tolist():
        count += 1
        delta = v - mean
        mean += delta / count
        m2 += delta * (v - mean)
    return mean, m2 / count


def naive_variance(values):
    """The 'sum of squares' formula E[x^2] - E[x]^2: fast but numerically unstable."""
    n = values.size
    return float(np.sum(values ** 2) / n - (np.sum(values) / n) ** 2)


def loop_grad(f, x, h):
    """Forward-difference gradient using a Python list for x (baseline for cost only)."""
    x = x.tolist()
    f0 = f(np.array(x))
    g = []
    for i in range(len(x)):
        xp = list(x)
        xp[i] += h
        g.append((f(np.array(xp)) - f0) / h)
    return np.array(g)


# ---------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------

def bench_speed(sizes):
    """Loop vs numcompute (vs NumPy) timings. Returns a list of result rows."""
    rng = np.random.default_rng(SEED)
    rows = []

    # 1. confusion matrix -----------------------------------------------
    n, k = sizes["n_labels"], 5
    y_true = rng.integers(0, k, n)
    y_pred = np.where(rng.random(n) < 0.7, y_true, rng.integers(0, k, n))
    t_loop, r_loop = time_it(lambda: loop_confusion_matrix(y_true, y_pred, k))
    t_vec, r_vec = time_it(lambda: metrics.confusion_matrix(y_true, y_pred))
    assert np.array_equal(r_loop, r_vec)
    rows.append(("metrics.confusion_matrix", "n={:,}, 5 classes".format(n),
                 t_loop, t_vec, None))

    # 2. macro F1 ---------------------------------------------------------
    t_loop, r_loop = time_it(lambda: loop_macro_f1(y_true, y_pred, k))
    t_vec, r_vec = time_it(lambda: metrics.f1(y_true, y_pred, average="macro"))
    assert np.isclose(r_loop, r_vec)
    rows.append(("metrics.f1 (macro)", "n={:,}, 5 classes".format(n),
                 t_loop, t_vec, None))

    # 3. AUC --------------------------------------------------------------
    n_auc = sizes["n_auc"]
    yb = rng.integers(0, 2, n_auc)
    score = np.round(yb * 0.6 + rng.random(n_auc), 2)  # rounding creates ties
    t_loop, r_loop = time_it(lambda: loop_auc(yb, score), repeats=1)

    def vec_auc():
        fpr, tpr, _ = metrics.roc_curve(yb, score)
        return metrics.auc(fpr, tpr)

    t_vec, r_vec = time_it(vec_auc)
    assert np.isclose(r_loop, r_vec)
    rows.append(("metrics.roc_curve + auc", "n={:,}, with ties".format(n_auc),
                 t_loop, t_vec, None))

    # 4. histogram --------------------------------------------------------
    n_h = sizes["n_hist"]
    x = rng.normal(size=n_h)
    t_loop, r_loop = time_it(lambda: loop_histogram(x.tolist(), 20))
    t_vec, (r_vec, _) = time_it(lambda: stats.histogram(x, bins=20))
    t_np, (r_np, _) = time_it(lambda: np.histogram(x, bins=20))
    assert np.array_equal(r_vec, r_np) and np.array_equal(r_loop, r_np)
    rows.append(("stats.histogram", "n={:,}, 20 bins".format(n_h),
                 t_loop, t_vec, t_np))

    # 5. axis-wise median ---------------------------------------------------
    r, c = sizes["median_shape"]
    x2d = rng.normal(size=(r, c))
    t_loop, r_loop = time_it(lambda: loop_column_medians(x2d))
    t_vec, r_vec = time_it(lambda: stats.median(x2d, axis=0))
    t_np, r_np = time_it(lambda: np.median(x2d, axis=0))
    assert np.allclose(r_loop, r_vec) and np.allclose(r_vec, r_np)
    rows.append(("stats.median (axis=0)", "{:,} x {:,}".format(r, c),
                 t_loop, t_vec, t_np))

    # 6. streaming variance ---------------------------------------------------
    n_s = sizes["n_stream"]
    values = rng.normal(size=n_s)

    def vec_stream():
        rs = stats.RunningStats()
        for chunk in np.array_split(values, 100):
            rs.update(chunk)
        return rs.mean, rs.variance()

    t_loop, r_loop = time_it(lambda: loop_welford(values))
    t_vec, r_vec = time_it(vec_stream)
    t_np = time_it(lambda: (values.mean(), values.var()))[0]
    assert np.allclose(r_loop, r_vec)
    rows.append(("stats.RunningStats (100 chunks)", "n={:,}".format(n_s),
                 t_loop, t_vec, t_np))

    # 7. gradient cost ---------------------------------------------------------
    dim = sizes["grad_dim"]
    xg = rng.normal(size=dim)

    def f(v):
        return float(np.sum(v ** 2))

    t_loop, r_loop = time_it(lambda: loop_grad(f, xg, 1e-5))
    t_vec, r_vec = time_it(lambda: optim.grad(f, xg, method="forward"))
    assert np.allclose(r_loop, r_vec, atol=1e-3)
    rows.append(("optim.grad (forward)", "dim={}".format(dim), t_loop, t_vec, None))

    return rows


def bench_stability():
    """Numerical stability checks. Returns a list of (case, value, true value) rows."""
    rng = np.random.default_rng(SEED)
    rows = []

    # variance of data with a huge offset and a tiny spread (true variance ~1.0)
    values = 1e9 + rng.normal(size=100_000)
    true_var = np.var(values - 1e9)
    rs = stats.RunningStats()
    for chunk in np.array_split(values, 100):
        rs.update(chunk)
    rows.append(("Variance, x = 1e9 + N(0,1): sum-of-squares formula",
                 naive_variance(values), true_var))
    rows.append(("Variance, x = 1e9 + N(0,1): RunningStats (Welford)",
                 rs.variance(), true_var))
    return rows


def bench_grad_error():
    """Error of central vs forward differences for f(x) = sin(x) at x = 1."""
    x0 = np.array([1.0])
    exact = np.cos(1.0)
    rows = []
    for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
        g_c = optim.grad(lambda v: np.sin(v[0]), x0, h=h, method="central")[0]
        g_f = optim.grad(lambda v: np.sin(v[0]), x0, h=h, method="forward")[0]
        rows.append((h, abs(g_c - exact), abs(g_f - exact)))
    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="smaller sizes for a fast run")
    args = parser.parse_args()

    if args.quick:
        sizes = {"n_labels": 20_000, "n_auc": 1_000, "n_hist": 50_000,
                 "median_shape": (500, 50), "n_stream": 50_000, "grad_dim": 50}
    else:
        sizes = {"n_labels": 200_000, "n_auc": 3_000, "n_hist": 500_000,
                 "median_shape": (2_000, 200), "n_stream": 500_000, "grad_dim": 200}

    print("NumCompute benchmarks: metrics, optim, stats")
    print("Python {} | NumPy {} | {} | seed {} | median of {} runs".format(
        platform.python_version(), np.__version__, platform.platform(), SEED, REPEATS))
    print()

    print("1) Speed: plain-Python loop vs numcompute (vectorised) vs NumPy")
    print("| Function | Input | Python loop | numcompute | NumPy | Speed-up vs loop |")
    print("|---|---|---|---|---|---|")
    for name, size, t_loop, t_vec, t_np in bench_speed(sizes):
        print("| {} | {} | {} | {} | {} | {:,.0f}x |".format(
            name, size, fmt_time(t_loop), fmt_time(t_vec),
            fmt_time(t_np) if t_np is not None else "-", t_loop / t_vec))
    print()
    print("(The AUC loop is timed once: it is O(n^2). All versions were checked to agree.)")
    print()

    print("2) Numerical stability")
    print("| Case | Result | True value |")
    print("|---|---|---|")
    for case, value, true in bench_stability():
        print("| {} | {:.6f} | {:.6f} |".format(case, value, true))
    print()

    print("3) Finite-difference error for d/dx sin(x) at x = 1 (exact = cos 1)")
    print("| h | central error | forward error |")
    print("|---|---|---|")
    for h, err_c, err_f in bench_grad_error():
        print("| {:.0e} | {:.2e} | {:.2e} |".format(h, err_c, err_f))


if __name__ == "__main__":
    main()
