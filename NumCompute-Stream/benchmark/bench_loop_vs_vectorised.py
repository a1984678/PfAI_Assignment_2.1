"""Benchmark: loop-based vs. vectorised implementations.

Compare the performance of loop-based and NumPy-vectorised implementations
under streaming scenarios:

1. Tree split search: course-style loop (one mask per feature and threshold, from the
   tree spike) vs ``tree._best_split`` (sort + cumulative class counts, all at once).
2. Tree predict: walk one row at a time vs ``predict`` (all rows move one level per step).
3. Streaming metrics over a chunked stream: per-sample Python loops vs ``Accuracy``,
   ``F1(average="macro")`` and ``StreamingAUC`` updated once per chunk.

Every pair is checked to give the same result before it is timed (``compare`` raises
otherwise). Timings are the minimum over interleaved runs (at least MIN_ROUNDS
rounds and MIN_SECONDS in total, so cheap cases get more runs): on a shared machine noise
only ever adds time, so the minimum is the most reliable estimate (the advice in the
``timeit`` docs); interleaving gives both versions the same machine conditions.

Run: ``python benchmark/bench_loop_vs_vectorised.py`` (or ``run_all.py`` for results.md).
"""

import time

import numpy as np

from numcompute_stream.metrics import F1, Accuracy, StreamingAUC
from numcompute_stream.tree import DecisionTreeClassifier, _best_split, _impurity

SEED = 0
MIN_ROUNDS = 5
MIN_SECONDS = 1.0


def compare(loop_func, vec_func):
    """
    Check two zero-argument functions agree, then time them in turns.

    Returns:
    -------
    - dict with 'loop' and 'vectorised' (minimum seconds) and 'speedup' (loop / vectorised)

    Raises:
    ------
    - ValueError: if the results differ (a fast wrong answer is not a speed-up)
    """
    expected, got = np.asarray(loop_func()), np.asarray(vec_func())   # also the warm-up
    if expected.shape != got.shape or not np.allclose(expected, got, equal_nan=True):
        raise ValueError("loop and vectorised versions return different results.")
    times = {"loop": [], "vectorised": []}
    start_all = time.perf_counter()
    while (len(times["loop"]) < MIN_ROUNDS
           or time.perf_counter() - start_all < MIN_SECONDS):
        for key, func in (("loop", loop_func), ("vectorised", vec_func)):
            start = time.perf_counter()
            func()
            times[key].append(time.perf_counter() - start)
    best = {key: min(values) for key, values in times.items()}
    return {**best, "speedup": best["loop"] / best["vectorised"]}


def make_data(n, d, n_classes, seed=SEED):
    """Continuous features; label from a noisy linear rule, cut into n_classes bands."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    score = X @ rng.normal(size=d) + 0.5 * rng.normal(size=n)
    y = np.searchsorted(np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]), score)
    return X, y


# ---------------------------------------------------------------- 1. split search
def best_split_loop(X, y, n_classes, criterion):
    """Course-style split search: every feature, every midpoint, one boolean mask each."""
    n = len(y)
    before = _impurity(np.bincount(y, minlength=n_classes), criterion)
    best, best_gain = None, 0.0
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for a, b in zip(values[:-1], values[1:]):
            t = a / 2 + b / 2
            if t >= b:
                t = a
            mask = X[:, f] <= t
            left = np.bincount(y[mask], minlength=n_classes)
            right = np.bincount(y[~mask], minlength=n_classes)
            after = (mask.sum() * _impurity(left, criterion)
                     + (~mask).sum() * _impurity(right, criterion)) / n
            if before - after > best_gain:               # strict: ties keep the first
                best, best_gain = (f, t), before - after
    return best


def _as_array(split):
    return np.array([np.nan, np.nan]) if split is None else np.array(split, dtype=float)


def bench_split(sizes=(250, 1000, 4000), d=8, n_classes=3):
    rows = []
    for n in sizes:
        X, y = make_data(n, d, n_classes)
        res = compare(
            lambda: _as_array(best_split_loop(X, y, n_classes, "gini")),
            lambda: _as_array(_best_split(X, y, n_classes, "gini")))
        rows.append((f"Split search (n={n}, d={d}, k={n_classes})", res))
    return rows


# ---------------------------------------------------------------- 2. tree predict
def predict_loop(tree, X):
    """Walk each row from the root to its leaf separately."""
    out = np.empty(len(X), dtype=tree.classes_.dtype)
    for i, row in enumerate(X):
        node = 0
        while tree.feature_[node] != -1:
            if row[tree.feature_[node]] <= tree.threshold_[node]:
                node = tree.left_[node]
            else:
                node = tree.right_[node]
        out[i] = tree.classes_[tree.value_[node].argmax()]
    return out


def bench_predict(n_train=5000, n_pred=20000, d=8, max_depth=8):
    X, y = make_data(n_train + n_pred, d, 3)
    tree = DecisionTreeClassifier(max_depth=max_depth).fit(X[:n_train], y[:n_train])
    X_pred = X[n_train:]
    res = compare(lambda: predict_loop(tree, X_pred),
                                     lambda: tree.predict(X_pred))
    return [(f"Tree predict ({n_pred} rows, depth {tree.depth_})", res)]


# ---------------------------------------------------------------- 3. streaming metrics
def metrics_loop(chunks, bins=1000):
    """Accuracy, macro F1 and binned AUC with per-sample Python updates."""
    counts = {}                                          # (true, pred) -> count
    hist = [[0] * bins, [0] * bins]                      # negatives, positives per bin
    for y_true, y_pred, y_score in chunks:
        for t, p, s in zip(y_true.tolist(), y_pred.tolist(), y_score.tolist()):
            counts[t, p] = counts.get((t, p), 0) + 1
            hist[t == 1][min(int(s * bins), bins - 1)] += 1
    total = sum(counts.values())
    acc = sum(c for (t, p), c in counts.items() if t == p) / total
    f1s = []
    for c in sorted({t for t, _ in counts} | {p for _, p in counts}):
        tp = counts.get((c, c), 0)
        fp = sum(v for (t, p), v in counts.items() if p == c and t != c)
        fn = sum(v for (t, p), v in counts.items() if t == c and p != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    neg_below, wins = 0, 0.0
    for neg, pos in zip(*hist):
        wins += pos * neg_below + 0.5 * pos * neg
        neg_below += neg
    auc = wins / (sum(hist[0]) * sum(hist[1]))
    return np.array([acc, sum(f1s) / len(f1s), auc])


def metrics_vectorised(chunks):
    acc, f1, auc = Accuracy(), F1(average="macro"), StreamingAUC()
    for y_true, y_pred, y_score in chunks:
        acc.update(y_true, y_pred)
        f1.update(y_true, y_pred)
        auc.update(y_true, y_score)
    return np.array([acc.result(), f1.result(), auc.result()])


def bench_metrics(chunk_sizes=(200, 2000, 20000), n_chunks=50):
    rng = np.random.default_rng(SEED)
    rows = []
    for size in chunk_sizes:
        chunks = []
        for _ in range(n_chunks):
            y_true = rng.integers(0, 2, size)
            y_score = np.clip(0.3 * y_true + rng.uniform(0, 0.7, size), 0, 1)
            chunks.append((y_true, (y_score > 0.5).astype(np.int64), y_score))
        res = compare(lambda: metrics_loop(chunks),
                                         lambda: metrics_vectorised(chunks))
        rows.append((f"Accuracy + macro F1 + AUC ({n_chunks} chunks x {size})", res))
    return rows


# ---------------------------------------------------------------- report
def run():
    """Run all comparisons and return the results as a markdown section."""
    rows = bench_split() + bench_predict() + bench_metrics()
    lines = ["## (a) Loop-based vs. vectorised implementations", "",
             f"Minimum over interleaved runs (at least {MIN_ROUNDS} rounds and "
             f"{MIN_SECONDS:g} s per case) after 1 warm-up. Each pair was checked to return "
             "the same result before timing.", "",
             "| Operation | Loop (ms) | Vectorised (ms) | Speed-up |",
             "|---|---:|---:|---:|"]
    for name, res in rows:
        lines.append(f"| {name} | {res['loop'] * 1e3:.1f} | {res['vectorised'] * 1e3:.2f} "
                     f"| {res['speedup']:.1f}x |")
    lines += ["", "- Split search: the loop is O(n^2 d) (a mask over n rows for each of ~n "
              "thresholds), the vectorised search O(n d log n), so the gap grows with n.",
              "- Metrics: the loop baseline is a bare dict counter with no input checks. Each "
              "vectorised metric validates its chunk (label set via np.unique, mixed-type "
              "check), a fixed cost per chunk, so it loses on tiny chunks and wins on large "
              "ones."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(run())
