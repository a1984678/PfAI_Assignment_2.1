"""Benchmark: effect of ``buffer_size`` on the decision tree.

Compare different ``buffer_size`` values (e.g., how accuracy and retraining
time change with the buffer size) under streaming scenarios.

Stream: 24,000 rows, 4 features in [-1, 1], 5% label noise, 3% missing values, chunks of
100. Concept drift at chunk 120: the label rule switches from a circle
(x0^2 + x1^2 < 0.5) to a line (x0 + x2 > 0). Model: Imputer -> StandardScaler ->
DecisionTreeClassifier(max_depth=6), trained test-then-train with StreamTrainer.

Rolling accuracy is used, not cumulative: cumulative accuracy averages over the whole
stream and hides the drift. Each buffer size is run on SEEDS streams and averaged.

Run: ``python benchmark/bench_buffer_size.py`` (or ``run_all.py`` for results.md).
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)

from numcompute_stream.io import iter_chunks  # noqa: E402
from numcompute_stream.metrics import Accuracy  # noqa: E402
from numcompute_stream.pipeline import Pipeline  # noqa: E402
from numcompute_stream.preprocessing import Imputer, StandardScaler  # noqa: E402
from numcompute_stream.stream import StreamTrainer  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier  # noqa: E402

BUFFER_SIZES = (100, 300, 1000, 2000, 4000)
SEEDS = (0, 1, 2)
N, CHUNK, DRIFT = 24000, 100, 120          # DRIFT = first chunk with the new rule
STABLE = slice(80, DRIFT)                  # every buffer size has a tree by chunk 40
FINAL = slice(-40, None)                   # the level each model settles at after drift
SMOOTH = 5                                 # window (chunks) of the recovery curve
TOLERANCE = 0.05                           # "recovered" = within 5 points of FINAL
TIMING = ("retrain_ms", "fit_s")
FIGURE = Path(__file__).with_name("buffer_size_drift.png")
COLORS = ("#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")   # ordinal: small->large


def make_stream(seed):
    """The drift stream from the demo notebook (circle rule, then a linear rule)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(N, 4))
    y = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 < 0.5, "in", "out")
    late = slice(DRIFT * CHUNK, None)
    y[late] = np.where(X[late, 0] + X[late, 2] > 0, "in", "out")
    flip = rng.random(N) < 0.05
    y[flip] = np.where(y[flip] == "in", "out", "in")
    X[rng.random(X.shape) < 0.03] = np.nan
    return X, y


def run_stream(buffer_size, seed):
    """Test-then-train over one stream. Returns per-chunk logs as arrays."""
    X, y = make_stream(seed)
    tree = DecisionTreeClassifier(max_depth=6, buffer_size=buffer_size)
    pipe = Pipeline([("impute", Imputer()), ("scale", StandardScaler()), ("tree", tree)])
    trainer = StreamTrainer(pipe, metrics={"rolling": Accuracy(window=10),
                                           "smooth": Accuracy(window=SMOOTH)})
    retrained = []
    for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=CHUNK):
        trainer.score_chunk(X_chunk, y_chunk)
        before = tree.n_retrains_
        trainer.fit_chunk(X_chunk, y_chunk)
        retrained.append(tree.n_retrains_ > before)
    log = {key: trainer.history(key)
           for key in ("accuracy", "rolling", "smooth", "fit_time", "model_bytes")}
    log["retrained"] = np.array(retrained)
    return log


def summarise(log):
    """Accuracy before/after drift, dip, recovery (chunks), retrain time, memory."""
    final = np.mean(log["accuracy"][FINAL])
    # start once the smoothing window holds only post-drift chunks
    back = np.flatnonzero(log["smooth"][DRIFT + SMOOTH - 1:] >= final - TOLERANCE)
    return {"stable": np.mean(log["accuracy"][STABLE]),
            "after": np.mean(log["accuracy"][DRIFT:]),
            "final": final,
            "dip": np.min(log["rolling"][DRIFT:]),
            "recovery": SMOOTH + back[0] if back.size else np.inf,
            "retrain_ms": 1e3 * np.median(log["fit_time"][log["retrained"]]),
            "n_retrains": int(log["retrained"].sum()),
            "fit_s": log["fit_time"].sum(),
            "peak_kb": log["model_bytes"].max() / 1024}


def plot(logs, path=FIGURE):
    """Rolling accuracy of every buffer size on one stream, drift marked."""
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    chunks = np.arange(N // CHUNK)
    for (size, log), color in zip(logs.items(), COLORS):
        ax.plot(chunks, log["rolling"], color=color, linewidth=2, label=f"buffer {size}")
    ax.axvline(DRIFT, color="#8a8a86", linestyle="--", linewidth=1)
    ax.text(DRIFT + 2, 0.97, "concept drift", color="#5f5e5a", fontsize=9, va="top")
    ax.set(title="Rolling accuracy (last 10 chunks) by buffer_size, seed 0",
           xlabel="Chunk (100 samples each)", ylabel="Accuracy", ylim=(0.4, 1.0))
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run():
    """Sweep buffer_size, save the figure and return the results as a markdown section."""
    rows, first_seed = {}, {}
    for size in BUFFER_SIZES:
        logs = [run_stream(size, seed) for seed in SEEDS]
        first_seed[size] = logs[0]
        runs = [summarise(log) for log in logs]
        # timings: minimum over seeds (machine noise only adds time); the rest: mean
        rows[size] = {key: (min if key in TIMING else np.mean)([r[key] for r in runs])
                      for key in runs[0]}
    figure = plot(first_seed)

    lines = ["## (b) Effect of `buffer_size` (accuracy and retraining time)", "",
             f"Drift stream ({N} rows, chunks of {CHUNK}, rule changes at chunk {DRIFT}), "
             "Imputer -> StandardScaler -> DecisionTreeClassifier(max_depth=6), "
             f"test-then-train. Mean over {len(SEEDS)} stream seeds (timings: minimum over "
             "seeds, as noise on this shared machine only adds time).", "",
             "| buffer_size | Accuracy before drift | Accuracy after drift | "
             "Final accuracy | Lowest rolling acc. | Recovery (chunks) | Retrains | "
             "Retrain time (ms) | "
             "Total fit time (s) | Peak model memory (kB) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for size, r in rows.items():
        lines.append(f"| {size} | {r['stable']:.3f} | {r['after']:.3f} | {r['final']:.3f} | "
                     f"{r['dip']:.3f} | "
                     f"{r['recovery']:.1f} | {r['n_retrains']:.0f} | {r['retrain_ms']:.1f} "
                     f"| {r['fit_s']:.2f} | {r['peak_kb']:.0f} |")
    lines += ["",
              f"- Accuracy before drift: mean per-chunk accuracy over chunks "
              f"{STABLE.start}-{STABLE.stop - 1}. After drift: chunks {DRIFT} to the end. "
              f"Final: the last {-FINAL.start} chunks (the level each model settles at).",
              "- Lowest rolling acc.: minimum of the 10-chunk rolling accuracy after drift.",
              f"- Recovery: chunks from the drift until the {SMOOTH}-chunk rolling accuracy "
              f"is within {TOLERANCE} of the final accuracy (so at least {SMOOTH}: the "
              "window must first fill with post-drift chunks).",
              "- Retrain time: median time of the fit_chunk calls that rebuilt the tree.",
              "", "Findings:", "",
              "- A larger buffer gives each tree more data, so it is more accurate while the "
              "stream is stable and once it has adapted (before drift and final columns).",
              "- But the tree only learns the new rule after retraining on a buffer of "
              "post-drift rows, so recovery takes about buffer_size / chunk size chunks "
              "(plus the smoothing window). The drift here falls on a buffer boundary for "
              "every size; in general it can take up to 2 x buffer_size samples. Until then "
              "the old tree is wrong on half the space (accuracy near 0.5).",
              "- The trade-off shows in accuracy after drift: a medium buffer (1000) does "
              "best over the drifted part of the stream.",
              "- Cost: one retrain gets slower as the buffer grows (more rows to sort), but "
              "total fit time falls because there are fewer retrains. Memory grows linearly "
              "with buffer_size (the buffered rows dominate the model's memory).",
              "", f"![Rolling accuracy by buffer_size]({figure.name})"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(run())
