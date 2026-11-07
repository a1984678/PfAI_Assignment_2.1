"""Tests for ``numcompute_stream.stats``.

Cover chunk-based mean, variance, quantiles and histograms (optionally with
sliding window), and the ``update_stats(X_chunk)`` API, checking streaming
results agree with full-batch results.
"""

import numpy as np
import pytest

from numcompute_stream.stats import StreamingHistogram, StreamingQuantiles, StreamingStats


def _stream(X, chunk_size):
    """Feed X to a fresh StreamingStats in chunks of chunk_size rows."""
    s = StreamingStats()
    for start in range(0, len(X), chunk_size):
        s.update_stats(X[start:start + chunk_size])
    return s


@pytest.mark.parametrize("chunk_size", [1, 7, 50, 1000])
def test_streaming_matches_batch(chunk_size):
    """Any chunking (including size-1 chunks and one big chunk) equals the batch result."""
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[0.0, 5.0, -3.0], scale=[1.0, 10.0, 0.1], size=(203, 3))
    s = _stream(X, chunk_size)

    np.testing.assert_array_equal(s.count, [203, 203, 203])
    np.testing.assert_allclose(s.mean, X.mean(axis=0))
    np.testing.assert_allclose(s.variance(), X.var(axis=0))
    np.testing.assert_allclose(s.variance(ddof=1), X.var(axis=0, ddof=1))
    np.testing.assert_allclose(s.std(), X.std(axis=0))
    np.testing.assert_array_equal(s.min, X.min(axis=0))
    np.testing.assert_array_equal(s.max, X.max(axis=0))
    assert s.n_features == 3


def test_update_stats_returns_self_and_shapes_are_per_feature():
    s = StreamingStats()
    assert s.update_stats(np.ones((4, 2))) is s
    for value in (s.count, s.mean, s.min, s.max, s.variance(), s.std()):
        assert value.shape == (2,)


def test_nan_skipped_per_column():
    """Each column counts only its own non-NaN values; matches np.nan* functions."""
    X = np.array([[1.0, np.nan],
                  [2.0, 4.0],
                  [np.nan, 6.0],
                  [4.0, np.nan],
                  [5.0, 9.0]])
    s = _stream(X, 2)
    np.testing.assert_array_equal(s.count, [4, 3])
    np.testing.assert_allclose(s.mean, np.nanmean(X, axis=0))
    np.testing.assert_allclose(s.variance(), np.nanvar(X, axis=0))
    np.testing.assert_array_equal(s.min, np.nanmin(X, axis=0))
    np.testing.assert_array_equal(s.max, np.nanmax(X, axis=0))


def test_all_nan_column_gives_nan_without_warnings_then_recovers():
    """A column with no values yet is NaN (no 0/0 RuntimeWarning); later data fills it in."""
    s = StreamingStats()
    s.update_stats([[1.0, np.nan], [3.0, np.nan]])
    np.testing.assert_array_equal(s.count, [2, 0])
    assert np.isnan(s.mean[1]) and np.isnan(s.variance()[1])
    assert np.isnan(s.min[1]) and np.isnan(s.max[1])
    assert s.mean[0] == 2.0

    s.update_stats([[5.0, 10.0]])
    np.testing.assert_array_equal(s.count, [3, 1])
    np.testing.assert_allclose(s.mean, [3.0, 10.0])
    np.testing.assert_allclose(s.variance(), [8.0 / 3.0, 0.0])


def test_ddof_larger_than_count_gives_nan_without_warnings():
    s = StreamingStats().update_stats([[7.0, 1.0]])
    np.testing.assert_array_equal(s.variance(ddof=1), [np.nan, np.nan])
    with pytest.raises(ValueError, match="ddof"):
        s.variance(ddof=-1)


def test_zero_variance_column_is_exactly_zero():
    X = np.column_stack([np.full(100, 3.7), np.arange(100.0)])
    s = _stream(X, 9)
    assert s.variance()[0] == 0.0
    assert s.std()[0] == 0.0


def test_large_offset_is_numerically_stable():
    """Values near 1e9 with spread 1: naive sum-of-squares would lose all precision.

    float64 spacing near 1e9 is ~1.2e-7, so each chunk mean carries ~1e-7 rounding and
    a relative error around 1e-8 is the floor for any streaming method; rtol=1e-6 leaves
    margin while the naive formula is wrong by orders of magnitude.
    """
    rng = np.random.default_rng(1)
    X = 1e9 + rng.normal(size=(1000, 1))
    s = _stream(X, 13)
    np.testing.assert_allclose(s.variance(), X.var(axis=0), rtol=1e-6)

    naive = np.mean(X ** 2, axis=0) - np.mean(X, axis=0) ** 2
    assert not np.allclose(naive, X.var(axis=0), rtol=1e-2)


def test_merge_equals_stats_on_concatenated_data():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(30, 2)), rng.normal(3.0, 2.0, size=(11, 2))
    merged = _stream(A, 4).merge(_stream(B, 5))
    both = np.vstack([A, B])
    np.testing.assert_array_equal(merged.count, [41, 41])
    np.testing.assert_allclose(merged.mean, both.mean(axis=0))
    np.testing.assert_allclose(merged.variance(), both.var(axis=0))


def test_merge_into_empty_and_mismatched_features():
    other = _stream(np.arange(6.0).reshape(3, 2), 2)
    merged = StreamingStats().merge(other)
    np.testing.assert_allclose(merged.mean, [2.0, 3.0])
    with pytest.raises(ValueError, match="features"):
        merged.merge(_stream(np.ones((2, 3)), 1))
    with pytest.raises(ValueError, match="StreamingStats"):
        merged.merge([1.0, 2.0])


def test_results_before_any_data_raise():
    s = StreamingStats()
    assert s.n_features is None
    with pytest.raises(RuntimeError, match="no data"):
        s.mean


@pytest.mark.parametrize("bad, match", [
    ([1.0, 2.0], "2-dimensional"),
    ([[1.0, np.inf]], "infinity"),
    ([["a"]], "numeric"),
])
def test_update_stats_rejects_bad_chunks(bad, match):
    with pytest.raises(ValueError, match=match):
        StreamingStats().update_stats(bad)


def test_feature_count_must_stay_the_same():
    s = StreamingStats().update_stats(np.ones((2, 3)))
    with pytest.raises(ValueError, match="X has 2 features, but 3"):
        s.update_stats(np.ones((2, 2)))


# ---------------------------------------------------------------------
# StreamingQuantiles (reservoir sampling)
# ---------------------------------------------------------------------


def _stream_q(X, chunk_size, k=1000, seed=0):
    """Feed X to a fresh StreamingQuantiles in chunks of chunk_size rows."""
    s = StreamingQuantiles(k=k, seed=seed)
    for start in range(0, len(X), chunk_size):
        s.update_stats(X[start:start + chunk_size])
    return s


@pytest.mark.parametrize("chunk_size", [1, 7, 100])
def test_quantiles_exact_while_stream_fits_in_reservoir(chunk_size):
    """With at most k rows seen, the reservoir holds every row: result == np.quantile."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    s = _stream_q(X, chunk_size, k=100)
    q = [0.0, 0.1, 0.5, 0.9, 1.0]
    np.testing.assert_allclose(s.quantile(q), np.quantile(X, q, axis=0))
    assert s.n_seen == 100


def test_quantiles_long_stream_is_close_and_memory_bounded():
    """10x more rows than k: estimates within a small rank error, sample stays k rows."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20_000, 2))
    s = _stream_q(X, 64, k=1000)
    assert s.sample.shape == (1000, 2)
    assert s.n_seen == 20_000

    q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    est = s.quantile(q)                                  # shape (5, 2)
    for col in range(2):
        rank = np.searchsorted(np.sort(X[:, col]), est[:, col]) / len(X)
        assert np.max(np.abs(rank - q)) < 0.05


def test_reservoir_keeps_early_and_late_rows_equally():
    """Every stream position must have the same chance of being kept (no recency bias)."""
    n, k = 20_000, 2000
    X = np.arange(n, dtype=float).reshape(-1, 1)         # value == position in stream
    s = _stream_q(X, 37, k=k)
    per_decile = np.histogram(s.sample[:, 0], bins=10, range=(0, n))[0]
    assert np.all(np.abs(per_decile - k / 10) < 60)      # expected 200 each


def test_quantiles_seeded_reproducible():
    X = np.random.default_rng(2).normal(size=(5000, 1))
    a, b = _stream_q(X, 50, k=100, seed=3), _stream_q(X, 50, k=100, seed=3)
    c = _stream_q(X, 50, k=100, seed=4)
    np.testing.assert_array_equal(a.sample, b.sample)
    assert not np.array_equal(a.sample, c.sample)


def test_reservoir_samples_whole_rows():
    """Columns are sampled together, so each kept row is a real row of the input."""
    x = np.arange(3000, dtype=float)
    X = np.column_stack([x, 2 * x])
    s = _stream_q(X, 128, k=200)
    np.testing.assert_array_equal(s.sample[:, 1], 2 * s.sample[:, 0])


def test_quantiles_skip_nan_per_column():
    X = np.array([[1.0, np.nan], [2.0, 10.0], [np.nan, 20.0], [4.0, 30.0], [5.0, np.nan]])
    s = _stream_q(X, 2, k=10)
    np.testing.assert_allclose(s.quantile([0.25, 0.5]), np.nanquantile(X, [0.25, 0.5], axis=0))


def test_quantiles_all_nan_column_is_nan_without_warnings():
    s = StreamingQuantiles(k=10).update_stats([[1.0, np.nan], [3.0, np.nan]])
    out = s.quantile(0.5)
    assert out[0] == 2.0 and np.isnan(out[1])


def test_quantile_output_shapes_and_q_validation():
    s = StreamingQuantiles(k=50).update_stats(np.ones((10, 4)))
    assert s.update_stats(np.ones((1, 4))) is s
    assert s.quantile(0.5).shape == (4,)
    assert s.quantile([0.1, 0.9]).shape == (2, 4)
    for bad in (-0.1, 1.5, [0.5, 2.0]):
        with pytest.raises(ValueError, match="q must be in"):
            s.quantile(bad)


def test_quantiles_before_data_and_bad_input():
    s = StreamingQuantiles()
    assert s.n_features is None and s.n_seen == 0
    with pytest.raises(RuntimeError, match="no data"):
        s.quantile(0.5)
    for bad_k in (0, -5, 2.5, True):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            StreamingQuantiles(k=bad_k)
    with pytest.raises(ValueError, match="2-dimensional"):
        s.update_stats([1.0, 2.0])
    s.update_stats(np.ones((3, 2)))
    with pytest.raises(ValueError, match="X has 3 features, but 2"):
        s.update_stats(np.ones((3, 3)))


# ---------------------------------------------------------------------
# StreamingHistogram (fixed bins, optional sliding window over chunks)
# ---------------------------------------------------------------------

def _stream_h(X, chunk_size, **kwargs):
    """Feed X to a fresh StreamingHistogram in chunks of chunk_size rows."""
    h = StreamingHistogram(**kwargs)
    for start in range(0, len(X), chunk_size):
        h.update_stats(X[start:start + chunk_size])
    return h


@pytest.mark.parametrize("chunk_size", [1, 13, 500])
def test_histogram_streaming_matches_np_histogram_per_column(chunk_size):
    X = np.random.default_rng(0).normal(size=(500, 3))
    h = _stream_h(X, chunk_size, bins=8, value_range=(-2.0, 2.0))
    assert h.counts.shape == (3, 8) and h.edges.shape == (3, 9)
    for col in range(3):
        counts, edges = np.histogram(X[:, col], bins=8, range=(-2.0, 2.0))
        np.testing.assert_array_equal(h.counts[col], counts)
        np.testing.assert_allclose(h.edges[col], edges)


def test_histogram_per_feature_ranges():
    X = np.column_stack([np.linspace(0, 1, 50), np.linspace(100, 200, 50)])
    h = _stream_h(X, 7, bins=5, value_range=[(0, 1), (100, 200)])
    for col, rng in enumerate([(0, 1), (100, 200)]):
        np.testing.assert_array_equal(h.counts[col], np.histogram(X[:, col], 5, rng)[0])


def test_histogram_right_edge_inclusive_and_out_of_range_counted():
    """Like np.histogram the last bin includes `high`; outside values are counted apart."""
    X = np.array([[0.0], [0.5], [1.0], [-0.1], [1.1], [2.0]])
    h = _stream_h(X, 2, bins=2, value_range=(0, 1))
    np.testing.assert_array_equal(h.counts, [[1, 2]])
    np.testing.assert_array_equal(h.n_out_of_range, [3])


def test_histogram_skips_nan():
    X = np.array([[np.nan, 1.0], [0.2, np.nan], [0.8, 0.4]])
    h = _stream_h(X, 1, bins=2, value_range=(0, 1))
    np.testing.assert_array_equal(h.counts, [[1, 1], [1, 1]])
    np.testing.assert_array_equal(h.n_out_of_range, [0, 0])


def test_histogram_sliding_window_keeps_only_recent_chunks():
    """window=2: only the last two chunks count, for both counts and out-of-range."""
    h = StreamingHistogram(bins=2, value_range=(0, 1), window=2)
    h.update_stats([[0.1], [0.1], [5.0]])   # dropped once the third chunk arrives
    h.update_stats([[0.9]])
    h.update_stats([[0.2], [0.7]])
    np.testing.assert_array_equal(h.counts, [[1, 2]])
    np.testing.assert_array_equal(h.n_out_of_range, [0])


def test_histogram_window_larger_than_chunks_seen_uses_everything():
    X = np.random.default_rng(1).uniform(size=(40, 2))
    windowed = _stream_h(X, 10, bins=4, value_range=(0, 1), window=100)
    full = _stream_h(X, 10, bins=4, value_range=(0, 1))
    np.testing.assert_array_equal(windowed.counts, full.counts)


def test_histogram_equal_low_high_widened_like_baseline():
    h = _stream_h(np.full((4, 1), 3.0), 2, bins=2, value_range=(3, 3))
    np.testing.assert_allclose(h.edges, [[2.5, 3.0, 3.5]])
    np.testing.assert_array_equal(h.counts, [[0, 4]])


@pytest.mark.parametrize("kwargs, match", [
    (dict(bins=0, value_range=(0, 1)), "bins must be a positive integer"),
    (dict(bins=3, value_range=(1, 0)), "low <= high"),
    (dict(bins=3, value_range=(0, np.inf)), "finite"),
    (dict(bins=3, value_range="ab"), "pair"),
    (dict(bins=3, value_range=(0, 1), window=0), "window must be a positive integer"),
])
def test_histogram_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StreamingHistogram(**kwargs)


def test_histogram_range_count_and_feature_checks():
    h = StreamingHistogram(bins=2, value_range=[(0, 1), (0, 1)])
    with pytest.raises(RuntimeError, match="no data"):
        h.counts
    with pytest.raises(ValueError, match="value_range has 2 pairs but X has 3 features"):
        h.update_stats(np.ones((2, 3)))
    h.update_stats(np.ones((2, 2)))
    with pytest.raises(ValueError, match="X has 1 features, but 2"):
        h.update_stats(np.ones((2, 1)))


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_streaming_stats_any_magnitude(scale):
    """Squares of 1e-200 underflow and of 1e200 overflow; mean and std must still be right."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]]) * scale
    s = _stream(X, 1)
    np.testing.assert_allclose(s.mean, [2.5 * scale])
    np.testing.assert_allclose(s.std(), [np.sqrt(1.25) * scale])


def test_streaming_stats_extreme_values_and_growing_magnitude():
    """1e308 sums overflow naively; a stream whose magnitude jumps up must rescale state."""
    s = _stream(np.array([[-1e308], [1e308], [1e308], [-1e308]]), 2)
    np.testing.assert_allclose(s.mean, [0.0], atol=1e292)
    np.testing.assert_allclose(s.std(), [1e308])

    X = np.array([[1.0], [3.0], [1e200], [3e200]])
    grown = _stream(X, 2)
    np.testing.assert_allclose(grown.mean, [X.mean()])
    np.testing.assert_allclose(grown.std(), [np.std(X / 1e200) * 1e200])
