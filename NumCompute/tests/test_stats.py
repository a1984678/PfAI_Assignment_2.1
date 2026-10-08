import numpy as np
import pytest

from numcompute import stats


# ---------------------------------------------------------------------
# Contract tests (must hold for every reducing statistic)
# ---------------------------------------------------------------------

REDUCERS = [stats.mean, stats.median, stats.std, stats.min, stats.max]

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])


@pytest.mark.parametrize("func", REDUCERS, ids=lambda f: f.__name__)
def test_axis_shapes(func):
    """axis=None -> float, axis=0 -> (n_cols,), axis=1 -> (n_rows,)."""
    assert isinstance(func(X), float)
    assert func(X, axis=0).shape == (3,)
    assert func(X, axis=1).shape == (2,)
    np.testing.assert_allclose(func(X, axis=-1), func(X, axis=1))


@pytest.mark.parametrize("func", REDUCERS, ids=lambda f: f.__name__)
def test_empty_raises(func):
    with pytest.raises(ValueError, match="Input must not be empty."):
        func(np.array([]))


@pytest.mark.parametrize("func", REDUCERS, ids=lambda f: f.__name__)
def test_non_numeric_raises(func):
    with pytest.raises(ValueError, match="Input must be numeric"):
        func(np.array(["a", "b"]))


@pytest.mark.parametrize("func", REDUCERS, ids=lambda f: f.__name__)
def test_axis_out_of_bounds_raises(func):
    with pytest.raises(ValueError, match="out of bounds"):
        func(X, axis=2)


@pytest.mark.parametrize("func", REDUCERS, ids=lambda f: f.__name__)
def test_all_equal_values(func):
    x = np.full(5, 7.0)
    expected = 0.0 if func is stats.std else 7.0
    assert func(x) == pytest.approx(expected)


@pytest.mark.parametrize("func", REDUCERS, ids=lambda f: f.__name__)
def test_all_nan_slice_gives_nan(func):
    x = np.array([[np.nan, 1.0], [np.nan, 3.0]])
    result = func(x, axis=0)
    assert np.isnan(result[0])
    assert not np.isnan(result[1])


@pytest.mark.parametrize("func", REDUCERS, ids=lambda f: f.__name__)
def test_non_contiguous_matches_copy(func):
    x = np.arange(20.0).reshape(4, 5)[:, ::2]   # strided view
    np.testing.assert_allclose(func(x, axis=0), func(x.copy(), axis=0))
    np.testing.assert_allclose(func(x.T, axis=1), func(x.T.copy(), axis=1))


@pytest.mark.parametrize("func", REDUCERS, ids=lambda f: f.__name__)
def test_skipna_false_propagates_nan(func):
    x = np.array([1.0, np.nan, 3.0])
    assert np.isnan(func(x, skipna=False))
    assert not np.isnan(func(x))


# ---------------------------------------------------------------------
# mean / median / std / min / max: known values
# ---------------------------------------------------------------------

def test_known_values():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert stats.mean(x) == pytest.approx(5.0)
    assert stats.median(x) == pytest.approx(4.5)
    assert stats.std(x) == pytest.approx(2.0)          # textbook population std
    assert stats.min(x) == 2.0
    assert stats.max(x) == 9.0


def test_nan_ignored_by_default():
    x = np.array([1.0, np.nan, 3.0])
    assert stats.mean(x) == pytest.approx(2.0)
    assert stats.median(x) == pytest.approx(2.0)


def test_std_ddof():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert stats.std(x, ddof=1) == pytest.approx(np.std(x, ddof=1))


def test_std_not_enough_values_gives_nan():
    assert np.isnan(stats.std([5.0], ddof=1))


def test_std_invalid_ddof_raises():
    with pytest.raises(ValueError, match="ddof must be a non-negative integer"):
        stats.std([1.0, 2.0], ddof=-1)


def test_std_large_offset_is_stable():
    """Values near 1e9 with a spread of 1: a naive E[x^2] - E[x]^2 loses all precision."""
    x = 1e9 + np.array([0.0, 1.0, 2.0])
    assert stats.std(x) == pytest.approx(np.sqrt(2 / 3), rel=1e-9)


# ---------------------------------------------------------------------
# quantile
# ---------------------------------------------------------------------

class TestQuantile:

    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert stats.quantile(x, 0.5) == pytest.approx(2.5)
        assert stats.quantile(x, 0.0) == 1.0
        assert stats.quantile(x, 1.0) == 4.0

    def test_multiple_q_shape(self):
        result = stats.quantile(X, [0.25, 0.5, 0.75], axis=0)
        assert result.shape == (3, 3)     # (n_q, n_cols), same convention as NumPy
        np.testing.assert_allclose(result, np.quantile(X, [0.25, 0.5, 0.75], axis=0))

    def test_matches_numpy_with_nans(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(6, 5))
        x[rng.random((6, 5)) < 0.3] = np.nan
        for axis in (None, 0, 1):
            np.testing.assert_allclose(stats.quantile(x, [0.1, 0.5, 0.9], axis=axis),
                                       np.nanquantile(x, [0.1, 0.5, 0.9], axis=axis))

    def test_ties(self):
        assert stats.quantile([3.0, 3.0, 3.0, 10.0], 0.5) == pytest.approx(3.0)

    @pytest.mark.parametrize("q", [-0.1, 1.5, np.nan])
    def test_invalid_q_raises(self, q):
        with pytest.raises(ValueError, match="q values must be between 0 and 1"):
            stats.quantile([1.0, 2.0], q)


# ---------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------

class TestHistogram:

    def test_basic(self):
        counts, edges = stats.histogram([1, 2, 2, 3, 3, 3], bins=3)
        np.testing.assert_array_equal(counts, [1, 2, 3])
        np.testing.assert_allclose(edges, [1, 5 / 3, 7 / 3, 3])

    def test_last_bin_includes_right_edge(self):
        counts, _ = stats.histogram([0.0, 1.0], bins=2)
        np.testing.assert_array_equal(counts, [1, 1])

    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000)
        for bins in (1, 7, 30):
            counts, edges = stats.histogram(x, bins=bins)
            ref_counts, ref_edges = np.histogram(x, bins=bins)
            np.testing.assert_array_equal(counts, ref_counts)
            np.testing.assert_allclose(edges, ref_edges)

    def test_all_equal_values(self):
        counts, edges = stats.histogram([5.0, 5.0, 5.0], bins=2)
        np.testing.assert_allclose(edges, [4.5, 5.0, 5.5])
        assert counts.sum() == 3

    def test_nan_ignored(self):
        counts, _ = stats.histogram([1.0, np.nan, 2.0], bins=2)
        assert counts.sum() == 2

    def test_value_range_drops_outside_values(self):
        counts, _ = stats.histogram([0.0, 0.5, 1.0, 5.0], bins=2, value_range=(0, 1))
        np.testing.assert_array_equal(counts, [1, 2])

    @pytest.mark.parametrize("bins", [0, -3, 2.5, True])
    def test_invalid_bins_raises(self, bins):
        with pytest.raises(ValueError, match="bins must be a positive integer"):
            stats.histogram([1.0, 2.0], bins=bins)

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="every value is NaN"):
            stats.histogram([np.nan, np.nan])

    def test_reversed_range_raises(self):
        with pytest.raises(ValueError, match="low <= high"):
            stats.histogram([1.0, 2.0], value_range=(2, 1))


# ---------------------------------------------------------------------
# RunningStats (Welford streaming)
# ---------------------------------------------------------------------

class TestRunningStats:

    def test_matches_numpy_over_chunks(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=1000)
        rs = stats.RunningStats()
        for chunk in np.array_split(data, 13):
            rs.update(chunk)
        assert rs.count == 1000
        assert rs.mean == pytest.approx(data.mean())
        assert rs.variance(ddof=1) == pytest.approx(data.var(ddof=1))
        assert rs.min == data.min()
        assert rs.max == data.max()

    def test_one_value_at_a_time(self):
        rs = stats.RunningStats()
        for value in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            rs.update(value)
        assert rs.mean == pytest.approx(5.0)
        assert rs.std() == pytest.approx(2.0)

    def test_large_offset_is_stable(self):
        """Welford keeps the variance right where sum-of-squares would fail."""
        data = 1e9 + np.array([0.0, 1.0, 2.0])
        rs = stats.RunningStats().update(data)
        assert rs.variance() == pytest.approx(2 / 3, rel=1e-9)

    def test_merge(self):
        a = stats.RunningStats().update([1.0, 2.0, 3.0])
        b = stats.RunningStats().update([10.0, 20.0])
        a.merge(b)
        all_data = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        assert a.count == 5
        assert a.mean == pytest.approx(all_data.mean())
        assert a.variance() == pytest.approx(all_data.var())

    def test_nan_skipped(self):
        rs = stats.RunningStats().update([1.0, np.nan, 3.0])
        assert rs.count == 2
        assert rs.mean == pytest.approx(2.0)

    def test_empty_state_is_nan(self):
        rs = stats.RunningStats()
        assert rs.count == 0
        assert np.isnan(rs.mean) and np.isnan(rs.variance())
        assert np.isnan(rs.min) and np.isnan(rs.max)

    def test_infinite_raises(self):
        with pytest.raises(ValueError, match="values must be finite"):
            stats.RunningStats().update([1.0, np.inf])

    def test_merge_wrong_type_raises(self):
        with pytest.raises(ValueError, match="Can only merge another RunningStats"):
            stats.RunningStats().merge([1.0, 2.0])
            