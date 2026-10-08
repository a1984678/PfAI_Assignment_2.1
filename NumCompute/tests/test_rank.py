import numpy as np
import pytest

from numcompute import rank


# ---------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------

class TestRank:

    data = np.array([10, 20, 20, 30])

    def test_average(self):
        np.testing.assert_array_equal(rank.rank(self.data), [1, 2.5, 2.5, 4])

    def test_dense(self):
        np.testing.assert_array_equal(rank.rank(self.data, "dense"), [1, 2, 2, 3])

    def test_ordinal(self):
        np.testing.assert_array_equal(rank.rank(self.data, "ordinal"), [1, 2, 3, 4])

    def test_triple_tie(self):
        x = np.array([5, 1, 5, 5, 0])
        np.testing.assert_array_equal(rank.rank(x), [4, 2, 4, 4, 1])
        np.testing.assert_array_equal(rank.rank(x, "dense"), [3, 2, 3, 3, 1])
        np.testing.assert_array_equal(rank.rank(x, "ordinal"), [3, 2, 4, 5, 1])

    def test_unsorted_input_maps_back_to_original_positions(self):
        np.testing.assert_array_equal(rank.rank([30, 10, 20]), [3, 1, 2])

    @pytest.mark.parametrize("method", ["average", "dense", "ordinal"])
    def test_all_equal_values(self, method):
        expected = {"average": [2, 2, 2], "dense": [1, 1, 1], "ordinal": [1, 2, 3]}
        np.testing.assert_array_equal(rank.rank([7, 7, 7], method), expected[method])

    def test_dtypes(self):
        assert rank.rank(self.data).dtype == np.float64
        assert rank.rank(self.data, "dense").dtype == np.int64

    def test_strings(self):
        np.testing.assert_array_equal(rank.rank(["b", "a", "b"], "dense"), [2, 1, 2])

    def test_axis(self):
        X = np.array([[3, 1, 2],
                      [1, 1, 5]])
        np.testing.assert_array_equal(rank.rank(X, axis=1), [[3, 1, 2], [1.5, 1.5, 3]])
        np.testing.assert_array_equal(rank.rank(X, axis=0), [[2, 1.5, 1], [1, 1.5, 2]])
        assert rank.rank(X).shape == (6,)  # axis=None ranks the flattened array

    def test_ranks_sum_is_fixed(self):
        """Average ranks of n values always add up to n(n+1)/2, ties or not."""
        x = np.random.default_rng(0).integers(0, 5, 100)
        assert rank.rank(x).sum() == 100 * 101 / 2

    def test_non_contiguous_input(self):
        x = np.array([3, 0, 1, 0, 2, 0])[::2]
        np.testing.assert_array_equal(rank.rank(x), [3, 1, 2])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Input must not be empty."):
            rank.rank(np.array([]))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="must not contain NaN"):
            rank.rank([1.0, np.nan])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            rank.rank(self.data, "min")

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            rank.rank(self.data, axis=1)


# ---------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------

METHODS = ["linear", "lower", "higher", "midpoint"]


class TestPercentile:

    data = np.array([1.0, 2.0, 3.0, 4.0])

    @pytest.mark.parametrize("method, expected", [
        ("linear", 1.75), ("lower", 1.0), ("higher", 2.0), ("midpoint", 1.5)])
    def test_methods_at_25th(self, method, expected):
        """25% of the way along 4 values is position 0.75, between 1 and 2."""
        assert rank.percentile(self.data, 25, method) == expected

    @pytest.mark.parametrize("method", METHODS)
    def test_matches_numpy(self, method):
        x = np.random.default_rng(0).normal(size=(20, 7))
        q = [0, 10, 33.3, 50, 90, 100]
        for axis in [None, 0, 1]:
            np.testing.assert_allclose(rank.percentile(x, q, method, axis=axis),
                                       np.percentile(x, q, axis=axis, method=method))

    @pytest.mark.parametrize("method", METHODS)
    def test_nan_ignored_like_nanpercentile(self, method):
        x = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        np.testing.assert_allclose(rank.percentile(x, 40, method, axis=1),
                                   np.nanpercentile(x, 40, axis=1, method=method))

    def test_extreme_q_give_min_and_max(self):
        assert rank.percentile(self.data, 0) == 1.0
        assert rank.percentile(self.data, 100) == 4.0

    def test_ties(self):
        assert rank.percentile([2, 2, 2, 9], 50) == 2.0

    def test_single_value(self):
        assert rank.percentile([7], 30, "midpoint") == 7.0

    def test_shapes(self):
        X = np.ones((3, 5))
        assert isinstance(rank.percentile(X, 50), float)
        assert rank.percentile(X, 50, axis=0).shape == (5,)
        assert rank.percentile(X, [25, 75], axis=1).shape == (2, 3)

    def test_all_nan_slice_gives_nan(self):
        result = rank.percentile(np.array([[np.nan, np.nan], [1.0, 2.0]]), 50, axis=1)
        assert np.isnan(result[0]) and result[1] == 1.5

    def test_integer_input(self):
        assert rank.percentile(np.array([1, 2, 3, 4]), 50) == 2.5

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Input must not be empty."):
            rank.percentile(np.array([]), 50)

    @pytest.mark.parametrize("q", [-1, 101, np.nan])
    def test_invalid_q_raises(self, q):
        with pytest.raises(ValueError, match="between 0 and 100"):
            rank.percentile(self.data, q)

    def test_invalid_interpolation_raises(self):
        with pytest.raises(ValueError, match="interpolation must be one of"):
            rank.percentile(self.data, 50, "nearest")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="must be numeric"):
            rank.percentile(["a", "b"], 50)
            