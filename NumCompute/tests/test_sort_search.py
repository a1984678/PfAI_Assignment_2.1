import numpy as np
import pytest

from numcompute import sort_search


# ---------------------------------------------------------------------
# stable_sort
# ---------------------------------------------------------------------

class TestStableSort:

    def test_basic(self):
        np.testing.assert_array_equal(sort_search.stable_sort([3, 1, 2]), [1, 2, 3])

    def test_ties_keep_original_order(self):
        """Equal keys keep their order: the indices of the two 1s stay 1 then 3."""
        _, idx = sort_search.stable_sort([2, 1, 3, 1], return_indices=True)
        np.testing.assert_array_equal(idx, [1, 3, 0, 2])

    def test_descending_is_also_stable(self):
        """Reversing an ascending sort would swap the 1s; a stable descending sort must not."""
        result, idx = sort_search.stable_sort([2, 1, 3, 1], descending=True,
                                              return_indices=True)
        np.testing.assert_array_equal(result, [3, 2, 1, 1])
        np.testing.assert_array_equal(idx, [2, 0, 1, 3])

    def test_axis(self):
        X = np.array([[3, 1], [1, 2]])
        np.testing.assert_array_equal(sort_search.stable_sort(X, axis=0), [[1, 1], [3, 2]])
        np.testing.assert_array_equal(sort_search.stable_sort(X, axis=1), [[1, 3], [1, 2]])
        np.testing.assert_array_equal(sort_search.stable_sort(X, axis=None), [1, 1, 2, 3])

    def test_strings(self):
        np.testing.assert_array_equal(sort_search.stable_sort(["b", "c", "a"]), ["a", "b", "c"])

    def test_nan_goes_last(self):
        result = sort_search.stable_sort([2.0, np.nan, 1.0])
        np.testing.assert_array_equal(result, [1.0, 2.0, np.nan])

    def test_empty_returns_empty(self):
        assert sort_search.stable_sort(np.array([])).shape == (0,)

    def test_input_not_mutated(self):
        x = np.array([3, 1, 2])
        sort_search.stable_sort(x)
        np.testing.assert_array_equal(x, [3, 1, 2])

    def test_non_contiguous_input(self):
        x = np.array([5, 0, 3, 0, 1, 0])[::2]
        np.testing.assert_array_equal(sort_search.stable_sort(x), [1, 3, 5])

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="axis"):
            sort_search.stable_sort([1, 2], axis=3)

    def test_object_dtype_raises(self):
        with pytest.raises(ValueError, match="numbers, strings or dates"):
            sort_search.stable_sort(np.array([1, "a", None], dtype=object))


# ---------------------------------------------------------------------
# multikey_sort
# ---------------------------------------------------------------------

class TestMultikeySort:

    X = np.array([[2, 9],
                  [1, 5],
                  [2, 3],
                  [1, 7]])

    def test_primary_then_secondary(self):
        result = sort_search.multikey_sort(self.X, [0, 1])
        np.testing.assert_array_equal(result, [[1, 5], [1, 7], [2, 3], [2, 9]])

    def test_key_order_matters(self):
        result = sort_search.multikey_sort(self.X, [1, 0])
        np.testing.assert_array_equal(result[:, 1], [3, 5, 7, 9])

    def test_mixed_directions(self):
        result = sort_search.multikey_sort(self.X, [0, 1], ascending=[False, True])
        np.testing.assert_array_equal(result, [[2, 3], [2, 9], [1, 5], [1, 7]])

    def test_descending_strings(self):
        names = np.array([["b", "2"], ["a", "1"], ["b", "1"], ["a", "2"]])
        result = sort_search.multikey_sort(names, [0, 1], ascending=[False, True])
        np.testing.assert_array_equal(result[:, 0], ["b", "b", "a", "a"])
        np.testing.assert_array_equal(result[:, 1], ["1", "2", "1", "2"])

    def test_full_ties_are_stable(self):
        X = np.array([[1, 0], [0, 0], [1, 0]])
        _, order = sort_search.multikey_sort(X, [1], return_indices=True)
        np.testing.assert_array_equal(order, [0, 1, 2])

    def test_single_int_key(self):
        result = sort_search.multikey_sort(self.X, 1)
        np.testing.assert_array_equal(result[:, 1], [3, 5, 7, 9])

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            sort_search.multikey_sort(np.array([1, 2, 3]), [0])

    def test_bad_key_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            sort_search.multikey_sort(self.X, [5])

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="at least one column"):
            sort_search.multikey_sort(self.X, [])

    def test_ascending_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="one entry per key"):
            sort_search.multikey_sort(self.X, [0, 1], ascending=[True])


# ---------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------

class TestTopk:

    x = np.array([4, 1, 7, 3, 7, 2])

    def test_largest(self):
        values, idx = sort_search.topk(self.x, 3)
        np.testing.assert_array_equal(values, [7, 7, 4])
        np.testing.assert_array_equal(idx, [2, 4, 0])

    def test_smallest(self):
        values, idx = sort_search.topk(self.x, 2, largest=False)
        np.testing.assert_array_equal(values, [1, 2])
        np.testing.assert_array_equal(idx, [1, 5])

    def test_values_only(self):
        values = sort_search.topk(self.x, 2, return_indices=False)
        np.testing.assert_array_equal(values, [7, 7])

    def test_tie_at_cutoff_keeps_earliest(self):
        """Three 5s compete for two places: the first two positions win, every run."""
        values, idx = sort_search.topk([5, 9, 5, 5], 3)
        np.testing.assert_array_equal(values, [9, 5, 5])
        np.testing.assert_array_equal(idx, [1, 0, 2])

    def test_all_equal_values(self):
        _, idx = sort_search.topk(np.zeros(5), 2)
        np.testing.assert_array_equal(idx, [0, 1])

    @pytest.mark.parametrize("k", [0, 6])
    def test_extreme_k(self, k):
        values, idx = sort_search.topk(self.x, k)
        assert values.shape == (k,)
        np.testing.assert_array_equal(values, np.sort(self.x)[::-1][:k])

    def test_matches_full_sort_on_random_data(self):
        x = np.random.default_rng(0).normal(size=1000)
        values, idx = sort_search.topk(x, 10)
        np.testing.assert_array_equal(values, np.sort(x)[::-1][:10])
        np.testing.assert_array_equal(x[idx], values)

    @pytest.mark.parametrize("k", [-1, 7, 2.5, True])
    def test_invalid_k_raises(self, k):
        with pytest.raises(ValueError, match="k must be"):
            sort_search.topk(self.x, k)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="must not contain NaN"):
            sort_search.topk([1.0, np.nan], 1)

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            sort_search.topk(np.ones((2, 2)), 1)


# ---------------------------------------------------------------------
# quickselect
# ---------------------------------------------------------------------

class TestQuickselect:

    def test_matches_sort_for_every_k(self):
        x = np.array([9, 2, 7, 2, 5, 1, 8])
        expected = np.sort(x)
        for k in range(x.size):
            assert sort_search.quickselect(x, k) == expected[k]

    def test_many_duplicates(self):
        x = np.array([3] * 50 + [1, 5])
        assert sort_search.quickselect(x, 0) == 1
        assert sort_search.quickselect(x, 25) == 3
        assert sort_search.quickselect(x, 51) == 5

    def test_random_data(self):
        x = np.random.default_rng(1).normal(size=501)
        assert sort_search.quickselect(x, 250) == np.median(x)

    def test_single_value(self):
        assert sort_search.quickselect([4.5], 0) == 4.5

    def test_input_not_mutated(self):
        x = np.array([3, 1, 2])
        sort_search.quickselect(x, 1)
        np.testing.assert_array_equal(x, [3, 1, 2])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            sort_search.quickselect(np.array([]), 0)

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError, match="k must be between 0 and 2"):
            sort_search.quickselect([1, 2, 3], 3)


# ---------------------------------------------------------------------
# binary_search
# ---------------------------------------------------------------------

class TestBinarySearch:

    arr = np.array([1, 3, 3, 3, 8])

    def test_found(self):
        assert sort_search.binary_search(self.arr, 8) == (4, True)

    def test_duplicates_give_first_position(self):
        assert sort_search.binary_search(self.arr, 3) == (1, True)

    def test_missing_gives_insertion_point(self):
        assert sort_search.binary_search(self.arr, 5) == (4, False)

    def test_smaller_and_larger_than_everything(self):
        assert sort_search.binary_search(self.arr, -10) == (0, False)
        assert sort_search.binary_search(self.arr, 99) == (5, False)

    def test_many_values_at_once(self):
        idx, found = sort_search.binary_search(self.arr, [0, 3, 9])
        np.testing.assert_array_equal(idx, [0, 1, 5])
        np.testing.assert_array_equal(found, [False, True, False])

    def test_empty_array(self):
        assert sort_search.binary_search([], 5) == (0, False)

    def test_strings(self):
        assert sort_search.binary_search(["apple", "kiwi", "pear"], "kiwi") == (1, True)

    def test_nan_never_found(self):
        idx, found = sort_search.binary_search([1.0, 2.0, np.nan], np.nan)
        assert found is False

    def test_check_sorted_raises_on_unsorted(self):
        with pytest.raises(ValueError, match="not sorted"):
            sort_search.binary_search([3, 1, 2], 1, check_sorted=True)

    def test_check_sorted_allows_trailing_nan(self):
        assert sort_search.binary_search([1.0, 2.0, np.nan], 2.0, check_sorted=True) == (1, True)

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            sort_search.binary_search(np.ones((2, 2)), 1)

    def test_incomparable_value_raises(self):
        with pytest.raises(ValueError, match="cannot be compared"):
            sort_search.binary_search([1, 2, 3], "a")
            