import numpy as np
import pytest

from numcompute import utils


# ---------------------------------------------------------------------
# pairwise_distances
# ---------------------------------------------------------------------

class TestPairwiseDistances:

    X = np.array([[0.0, 0.0], [3.0, 4.0]])

    def test_euclidean_known_value(self):
        D = utils.pairwise_distances(self.X)
        np.testing.assert_allclose(D, [[0, 5], [5, 0]])

    def test_manhattan_known_value(self):
        D = utils.pairwise_distances(self.X, metric="manhattan")
        np.testing.assert_allclose(D, [[0, 7], [7, 0]])

    def test_cosine_known_values(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
        D = utils.pairwise_distances(X, metric="cosine")
        np.testing.assert_allclose(D[0], [0, 1, 0], atol=1e-12)

    def test_cosine_zero_vector_has_distance_one(self):
        D = utils.pairwise_distances([[0.0, 0.0], [1.0, 1.0]], metric="cosine")
        assert D[0, 1] == 1.0 and D[0, 0] == 0.0

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
    def test_matches_brute_force_and_is_symmetric(self, metric):
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(6, 4)), rng.normal(size=(5, 4))
        D = utils.pairwise_distances(X, Y, metric)
        assert D.shape == (6, 5)
        for i in range(6):
            for j in range(5):
                a, b = X[i], Y[j]
                expected = {"euclidean": np.linalg.norm(a - b),
                            "manhattan": np.abs(a - b).sum(),
                            "cosine": 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))}[metric]
                assert D[i, j] == pytest.approx(expected)
        S = utils.pairwise_distances(X, metric=metric)
        np.testing.assert_allclose(S, S.T)

    def test_self_distance_is_exactly_zero(self):
        X = np.random.default_rng(1).normal(size=(5, 3)) * 1e6
        assert (np.diag(utils.pairwise_distances(X)) == 0).all()

    def test_never_negative_or_nan(self):
        X = np.full((4, 3), 1e8) + np.random.default_rng(2).normal(size=(4, 3)) * 1e-4
        D = utils.pairwise_distances(X)
        assert (D >= 0).all() and not np.isnan(D).any()

    def test_input_not_modified(self):
        X = np.array([[1, 2], [3, 4]])
        utils.pairwise_distances(X)
        np.testing.assert_array_equal(X, [[1, 2], [3, 4]])

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            utils.pairwise_distances(self.X, metric="hamming")

    def test_column_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of columns"):
            utils.pairwise_distances(self.X, np.ones((2, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            utils.pairwise_distances([1.0, 2.0])

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN or infinity"):
            utils.pairwise_distances([[1.0, np.nan]])

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="must be numeric"):
            utils.pairwise_distances([["a", "b"]])


# ---------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------

class TestActivations:

    def test_sigmoid_values(self):
        np.testing.assert_allclose(utils.sigmoid([0, 1, -1]),
                                   [0.5, 1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])

    def test_sigmoid_extreme_inputs_do_not_overflow(self):
        with np.errstate(over="raise", invalid="raise"):
            out = utils.sigmoid([-1000.0, 1000.0])
        np.testing.assert_array_equal(out, [0.0, 1.0])

    def test_sigmoid_symmetry(self):
        x = np.linspace(-5, 5, 11)
        np.testing.assert_allclose(utils.sigmoid(x) + utils.sigmoid(-x), 1.0)

    def test_scalar_in_scalar_out(self):
        assert isinstance(utils.sigmoid(0), float)
        assert isinstance(utils.relu(-2), float)
        assert isinstance(utils.tanh(0), float)

    def test_relu(self):
        np.testing.assert_array_equal(utils.relu([-2, 0, 3]), [0, 0, 3])

    def test_leaky_relu(self):
        np.testing.assert_allclose(utils.leaky_relu([-10, 0, 5], alpha=0.1), [-1, 0, 5])

    def test_tanh_matches_numpy(self):
        x = np.linspace(-3, 3, 7)
        np.testing.assert_allclose(utils.tanh(x), np.tanh(x))

    def test_shape_preserved(self):
        assert utils.relu(np.ones((2, 3, 4))).shape == (2, 3, 4)

    def test_integer_input(self):
        assert utils.relu(np.array([-1, 2])).dtype == np.float64

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="must be numeric"):
            utils.sigmoid(["a"])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            utils.relu([])

    @pytest.mark.parametrize("alpha", [-0.1, "a", None])
    def test_bad_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            utils.leaky_relu([1.0], alpha=alpha)


# ---------------------------------------------------------------------
# logsumexp, softmax, log_softmax
# ---------------------------------------------------------------------

class TestLogsumexp:

    def test_matches_naive_on_small_values(self):
        x = np.array([1.0, 2.0, 3.0])
        assert utils.logsumexp(x) == pytest.approx(np.log(np.exp(x).sum()))

    def test_large_values_do_not_overflow(self):
        assert utils.logsumexp([1000.0, 1000.0]) == pytest.approx(1000 + np.log(2))

    def test_very_negative_values_do_not_underflow(self):
        assert utils.logsumexp([-1000.0, -1000.0]) == pytest.approx(-1000 + np.log(2))

    def test_all_neg_inf_gives_neg_inf(self):
        assert utils.logsumexp([-np.inf, -np.inf]) == -np.inf

    def test_one_neg_inf_is_ignored(self):
        assert utils.logsumexp([0.0, -np.inf]) == 0.0

    def test_axis_and_keepdims(self):
        X = np.random.default_rng(0).normal(size=(3, 4))
        np.testing.assert_allclose(utils.logsumexp(X, axis=1),
                                   np.log(np.exp(X).sum(axis=1)))
        assert utils.logsumexp(X, axis=0, keepdims=True).shape == (1, 4)
        assert utils.logsumexp(X, keepdims=True).shape == (1, 1)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="must not contain NaN"):
            utils.logsumexp([1.0, np.nan])

    def test_bad_axis_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            utils.logsumexp([1.0, 2.0], axis=1)


class TestSoftmax:

    def test_sums_to_one(self):
        X = np.random.default_rng(0).normal(size=(4, 5))
        np.testing.assert_allclose(utils.softmax(X).sum(axis=1), 1.0)

    def test_known_value(self):
        np.testing.assert_allclose(utils.softmax([0.0, 0.0]), [0.5, 0.5])

    def test_large_inputs_stable(self):
        out = utils.softmax([1000.0, 1000.0, 0.0])
        np.testing.assert_allclose(out, [0.5, 0.5, 0.0], atol=1e-12)

    def test_shift_invariance(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(utils.softmax(x), utils.softmax(x + 100))

    def test_axis_zero(self):
        X = np.random.default_rng(1).normal(size=(3, 4))
        np.testing.assert_allclose(utils.softmax(X, axis=0).sum(axis=0), 1.0)

    def test_scalar_raises(self):
        with pytest.raises(ValueError, match="at least 1-dimensional"):
            utils.softmax(3.0)

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="NaN or infinity"):
            utils.softmax([1.0, np.inf])

    def test_log_softmax_matches_log_of_softmax(self):
        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(utils.log_softmax(x), np.log(utils.softmax(x)))

    def test_log_softmax_stays_finite_for_tiny_probabilities(self):
        out = utils.log_softmax([0.0, 1000.0])
        assert out[0] == pytest.approx(-1000.0) and np.isfinite(out).all()


# ---------------------------------------------------------------------
# row_topk
# ---------------------------------------------------------------------

class TestRowTopk:

    M = np.array([[5, 1, 9, 3],
                  [2, 8, 8, 0]])

    def test_largest(self):
        np.testing.assert_array_equal(utils.row_topk(self.M, 2), [[2, 0], [1, 2]])

    def test_smallest(self):
        np.testing.assert_array_equal(utils.row_topk(self.M, 2, largest=False),
                                      [[1, 3], [3, 0]])

    def test_ties_resolved_by_column_order(self):
        np.testing.assert_array_equal(utils.row_topk(self.M, 1)[1], [1])

    def test_k_equals_width_gives_full_order(self):
        np.testing.assert_array_equal(utils.row_topk(self.M, 4, largest=False)[0],
                                      np.argsort(self.M[0], kind="stable"))

    def test_nearest_neighbours_use_case(self):
        X = np.array([[0.0], [1.0], [10.0]])
        D = utils.pairwise_distances(X)
        np.testing.assert_array_equal(utils.row_topk(D, 2, largest=False)[:, 1], [1, 0, 1])

    @pytest.mark.parametrize("k", [0, 5, 1.5, True])
    def test_bad_k_raises(self, k):
        with pytest.raises(ValueError, match="k must be an integer between 1 and 4"):
            utils.row_topk(self.M, k)

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            utils.row_topk([1, 2, 3], 1)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="must not contain NaN"):
            utils.row_topk([[1.0, np.nan]], 1)


# ---------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------

class TestBatching:

    def test_batch_indices_cover_everything_once(self):
        got = np.concatenate(list(utils.batch_indices(10, 3)))
        np.testing.assert_array_equal(got, np.arange(10))

    def test_last_batch_is_shorter(self):
        sizes = [len(b) for b in utils.batch_indices(10, 4)]
        assert sizes == [4, 4, 2]

    def test_drop_last(self):
        sizes = [len(b) for b in utils.batch_indices(10, 4, drop_last=True)]
        assert sizes == [4, 4]

    def test_batch_larger_than_n(self):
        assert [len(b) for b in utils.batch_indices(3, 10)] == [3]

    def test_shuffle_is_a_permutation_and_repeatable(self):
        a = np.concatenate(list(utils.batch_indices(20, 6, shuffle=True, seed=7)))
        b = np.concatenate(list(utils.batch_indices(20, 6, shuffle=True, seed=7)))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(a), np.arange(20))
        assert not np.array_equal(a, np.arange(20))

    def test_batches_keep_X_and_y_aligned(self):
        X = np.arange(10).reshape(5, 2)
        y = np.array([0, 1, 2, 3, 4])
        for xb, yb in utils.batches(X, y, batch_size=2, shuffle=True, seed=1):
            np.testing.assert_array_equal(xb[:, 0] // 2, yb)

    def test_single_array_gives_one_tuple_entries(self):
        out = list(utils.batches(np.arange(5), batch_size=2))
        assert len(out) == 3 and len(out[0]) == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            list(utils.batches(np.ones(3), np.ones(4), batch_size=2))

    def test_no_arrays_raises(self):
        with pytest.raises(ValueError, match="At least one array"):
            list(utils.batches(batch_size=2))

    @pytest.mark.parametrize("bs", [0, -1, 1.5, True])
    def test_bad_batch_size_raises(self, bs):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            list(utils.batch_indices(5, bs))

    def test_bad_n_raises(self):
        with pytest.raises(ValueError, match="n must be a positive integer"):
            list(utils.batch_indices(0, 2))