"""
Regression tests for the bugs found in the code review.

Each test failed before its fix and passes after it, so a later change that brings a bug back
is caught.
"""

import numpy as np
import pytest

from numcompute import benchmarking, io, preprocessing, rank, stats, utils
from numcompute.pipeline import Estimator, Pipeline


# ---------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------

class TestUtils:

    def test_close_points_far_from_origin_keep_their_distance(self):
        """A non-negative, finite answer can still be wrong: 1e9 and 1e9 + 1 are 1 apart."""
        D = utils.pairwise_distances([[1e9], [1e9 + 1]])
        assert D[0, 1] == 1.0

    def test_close_points_match_brute_force_with_large_offset(self):
        X = 1e8 + np.random.default_rng(3).normal(size=(5, 3))
        D = utils.pairwise_distances(X)
        for i in range(5):
            for j in range(5):
                assert D[i, j] == pytest.approx(np.linalg.norm(X[i] - X[j]), rel=1e-9)

    def test_huge_values_do_not_overflow(self):
        assert utils.pairwise_distances([[1e200, 0.0], [0.0, 0.0]])[0, 1] == pytest.approx(1e200)

    def test_cosine_zero_vector_convention(self):
        """Y omitted: the diagonal is 0. Y given: a zero vector is at distance 1 from everything."""
        Z = [[0.0, 0.0], [1.0, 1.0]]
        assert utils.pairwise_distances(Z, metric="cosine")[0, 0] == 0.0
        assert utils.pairwise_distances(Z, [[0.0, 0.0]], metric="cosine")[0, 0] == 1.0

    def test_softmax_huge_equal_inputs(self):
        np.testing.assert_allclose(utils.softmax([1e16, 1e16]), [0.5, 0.5])

    def test_log_softmax_huge_equal_inputs(self):
        np.testing.assert_allclose(utils.log_softmax([1e16, 1e16]), [-np.log(2)] * 2)

    def test_softmax_axis_none_sums_to_one(self):
        assert utils.softmax(np.arange(6.0).reshape(2, 3), axis=None).sum() == pytest.approx(1.0)

    def test_row_topk_keeps_integer_precision(self):
        M = np.array([[2 ** 53, 2 ** 53 + 1]])
        np.testing.assert_array_equal(utils.row_topk(M, 1), [[1]])
        np.testing.assert_array_equal(utils.row_topk(M, 1, largest=False), [[0]])

    def test_row_topk_unsigned_integers(self):
        M = np.array([[1, 3, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(utils.row_topk(M, 2), [[1, 2]])

    @pytest.mark.parametrize("alpha", [np.nan, np.inf, True])
    def test_leaky_relu_rejects_nan_inf_bool_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            utils.leaky_relu([1.0], alpha=alpha)

    def test_batch_indices_does_not_allocate_n_values_without_shuffle(self):
        first = next(iter(utils.batch_indices(10 ** 9, 4)))  # an arange(10**9) would need 8 GB
        np.testing.assert_array_equal(first, [0, 1, 2, 3])


# ---------------------------------------------------------------------
# stats and rank
# ---------------------------------------------------------------------

class TestStatsRank:

    def test_median_of_huge_opposite_values_does_not_overflow(self):
        assert stats.median([-1e308, 1e308]) == 0.0

    def test_quantile_of_huge_opposite_values_does_not_overflow(self):
        np.testing.assert_allclose(stats.quantile([-1e308, 1e308], [0.25, 0.75]),
                                   [-5e307, 5e307])

    @pytest.mark.parametrize("method", ["linear", "lower", "higher", "midpoint"])
    def test_percentile_of_huge_opposite_values_does_not_overflow(self, method):
        result = rank.percentile([-1e308, 1e308], 50, method)
        assert np.isfinite(result)
        if method in ("linear", "midpoint"):
            assert result == 0.0

    def test_percentile_midpoint_of_huge_equal_sign_values_does_not_overflow(self):
        assert rank.percentile([1e308, 1.7e308], 50, "midpoint") == pytest.approx(1.35e308)


# ---------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------

class TestPreprocessing:

    @pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
    def test_standard_scaler_any_magnitude(self, scale):
        X = np.array([[1.0], [2.0], [3.0]]) * scale
        out = preprocessing.StandardScaler().fit_transform(X)
        np.testing.assert_allclose(out[:, 0], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)], atol=1e-12)

    def test_standard_scaler_huge_constant_column_gives_zeros(self):
        """The mean of [1e308, 1e308] summed to inf, so the output was -inf."""
        out = preprocessing.StandardScaler().fit_transform([[1e308], [1e308]])
        np.testing.assert_array_equal(out, [[0.0], [0.0]])

    def test_standard_scaler_huge_opposite_values(self):
        out = preprocessing.StandardScaler().fit_transform([[-1e308], [1e308]])
        np.testing.assert_allclose(out, [[-1.0], [1.0]])

    def test_minmax_range_wider_than_float_max(self):
        """max - min of -1e308 and 1e308 overflowed, so the output was [0, nan]."""
        out = preprocessing.MinMaxScaler().fit_transform([[-1e308], [1e308]])
        np.testing.assert_allclose(out, [[0.0], [1.0]])

    @pytest.mark.parametrize("feature_range", [(0, np.nan), (0, np.inf), (-np.inf, 1)])
    def test_minmax_non_finite_range_raises(self, feature_range):
        with pytest.raises(ValueError, match="finite"):
            preprocessing.MinMaxScaler(feature_range=feature_range)

    @pytest.mark.parametrize("feature_range", [(1,), "ab", 5, (0, 1, 2)])
    def test_minmax_range_must_be_a_pair(self, feature_range):
        with pytest.raises(ValueError, match="pair of numbers"):
            preprocessing.MinMaxScaler(feature_range=feature_range)

    def test_imputer_rejects_bool_fill_value(self):
        with pytest.raises(ValueError, match="fill_value must be a numeric type"):
            preprocessing.SimpleImputer(fill_value=True)

    def test_one_hot_string_categories(self):
        X = np.array([["red"], ["blue"], ["red"]])
        enc = preprocessing.OneHotEncoder().fit(X)
        np.testing.assert_array_equal(enc.transform(np.array([["blue"], ["red"]])),
                                      [[1, 0], [0, 1]])

    def test_one_hot_text_and_number_columns(self):
        X = np.array([["red", 1.0], ["blue", 2.0], ["red", 1.0]], dtype=object)
        out = preprocessing.OneHotEncoder().fit_transform(X)
        assert out.shape == (3, 4) and out.dtype == np.float64

    def test_one_hot_nan_in_text_column_is_its_own_category(self):
        X = np.array([["red"], [np.nan], ["blue"]], dtype=object)
        enc = preprocessing.OneHotEncoder().fit(X)
        np.testing.assert_array_equal(enc.transform(X), [[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    @pytest.mark.parametrize("X", [[[1], ["a"]], [["a"], [1.5]],
                                   np.array([[1], ["a"]], dtype=object)])
    def test_one_hot_column_mixing_numbers_and_text_raises(self, X):
        with pytest.raises(TypeError, match="Mixed types"):
            preprocessing.OneHotEncoder().fit(X)

    def test_one_hot_text_that_looks_like_a_number_stays_text(self):
        """fit sees "red" so "1" is text; transform of "1" alone must still be text, not 1.0."""
        enc = preprocessing.OneHotEncoder().fit([["red"], ["1"]])
        np.testing.assert_array_equal(enc.transform([["1"]]), [[1, 0]])

    def test_one_hot_keeps_distinct_large_integers_apart(self):
        """float64 cannot tell 2**53 from 2**53 + 1; the encoder must."""
        out = preprocessing.OneHotEncoder().fit_transform([[2 ** 53], [2 ** 53 + 1]])
        np.testing.assert_array_equal(out, [[1, 0], [0, 1]])

    def test_one_hot_large_integers_stay_apart_when_floats_are_present(self):
        """A float in the list made NumPy convert everything to float64 and merge the integers."""
        enc = preprocessing.OneHotEncoder().fit([[2 ** 53], [2 ** 53 + 1], [0.5]])
        assert len(enc.categories_[0]) == 3
        np.testing.assert_array_equal(enc.transform([[2 ** 53], [2 ** 53 + 1], [0.5]]),
                                      [[0, 1, 0], [0, 0, 1], [1, 0, 0]])


# ---------------------------------------------------------------------
# pipeline: plain objects with only fit / transform / predict
# ---------------------------------------------------------------------

class PlainScaler:
    """No base class and no fit_transform: only fit and transform."""

    def fit(self, X, y=None):
        self.shift_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.shift_


class PlainModel:

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.sum(X, axis=1)


class TestPipeline:

    def test_accepts_objects_with_only_fit_and_transform(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        pipe = Pipeline([("plain", PlainScaler()), ("scale", preprocessing.StandardScaler())])
        np.testing.assert_almost_equal(pipe.fit_transform(X), [[-1, -1], [1, 1]])
        np.testing.assert_almost_equal(pipe.transform(X), [[-1, -1], [1, 1]])

    def test_fit_then_predict_with_plain_objects(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        pipe = Pipeline([("plain", PlainScaler()), ("model", PlainModel())])
        pipe.fit(X)
        np.testing.assert_array_equal(pipe.predict(X), [-2.0, 2.0])

    def test_rejects_step_with_only_fit(self):
        class OnlyFit:
            def fit(self, X, y=None):
                return self

        with pytest.raises(TypeError, match="must be a transformer"):
            Pipeline([("bad", OnlyFit()), ("model", PlainModel())])

    def test_estimator_fit_does_not_require_y(self):
        with pytest.raises(NotImplementedError):
            Estimator().fit(np.ones((2, 2)))


# ---------------------------------------------------------------------
# benchmarking
# ---------------------------------------------------------------------

class TestBenchmarking:

    @pytest.mark.parametrize("repeats", ["5", True])
    def test_time_it_rejects_non_integer_repeats(self, repeats):
        with pytest.raises(ValueError, match="repeats"):
            benchmarking.time_it(sum, [1], repeats=repeats)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_report_rejects_non_finite_time(self, bad):
        with pytest.raises(ValueError, match="positive and finite"):
            benchmarking.report({"a": bad, "b": 1.0})


# ---------------------------------------------------------------------
# io
# ---------------------------------------------------------------------

class TestIO:

    @pytest.mark.parametrize("text", ["", "\n", "   \n"])
    def test_empty_like_files_give_shape_0_0(self, tmp_path, text):
        file = tmp_path / "test.csv"
        file.write_text(text)
        assert io.load_csv(file).shape == (0, 0)

    def test_header_only_file_gives_shape_0_0(self, tmp_path, recwarn):
        file = tmp_path / "test.csv"
        file.write_text("a,b\n")
        assert io.load_csv(file, skip_header=1).shape == (0, 0)
        assert len(recwarn) == 0  # no "Empty input file" warning from genfromtxt

    def test_skip_header_skips_a_real_header_row(self, tmp_path):
        file = tmp_path / "test.csv"
        file.write_text("a,b\n1,2\n3,4")
        np.testing.assert_array_equal(io.load_csv(file, skip_header=1), [[1, 2], [3, 4]])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            io.load_csv(tmp_path / "does_not_exist.csv")
