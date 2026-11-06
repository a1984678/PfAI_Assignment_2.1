"""Tests for the private shared validation helpers in ``numcompute_stream._validation``."""

import numpy as np
import pytest

from numcompute_stream._validation import (
    check_array,
    check_n_features,
    check_positive_int,
    check_X_y,
)


def test_check_array_coerces_ints_to_float64_2d():
    out = check_array([[1, 2], [3, 4]])
    assert out.dtype == np.float64
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, [[1.0, 2.0], [3.0, 4.0]])


def test_check_array_does_not_copy_float64_input():
    X = np.ones((3, 2))
    assert check_array(X) is X


@pytest.mark.parametrize("bad, match", [
    ([1.0, 2.0], "2-dimensional"),
    (np.ones((2, 2, 2)), "2-dimensional"),
    (np.empty((0, 3)), "at least one row"),
    (np.empty((3, 0)), "at least one row"),
    ([["a", "b"]], "numeric"),
    ([[1.0, np.inf]], "infinity"),
])
def test_check_array_rejects_bad_input(bad, match):
    with pytest.raises(ValueError, match=match):
        check_array(bad)


def test_check_array_nan_only_when_allowed():
    X = [[1.0, np.nan]]
    with pytest.raises(ValueError, match="NaN"):
        check_array(X)
    out = check_array(X, allow_nan=True)
    assert np.isnan(out[0, 1])
    with pytest.raises(ValueError, match="infinity"):
        check_array([[np.nan, -np.inf]], allow_nan=True)


def test_check_array_uses_name_in_message():
    with pytest.raises(ValueError, match="X_chunk must be 2-dimensional"):
        check_array([1.0], name="X_chunk")


def test_check_X_y_keeps_string_labels():
    X, y = check_X_y([[1], [2]], ["cat", "dog"])
    assert X.shape == (2, 1)
    assert y.dtype.kind == "U"


@pytest.mark.parametrize("y, match", [
    ([0, 1, 0], "same number of samples"),
    ([[0], [1]], "1-dimensional"),
])
def test_check_X_y_rejects_mismatched_y(y, match):
    with pytest.raises(ValueError, match=match):
        check_X_y(np.ones((2, 3)), y)


def test_check_n_features_first_chunk_sets_then_must_match():
    n = check_n_features(np.ones((4, 3)), None)
    assert n == 3
    assert check_n_features(np.ones((1, 3)), n) == 3
    with pytest.raises(ValueError, match="X has 2 features, but 3"):
        check_n_features(np.ones((4, 2)), n)


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "5", None])
def test_check_positive_int_rejects(bad):
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        check_positive_int(bad, "chunk_size")


def test_check_positive_int_accepts_numpy_int():
    out = check_positive_int(np.int64(5), "chunk_size")
    assert out == 5 and type(out) is int
