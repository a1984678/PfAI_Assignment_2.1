import numpy as np
import pytest

from numcompute import preprocessing


# ---------------------------------------------------------------------
# Basic / expected behaviour
# ---------------------------------------------------------------------

def test_basic_standard_scaler():

    X = np.array([[1, 10], [3, 30]])
    Y = np.array([[2, 20], [1, 10]])

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    Y_transformed = preprocessor.transform(Y)

    np.testing.assert_array_almost_equal(Y_transformed, np.array([[0, 0], [-1, -1]]), decimal=6)

    
# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_empty_fit():
    X = np.array([]).reshape(0, 2)

    preprocessor = preprocessing.StandardScaler()
    with pytest.raises(ValueError, match="Cannot fit StandardScaler on an empty array."):
        preprocessor.fit(X)

def test_empty_transform():
    X = np.array([[1, 10], [3, 30]])
    Y = np.array([]).reshape(0, 2)

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    with pytest.raises(ValueError, match="Cannot transform an empty array."):
        preprocessor.transform(Y)


def test_constant_column():
    X = np.array([[1, 10], [1, 30]])
    Y = np.array([[1, 20], [1, 10]])

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    Y_transformed = preprocessor.transform(Y)

    assert np.all(Y_transformed[:, 0] == 0)  # All values in the constant column should be 0


def test_transform_without_fit():
    Y = np.array([[1, 20], [1, 10]])

    preprocessor = preprocessing.StandardScaler()
    with pytest.raises(RuntimeError, match="fit must be called before transform."):
        preprocessor.transform(Y)


def test_column_count_mismatch():
    X = np.array([[1, 10], [3, 30]])
    Y = np.array([[1, 20, 5], [1, 10, 6]])  # Mismatched column count

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    with pytest.raises(ValueError, match="Incompatible dimensionality"):
        preprocessor.transform(Y)

def test_nan_in_input():
    X = np.array([[1, 10], [3, np.nan]])
    Y = np.array([[1, 20], [1, 10]])

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    Y_transformed = preprocessor.transform(Y)

    assert not np.isnan(Y_transformed).any()  # Ensure no NaN values in the output

# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_invalid_input_type_raises():
    with pytest.raises(TypeError):
        if not isinstance('a',int):
            raise TypeError("'a' is not an int")
        pass