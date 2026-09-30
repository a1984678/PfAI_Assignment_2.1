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


def test_empty_transform():
    X = np.array([[1, 10], [3, 30]])
    Y = np.array([]).reshape(0, 2)

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)

    assert preprocessor.transform(Y).shape == (0, 2)  # Should return an empty array with the correct shape


def test_constant_column():
    X = np.array([[1, 10], [1, 30]])
    Y = np.array([[1, 20], [1, 10]])

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    Y_transformed = preprocessor.transform(Y)

    assert np.all(Y_transformed[:, 0] == 0)  # All values in the constant column should be 0

def test_small_values_constant_column():
    X = np.array([[0.1], [0.1], [0.1]])
    Y = np.array([[0.1], [0.2]])

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    Y_transformed = preprocessor.transform(Y)

    np.testing.assert_allclose(Y_transformed[:, 0], np.array([0, 0.1]), atol=1e-8)  # The first column should be close to [0, 0.1]


def test_nanometre_scale_column_is_standardized():
    X = np.array([[1e-9], [2e-9], [3e-9]])
    Y = [[1e-9], [2e-9]]

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)

    expected = np.array([-np.sqrt(1.5), 0.0, np.sqrt(1.5)])  # ≈ [-1.2247, 0, 1.2247]
    np.testing.assert_allclose(X_transformed[:, 0], expected, rtol=1e-7, atol=1e-12)


def test_nan_in_fit():
    X = np.array([[1, 10], [3, np.nan]])
    Y = np.array([[1, 20], [1, 10]])

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    Y_transformed = preprocessor.transform(Y)

    assert not np.isnan(Y_transformed).any()  # Ensure no NaN values in the output

def test_nan_in_transform():
    X = np.array([[1, 10], [3, 30]])
    Y = np.array([[1, np.nan], [1, 10]])

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    Y_transformed = preprocessor.transform(Y)

    # Input NaN in transform should be output as NaN in the corresponding position
    assert np.isnan(Y_transformed[0, 1])  # The first row, second column should be NaN
    assert not np.isnan(Y_transformed[1, 1])  # The second row, second column should not be NaN
    

def test_nan_column():
    X = np.array([[np.nan, 10], [np.nan, 30]])

    preprocessor = preprocessing.StandardScaler()

    with pytest.raises(ValueError, match="Cannot fit StandardScaler on a column with all NaN values."):
        preprocessor.fit(X)
    

# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_empty_fit():
    X = np.array([]).reshape(0, 2)

    preprocessor = preprocessing.StandardScaler()
    with pytest.raises(ValueError, match="Cannot fit StandardScaler on an empty array."):
        preprocessor.fit(X)

def test_1d_array():
    X = np.array([1, 2, 3])  # 1D array

    preprocessor = preprocessing.StandardScaler()
    with pytest.raises(ValueError, match="Input data must be at least 2 dimensions. Got shape:"):
        preprocessor.fit(X)

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