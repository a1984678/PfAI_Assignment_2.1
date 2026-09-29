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

def test_empty_array():
    X = np.array([]).reshape(0, 2)

    preprocessor = preprocessing.StandardScaler()
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)

    np.testing.assert_array_equal(X_transformed, np.array([]).reshape(0, 2))

# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_invalid_input_type_raises():
    with pytest.raises(TypeError):
        if not isinstance('a',int):
            raise TypeError("'a' is not an int")
        pass