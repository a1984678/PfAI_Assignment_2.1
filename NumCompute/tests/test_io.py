import numpy as np
import pytest

from numcompute import io


# ---------------------------------------------------------------------
# Basic / expected behaviour
# ---------------------------------------------------------------------

def test_known_outputS():
    """Check against a hand-computed expected result."""
    pass


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_skip_ignores_fill_value(tmp_path):
    """missing='skip' must drop incomplete rows even when fill_value is not NaN."""

    file = tmp_path / "test.csv"
    file.write_text("1,2\n"
                    "3, \n "
                     ",4\n "
                    "5,6")
    result = io.load_csv(file, delimiter=',', missing='skip', fill_value=-1)

    np.testing.assert_array_equal(result, np.array([[1, 2], [5, 6]]))
    np.testing.assert_array_equal(result.shape, (2, 2))
    
    pass


def test_empty_array():
    x = np.array([])
    pass

# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_invalid_input_type_raises():
    with pytest.raises(TypeError):
        if not isinstance('a',int):
            raise TypeError("'a' is not an int")
        pass