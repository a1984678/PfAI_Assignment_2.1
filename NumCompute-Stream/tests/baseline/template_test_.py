import numpy as np
import pytest


# ---------------------------------------------------------------------
# Basic / expected behaviour
# ---------------------------------------------------------------------

def test_basic_case():
    """Sanity check on a simple, well-formed input."""
    pass


def test_known_output():
    """Check against a hand-computed expected result."""
    pass


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_empty_array():
    x = np.array([])
    assert x.size == 0


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_invalid_input_type_raises():
    with pytest.raises(TypeError):
        if not isinstance('a', int):
            raise TypeError("'a' is not an int")
