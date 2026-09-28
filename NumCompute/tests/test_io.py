import numpy as np
import pytest

from numcompute import io


# ---------------------------------------------------------------------
# Basic / expected behaviour
# ---------------------------------------------------------------------

def test_blanks_become_fill_values(tmp_path):
    """ missing='fill' must replace blanks with the fill_value."""
    file = tmp_path / "test.csv"
    file.write_text("1,2\n"
                    "3, \n"
                     ",4\n"
                    "5,6")
    result = io.load_csv(file, delimiter=',', missing='fill', fill_value=0)

    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 0], [0, 4], [5, 6]]))

    

def test_tab_delimiter(tmp_path):
    """Test that tab delimiter works correctly."""
    file = tmp_path / "test.csv"
    file.write_text("1\t2\n"
                    "3\t4\n"
                    "5\t6")
    result = io.load_csv(file, delimiter='\t', missing='fill', fill_value=0)

    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4], [5, 6]]))
    

def test_single_row(tmp_path):
    """Test that a single row is loaded correctly."""
    file = tmp_path / "test.csv"
    file.write_text("1,2,3")
    result = io.load_csv(file, delimiter=',', missing='fill', fill_value=0)

    assert result.shape == (1,3)
    

# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_skip_ignores_fill_value(tmp_path):
    """missing='skip' must drop incomplete rows even when fill_value is not NaN."""

    file = tmp_path / "test.csv"
    file.write_text("1,2\n"
                    "3, \n"
                     ",4\n"
                    "5,6")
    result = io.load_csv(file, delimiter=',', missing='skip', fill_value=-1)

    np.testing.assert_array_equal(result, np.array([[1, 2], [5, 6]]))
    assert result.shape == (2, 2)  # Expecting 2 rows and 2 columns
    

def test_all_rows_skipped(tmp_path):
    """Test that all rows are skipped when missing='skip' and all rows have missing values."""
    file = tmp_path / "test.csv"
    file.write_text("1, \n"
                    " ,2\n"
                    " , ")
    result = io.load_csv(file, delimiter=',', missing='skip', fill_value=0)

    assert result.shape == (0, 2)  # Expecting an empty array with shape (0, 2)
    
def test_empty_file(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text("")
    
    result = io.load_csv(file, delimiter=',', missing='fill', fill_value=0)
    assert result.shape == (0, 0)  # Expecting an empty array with shape (0, 0)
    

# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_invalid_missing_input_raises(tmp_path):
    """Test that an invalid 'missing' parameter raises a ValueError."""

    file = tmp_path / "test.csv"
    file.write_text("1,2\n3,4")

    with pytest.raises(ValueError,match="missing parameter must be 'fill' or 'skip'"):
        io.load_csv(file, missing="invalid")
        