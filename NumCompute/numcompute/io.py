"""
io.py

- Load CSV files using numpy.genfromtxt
- Support custom delimiters (e.g. comma, tab)
- Handle missing values (skip or fill with a placeholder value)
- Support numeric (dtype=float) and text/mixed (dtype=object) data
- Return data as NumPy arrays
"""

import numpy as np
from os import path, PathLike


def _classify_cells(file_path: str | PathLike[str], delimiter: str | None,
                    skip_header: int):
    """Load the file twice with genfromtxt and classify every cell.

    Returns (as_float, as_str, is_number, is_text, is_missing). Each cell is
    exactly one of number / text / missing. Blank, whitespace-only and literal
    'nan' cells are missing; anything else that doesn't parse as a float is text.
    """
    as_float = np.genfromtxt(file_path, dtype=float, delimiter=delimiter,
                             skip_header=skip_header, ndmin=2)
    as_str = np.char.strip(np.genfromtxt(file_path, dtype=str, delimiter=delimiter,
                                         skip_header=skip_header, ndmin=2))

    is_number = ~np.isnan(as_float)
    is_missing = (as_str == "") | (np.char.lower(as_str) == "nan")
    is_text = ~is_number & ~is_missing
    return as_float, as_str, is_number, is_text, is_missing


def _describe_columns(cols, mask, as_str):
    """"column 1 (row 2: 'NA')" for each column, using its first cell where mask is True."""
    rows = mask[:, cols].argmax(axis=0)
    return ", ".join(f"column {c} (row {r}: {str(as_str[r, c])!r})" for c, r in zip(cols, rows))


def _fill_kind(fill_value) -> str:
    """'number', 'text' or 'missing' (a NaN fill); anything else can't go in a CSV cell."""
    if isinstance(fill_value, str):
        return "text"
    if (isinstance(fill_value, (int, float, np.integer, np.floating)) and
        not isinstance(fill_value, bool)):
        return "missing" if np.isnan(fill_value) else "number"
    raise TypeError(
        f"fill_value must be a number or a string, got {type(fill_value).__name__}"
    )


def load_csv(file_path, delimiter=',', missing="fill", fill_value=np.nan,
             skip_header=0, dtype=float):
    """
    Load a CSV file into a NumPy array.

    Parameters:
    ----------
    - file_path: path to the CSV file
    - delimiter: str, delimiter used in the CSV file (default is ',')
    - missing: str, how to handle missing values ('fill' or 'skip')
    - fill_value: number or str to fill in for missing values (default is np.nan);
    ignored (and not validated) when missing='skip'
    - skip_header: int, number of lines to skip at the beginning of the file (default is 0)
    - dtype: float (numeric data only) or object (columns may be numeric or text)

    Returns:
    -------
    - data: np.ndarray, loaded data as a NumPy array (n_rows, n_columns)
    - with dtype=float, a float64 array
    - with dtype=object, numbers are Python floats, text cells are Python strs,
    and filled cells hold fill_value as given
    - on empty files, returns an empty array with shape (0, 0)
    - on skipping rows, returns an array with shape (n_rows, n_columns) where n_rows
    is the number of complete rows
    - on filling missing values, returns an array with shape (n_rows, n_columns) where n_rows
    is the total number of rows in the file

    Raises:
    ------
    - ValueError: if 'missing' is not 'fill' or 'skip'
    - TypeError: if dtype is not float or object; if fill_value is not a number or
    string; if a column mixes numbers and text; if dtype=float and a column
    contains text or fill_value is a string; or if fill_value would put a number
    into a text column (or vice versa)
    - FileNotFoundError: if the specified file does not exist

    Complexity:
    ----------
    - Time complexity: O(n*m) where n is the number of rows and m is the number of columns
    in the CSV file
    - Space complexity: O(n*m) for storing the loaded data in memory

    Note:
    ----
    - Blank, whitespace-only and 'nan' (any case) cells are treated as missing.
    - NaN values are used to represent missing values when fill_value is not specified.
      A NaN fill is allowed in any column, including text columns.
    - Row and column numbers in error messages are 0-indexed, and rows are counted
      after the skip_header lines, so 'row 0' is the first data row, not line 1 of the file.
    """

    if missing not in ("fill", "skip"):
        raise ValueError("missing parameter must be 'fill' or 'skip'")

    if dtype not in (float, object):
        raise TypeError("dtype must be float or object")

    kind = _fill_kind(fill_value) if missing == "fill" else None

    if dtype is float and kind == "text":
        raise TypeError("Cannot fill missing values with text when dtype is float")

    # Handle empty files
    if path.getsize(file_path) == 0:
        return np.empty((0, 0), dtype=dtype)

    as_float, as_str, is_number, is_text, is_missing = _classify_cells(
        file_path, delimiter, skip_header
    )
    has_text = is_text.any(axis=0)
    has_number = is_number.any(axis=0)

    mixed = np.flatnonzero(has_text & has_number)
    if mixed.size:
        raise TypeError(f"Mixed types in columns: {_describe_columns(mixed, is_text, as_str)}")

    if dtype is float:
        text_cols = np.flatnonzero(has_text)
        if text_cols.size:
            raise TypeError(f"Text in columns: {_describe_columns(text_cols, is_text, as_str)}")
        data = as_float
    else:
        data = np.where(is_text, as_str.astype(object), as_float.astype(object))

    if kind is None:
        return data[~is_missing.any(axis=1)]  # Drop rows with any missing values

    # Refuse fills that would create a column mixing numbers and text.
    # A NaN fill is exempt: it reads back as missing, not as a number.
    if kind != "missing":
        other_kind = has_text if kind == "number" else has_number
        other_mask = is_text if kind == "number" else is_number
        clash = np.flatnonzero(is_missing.any(axis=0) & other_kind)
        if clash.size:
            raise TypeError(
                f"Cannot fill missing values in columns: "
                f"{_describe_columns(clash, other_mask, as_str)} with {kind} {fill_value!r}"
            )

    return np.where(is_missing, fill_value, data)
