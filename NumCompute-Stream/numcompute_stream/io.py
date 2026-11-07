"""Custom I/O pipeline.

- Load a dataset from a CSV file (used by ``demo/stream_demo.ipynb``)
- Split the dataset into chunks to simulate a streaming data setting

Streaming API (NumCompute-Stream)
---------------------------------
- ``iter_chunks(X, y=None, *, chunk_size, shuffle=False, seed=0)``: lazy generator of
  (X_chunk, y_chunk) pairs (or X_chunk alone); the last chunk may be shorter

Example
-------
>>> data = load_csv("data.csv", skip_header=1)
>>> X, y = data[:, :-1], data[:, -1]
>>> for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=100):
...     model.partial_fit(X_chunk, y_chunk)

Baseline (adapted from NumCompute, Assignment 2.1)
--------------------------------------------------
- Load CSV files using numpy.genfromtxt (``load_csv``)
- Support custom delimiters (e.g. comma, tab)
- Handle missing values (skip or fill with a placeholder value)
- Support numeric (dtype=float) and text/mixed (dtype=object) data
- Return data as NumPy arrays
"""

import warnings
from os import path, PathLike

import numpy as np

from ._validation import check_positive_int
from .utils import batches


def _classify_cells(file_path: str | PathLike[str], delimiter: str | None,
                    skip_header: int):
    """Load the file twice with genfromtxt and classify every cell.

    Returns (as_float, as_str, is_number, is_text, is_missing). Each cell is
    exactly one of number / text / missing. Blank, whitespace-only and literal
    'nan' cells are missing; anything else that doesn't parse as a float is text.
    """
    # genfromtxt warns "Empty input file" when nothing is left after the header; load_csv
    # handles that case itself by returning an empty array, so the warning is not needed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        as_float = np.genfromtxt(file_path, dtype=float, delimiter=delimiter,
                                 skip_header=skip_header, ndmin=2)
        as_str = np.char.strip(np.genfromtxt(file_path, dtype=str, delimiter=delimiter,
                                             skip_header=skip_header, ndmin=2))

    is_number = ~np.isnan(as_float)
    is_missing = (as_str == "") | (np.char.lower(as_str) == "nan")
    is_text = ~is_number & ~is_missing
    return as_float, as_str, is_number, is_text, is_missing


def _describe_columns(cols, mask, as_str):
    """Describe each column by its first cell where mask is True, e.g. "column 1 (row 2: 'NA')"."""
    rows = mask[:, cols].argmax(axis=0)
    return ", ".join(f"column {c} (row {r}: {str(as_str[r, c])!r})" for c, r in zip(cols, rows))


def _fill_kind(fill_value) -> str:
    """'number', 'text' or 'missing' (a NaN fill); anything else can't go in a CSV cell."""
    if isinstance(fill_value, str):
        return "text"
    is_number = (isinstance(fill_value, (int, float, np.integer, np.floating))
                 and not isinstance(fill_value, bool))
    if is_number:
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
    - on empty files, blank-only files and files with nothing after skip_header, returns an
    empty array with shape (0, 0)
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

    # Blank-only files, or files with nothing after the skipped header lines
    if as_float.size == 0:
        return np.empty((0, 0), dtype=dtype)

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


def iter_chunks(X, y=None, *, chunk_size, shuffle=False, seed=0):
    """
    Split a dataset into chunks to simulate a data stream.

    Thin wrapper over ``utils.batches``: validates the inputs with streaming-specific
    error messages, then yields consecutive row slices.

    Parameters:
    ----------
    - X: array-like of shape (n_samples, ...), any dtype; NaN (missing) values are passed
    through unchanged
    - y: array-like of shape (n_samples,) or None (default None)
    - chunk_size: int >= 1, rows per chunk; the last chunk may be shorter
    - shuffle: bool, shuffle rows once before chunking (default False)
    - seed: int, random seed used when shuffle=True (default 0)

    Yields:
    ------
    - (X_chunk, y_chunk) if y is given, otherwise X_chunk. X_chunk has shape
    (chunk_size, ...) except possibly the last, which holds the remaining rows.

    Raises:
    ------
    - ValueError: if chunk_size is not a positive integer
    - ValueError: if X has no rows
    - ValueError: if y is given and its length differs from the number of rows in X

    Note:
    ----
    - This is a generator, so the errors above are raised on the first ``next()``,
    not when iter_chunks is called.
    - Chunks are copies (``batches`` indexes with an index array), so changing a chunk does
    not change X.

    Complexity:
    ----------
    - Time complexity: O(n_samples * d) in total over all chunks, d = values per row
    - Space complexity: O(chunk_size * d) per chunk, plus O(n_samples) for the
    shuffled index array when shuffle=True
    """
    chunk_size = check_positive_int(chunk_size, "chunk_size")
    X = np.asarray(X)
    if X.ndim == 0 or X.shape[0] == 0:
        raise ValueError(f"X has no rows to stream. Got shape: {X.shape}")

    if y is None:
        for (X_chunk,) in batches(X, batch_size=chunk_size, shuffle=shuffle, seed=seed):
            yield X_chunk
        return

    y = np.asarray(y)
    if y.ndim == 0 or y.shape[0] != X.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. "
                         f"Got {X.shape[0]} and {y.shape[0] if y.ndim else 'a scalar'}.")
    yield from batches(X, y, batch_size=chunk_size, shuffle=shuffle, seed=seed)
