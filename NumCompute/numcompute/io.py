"""
io.py

- Load CSV files using numpy.genfromtxt
- Support custom delimiters (e.g. comma, tab)
- Handle missing values (skip or fill with a placeholder value)
- Return data as NumPy arrays
"""

import numpy as np
from os import path


def load_csv(file_path, delimiter=',', missing="fill", fill_value=np.nan, skip_header=0):
    """
    Load a CSV file into a NumPy array.

    Parameters:
    ----------
    - file_path: path to the CSV file
    - delimiter: str, delimiter used in the CSV file (default is ',')
    - missing: str, how to handle missing values ('fill' or 'skip')
    - fill_value: value to fill in for missing values (default is np.nan)
    - skip_header: int, number of lines to skip at the beginning of the file (default is 0)

    Returns:
    -------
    - data: np.ndarray, loaded data as a NumPy array (n_rows, n_columns)
    - on empty files, returns an empty array with shape (0, 0)
    - on skipping rows, returns an array with shape (n_rows, n_columns) where n_rows
    is the number of complete rows
    - on filling missing values, returns an array with shape (n_rows, n_columns) where n_rows
    is the total number of rows in the file

    Raises:
    ------
    - ValueError: if the 'missing' parameter is not 'fill' or 'skip'
    - FileNotFoundError: if the specified file does not exist

    Complexity:
    ----------
    - Time complexity: O(n*m) where n is the number of rows and m is the number of columns
    in the CSV file
    - Space complexity: O(n*m) for storing the loaded data in memory

    Note:
    ----
    - NaN values are used to represent missing values when fill_value is not specified.
    """

    if missing not in ["fill", "skip"]:
        raise ValueError("missing parameter must be 'fill' or 'skip'")

    # Handle empty files
    if path.getsize(file_path) == 0:
        return np.empty((0, 0))

    data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=skip_header, ndmin=2)

    if missing == "skip":
        data = data[~np.isnan(data).any(axis=1)]  # Remove rows with NaN values
    else:
        data = np.where(np.isnan(data), fill_value, data)  # Fill NaN values with fill_value

    return data
