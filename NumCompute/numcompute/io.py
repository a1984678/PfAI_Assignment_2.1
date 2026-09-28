"""
io.py

- Load CSV files using numpy.loadtxt or numpy.genfromtxt
- Support custom delimiters (e.g. comma, tab)
- Handle missing values (skip or fill with a placeholder value)
- Return data as NumPy arrays
"""

import numpy as np

def load_csv(file_path, delimiter=',', missing="fill", fill_value=np.nan , skip_header=0):
    """
    Load a CSV file into a NumPy array.

    Parameters:
    - file_path: str, path to the CSV file
    - delimiter: str, delimiter used in the CSV file (default is ',')
    - missing: str, how to handle missing values ('fill' or 'skip')
    - fill_value: value to fill in for missing values (default is np.nan)
    - skip_header: int, number of lines to skip at the beginning of the file (default is 0)

    Returns:
    - data: np.ndarray, loaded data as a NumPy array
    """
    if missing not in ["fill", "skip"]:
        raise ValueError("missing parameter must be 'fill' or 'skip'")

    data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=skip_header, ndmin=2)
    
    if missing == "skip":
        data = data[~np.isnan(data).any(axis=1)]  # Remove rows with NaN values
    else:
        data = np.where(np.isnan(data), fill_value, data)  # Fill NaN values with fill_value
    
    return data