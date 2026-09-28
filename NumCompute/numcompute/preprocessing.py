"""
preprocessing.py

Implement:
- StandardScaler   (z-score standardization)
- MinMaxScaler     (scale to a given range)
- OneHotEncoder    (for categorical variables)
- (Optional) Simple imputer to replace NaN with a constant

API:
- fit(X) -> self
- transform(X) -> X_out
- fit_transform(X) -> X_out

Fully vectorized using NumPy; avoid Python loops.
"""