"""Streaming preprocessing transformers.

- Scalers (e.g., ``StandardScaler``) must support ``.partial_fit()``:

  - Maintain running mean/var (e.g., Welford or EMA)

- ``Imputer`` updates missing-value estimates on the fly
- ``OneHotEncoder`` expands categories incrementally
"""
