"""
pipeline.py

Minimal Transformer / Estimator protocol:
- Preprocessors: fit, transform
- Models: fit, predict (no ML model implementation required)

Pipeline chaining, e.g.:

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('encode', OneHotEncoder())
    ])
    X_tr = pipe.fit_transform(X)
"""

class Transformer:
    """Base class for all transformers."""

    def fit(self, X):
        """Compute the parameters needed for transformation."""
        raise NotImplementedError

    def transform(self, X):
        """Apply the transformation to the data."""
        raise NotImplementedError

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        self.fit(X)
        return self.transform(X)