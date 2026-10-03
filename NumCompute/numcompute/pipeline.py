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

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Compute the parameters needed for transformation."""
        raise NotImplementedError

    def transform(self, X):
        """Apply the transformation to the data."""
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)


class Estimator:
    """Base class for all estimators."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit the model to the data."""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions using the fitted model."""
        raise NotImplementedError


class Pipeline:
    """A simple pipeline for chaining transformers and estimators."""

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        """Fit all transformers and the final estimator."""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions using the fitted pipeline."""
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        """Fit all transformers and transform the data."""
        raise NotImplementedError