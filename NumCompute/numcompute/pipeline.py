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
        if not self.steps:
            return self

        transformed = X
        for name, transformer in self.steps[:-1]:
            if not hasattr(transformer, "fit_transform"):
                raise TypeError(f"Pipeline step {name!r} must be a transformer")
            transformed = transformer.fit_transform(transformed, y)

        name, estimator = self.steps[-1]
        if not hasattr(estimator, "fit"):
            raise TypeError(f"Final pipeline step {name!r} must implement fit")
        estimator.fit(transformed, y)
        return self

    def transform(self, X):
        """Apply all transformers to the data."""
        transformed = X
        for name, transformer in self.steps:
            if not hasattr(transformer, "transform"):
                raise TypeError(f"Pipeline step {name!r} must implement transform")
            transformed = transformer.transform(transformed)
        return transformed

    def fit_transform(self, X, y=None):
        """Fit all transformers and transform the data."""
        transformed = X
        for name, transformer in self.steps:
            if not hasattr(transformer, "fit_transform"):
                raise TypeError(f"Pipeline step {name!r} must implement fit_transform")
            transformed = transformer.fit_transform(transformed, y)
        return transformed

    def predict(self, X):
        """Make predictions using the fitted pipeline."""
        if not self.steps:
            raise ValueError("Cannot predict with an empty pipeline")

        transformed = X
        for name, transformer in self.steps[:-1]:
            if not hasattr(transformer, "transform"):
                raise TypeError(f"Pipeline step {name!r} must implement transform")
            transformed = transformer.transform(transformed)

        name, estimator = self.steps[-1]
        if not hasattr(estimator, "predict"):
            raise TypeError(f"Final pipeline step {name!r} must implement predict")
        return estimator.predict(transformed)


