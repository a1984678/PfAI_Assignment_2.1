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
    """
    Base class for all transformers.

    Subclasses must implement fit and transform; fit_transform is provided.

    API:
    ----
    - fit(X, y=None) -> self
    - transform(X) -> X_out
    - fit_transform(X, y=None) -> X_out
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Compute the parameters needed for transformation.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: ignored by most transformers, accepted for Pipeline compatibility

        Returns:
        -------
        - self: object

        Raises:
        ------
        - NotImplementedError: if the subclass does not override fit
        """
        raise NotImplementedError

    def transform(self, X):
        """
        Apply the transformation to the data.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: transformed array

        Raises:
        ------
        - NotImplementedError: if the subclass does not override transform
        """
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: passed through to fit

        Returns:
        -------
        - X_out: transformed array

        Note:
        ----
        - Equivalent to calling fit(X, y) followed by transform(X).
        """
        self.fit(X, y)
        return self.transform(X)


class Estimator:
    """
    Base class for all estimators.

    Subclasses must implement fit and predict.

    API:
    ----
    - fit(X, y=None) -> self
    - predict(X) -> y_pred
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,), target values

        Returns:
        -------
        - self: object

        Raises:
        ------
        - NotImplementedError: if the subclass does not override fit
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Make predictions using the fitted model.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - y_pred: array of shape (n_samples,)

        Raises:
        ------
        - NotImplementedError: if the subclass does not override predict
        """
        raise NotImplementedError


class Pipeline:
    """
    Chain transformers, optionally ending in an estimator.

    Each step's output is the next step's input. All steps except the last
    must be transformers; the last step may be a transformer or an estimator.

    Parameters:
    ----------
    - steps: list of (name, object) tuples
        Intermediate objects must implement fit_transform; the final object must
        implement fit.

    Attributes:
    ----------
    - steps: the list of (name, object) tuples passed to the constructor.

    Complexity:
    ----------
    - Time complexity: the sum of the complexities of each step.
    - Space complexity: one intermediate array per step is held while it is processed.

    Note:
    ----
    - Steps are validated when the Pipeline is created, not when it is fitted.
    - Steps are not copied; fitting the Pipeline fits the objects you passed in.
    - y is passed to every step's fit, so transformers must accept a y argument.
    - An empty Pipeline is allowed: fit does nothing and transform returns X unchanged.
    """

    def __init__(self, steps):
        """
        Store and validate the pipeline steps.

        Parameters:
        ----------
        - steps: list of (name, object) tuples

        Raises:
        ------
        - TypeError: if any step before the last does not implement fit_transform
        - TypeError: if the final step does not implement fit
        """
        self.steps = steps
        if not self.steps:
            return

        for name, transformer in self.steps[:-1]:
            if not hasattr(transformer, "fit_transform"):
                raise TypeError(f"Pipeline step {name!r} must be a transformer")

        name, estimator = self.steps[-1]
        if not hasattr(estimator, "fit"):
            raise TypeError(f"Final pipeline step {name!r} must implement fit")

    def fit(self, X, y=None):
        """
        Fit all transformers in order, then fit the final step.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,), passed to every step's fit (default None)

        Returns:
        -------
        - self: object
        """
        if not self.steps:
            return self

        transformed = X
        for _, transformer in self.steps[:-1]:
            transformed = transformer.fit_transform(transformed, y)

        _, estimator = self.steps[-1]
        estimator.fit(transformed, y)
        return self

    def transform(self, X):
        """
        Apply every step's transform to the data in order.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: output of the final step's transform

        Raises:
        ------
        - TypeError: if any step (including the last) does not implement transform
        - RuntimeError: propagated from a step that has not been fitted
        """
        transformed = X
        for name, transformer in self.steps:
            if not hasattr(transformer, "transform"):
                raise TypeError(f"Pipeline step {name!r} must implement transform")
            transformed = transformer.transform(transformed)
        return transformed

    def fit_transform(self, X, y=None):
        """
        Fit every step and transform the data in a single pass.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,), passed to every step's fit (default None)

        Returns:
        -------
        - X_out: output of the final step's fit_transform

        Raises:
        ------
        - TypeError: if any step (including the last) does not implement fit_transform
        """
        transformed = X
        for name, transformer in self.steps:
            if not hasattr(transformer, "fit_transform"):
                raise TypeError(f"Pipeline step {name!r} must implement fit_transform")
            transformed = transformer.fit_transform(transformed, y)
        return transformed

    def predict(self, X):
        """
        Transform the data through all transformers, then predict with the final step.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - y_pred: output of the final step's predict

        Raises:
        ------
        - ValueError: if the pipeline has no steps
        - TypeError: if any step before the last does not implement transform
        - TypeError: if the final step does not implement predict
        """
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
