"""Chained streaming pipeline.

- Must support ``partial_fit()`` for models and transformers
- Support incremental transformation + prediction in chained pipeline

Example
-------
>>> pipe = Pipeline([
...     ('scale', StandardScaler()),
...     ('model', DecisionTreeClassifier(max_depth=5, buffer_size=500))
... ])
>>> pipe.partial_fit(X_chunk, y_chunk)

Baseline (adapted from NumCompute, Assignment 2.1)
--------------------------------------------------
Minimal Transformer / Estimator protocol:
- Preprocessors: fit, transform
- Models: fit, predict (no ML model implementation required)

Pipeline chaining, e.g.:

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('encode', OneHotEncoder())
    ])
    X_tr = pipe.fit_transform(X)

Streaming (2.2)
---------------
- ``Pipeline.partial_fit(X_chunk, y_chunk)``: each transformer is updated with the chunk
  and transforms it for the next step; the final step gets ``partial_fit``. Every step is
  checked for streaming support before any step is updated.
- ``Pipeline.predict_proba(X)``: transform, then the final step's ``predict_proba``.
- ``Pipeline.supports_partial_fit``: True if every step streams (pipelines can be nested).
"""

import numpy as np


def _supports_partial_fit(step):
    """The step's supports_partial_fit flag, or (no flag) whether it has partial_fit."""
    return getattr(step, "supports_partial_fit", hasattr(step, "partial_fit"))


class Transformer:
    """
    Base class for all transformers.

    Subclasses must implement fit and transform; fit_transform is provided.

    API:
    ----
    - fit(X, y=None) -> self
    - transform(X) -> X_out
    - fit_transform(X, y=None) -> X_out
    - partial_fit(X, y=None) -> self (optional, streaming)

    Attributes:
    ----------
    - supports_partial_fit: bool class attribute, False here. Subclasses that implement
    partial_fit set it to True so a Pipeline can check streaming support up front.
    """

    supports_partial_fit = False

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

    def partial_fit(self, X, y=None):
        """
        Update the fitted parameters with one chunk of data (streaming).

        Unlike fit, repeated calls accumulate: each call updates the state learned from
        earlier chunks instead of starting again.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), one chunk of the stream
        - y: array-like of shape (n_samples,), ignored by most transformers (default None)

        Returns:
        -------
        - self: object

        Raises:
        ------
        - NotImplementedError: if the subclass does not support streaming updates
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support partial_fit"
        )

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
    - partial_fit(X, y=None) -> self (optional, streaming)

    Attributes:
    ----------
    - supports_partial_fit: bool class attribute, False here. Subclasses that implement
    partial_fit set it to True so a Pipeline can check streaming support up front.
    """

    supports_partial_fit = False

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit the model to the data.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,), target values (None if the model needs none)

        Returns:
        -------
        - self: object

        Raises:
        ------
        - NotImplementedError: if the subclass does not override fit
        """
        raise NotImplementedError

    def partial_fit(self, X, y=None):
        """
        Update the model with one chunk of data (streaming).

        Unlike fit, repeated calls accumulate: each call updates the state learned from
        earlier chunks instead of starting again.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), one chunk of the stream
        - y: array-like of shape (n_samples,), target values for the chunk (default None)

        Returns:
        -------
        - self: object

        Raises:
        ------
        - NotImplementedError: if the subclass does not support streaming updates
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support partial_fit"
        )

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


def _fit_transform_step(name, step, X, y):
    """
    Fit one step and return its transformed output.

    Uses step.fit_transform(X, y) if the step has it, otherwise step.fit(X, y) followed by
    step.transform(X), so a plain object with only fit and transform also works.

    Raises:
    ------
    - TypeError: if the step has neither fit_transform nor both fit and transform
    """
    if hasattr(step, "fit_transform"):
        return step.fit_transform(X, y)
    if hasattr(step, "fit") and hasattr(step, "transform"):
        step.fit(X, y)
        return step.transform(X)
    raise TypeError(f"Pipeline step {name!r} must implement fit_transform, or fit and transform")


class Pipeline:
    """
    Chain transformers, optionally ending in an estimator.

    Each step's output is the next step's input. All steps except the last
    must be transformers; the last step may be a transformer or an estimator.

    Parameters:
    ----------
    - steps: list of (name, object) tuples
        Intermediate objects must implement fit_transform, or fit and transform; the final
        object must implement fit.

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
    - y is passed to every step's fit (as fit(X, y)), so every step's fit must accept a second
    argument; the Transformer and Estimator base classes already do.
    - Any object with fit and transform works as a step, it does not have to inherit from
    Transformer. A step with fit_transform is called through it, otherwise fit then transform.
    - An empty Pipeline is allowed: fit does nothing and transform returns X unchanged.
    - Streaming: partial_fit updates then transforms each step per chunk. Each transformer's
    output width is recorded by the first fit / partial_fit and must not change later, since
    the final model's buffer can't mix widths; fit resets it.
    """

    def __init__(self, steps):
        """
        Store and validate the pipeline steps.

        Parameters:
        ----------
        - steps: list of (name, object) tuples

        Raises:
        ------
        - TypeError: if any step before the last has neither fit_transform nor fit and transform
        - TypeError: if the final step does not implement fit
        """
        self.steps = steps
        self._widths = {}         # output columns of each transformer, by step index
        if not self.steps:
            return

        for name, transformer in self.steps[:-1]:
            if not (hasattr(transformer, "fit_transform")
                    or (hasattr(transformer, "fit") and hasattr(transformer, "transform"))):
                raise TypeError(f"Pipeline step {name!r} must be a transformer "
                                "(fit_transform, or fit and transform)")

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
        self._widths = {}         # fit starts again: any output width is allowed
        if not self.steps:
            return self

        transformed = X
        for i, (name, transformer) in enumerate(self.steps[:-1]):
            transformed = _fit_transform_step(name, transformer, transformed, y)
            self._check_width(i, name, transformed)

        _, estimator = self.steps[-1]
        estimator.fit(transformed, y)
        return self

    def _check_width(self, i, name, transformed):
        """
        Record step i's output width on first use; raise if a later chunk differs.

        Raises:
        ------
        - ValueError: if the number of output columns changed since it was recorded
        """
        if np.ndim(transformed) != 2:
            return
        width = np.shape(transformed)[1]
        expected = self._widths.setdefault(i, width)
        if width != expected:
            raise ValueError(
                f"Pipeline step {name!r} output changed from {expected} to {width} columns "
                "between chunks, so later steps can't use it (e.g. a OneHotEncoder found a "
                "new category). Fit that step on all known categories first.")

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
        - TypeError: if any step (including the last) has neither fit_transform nor fit and
        transform
        """
        self._widths = {}         # fit starts again: any output width is allowed
        transformed = X
        for i, (name, transformer) in enumerate(self.steps):
            transformed = _fit_transform_step(name, transformer, transformed, y)
            if i < len(self.steps) - 1:
                self._check_width(i, name, transformed)
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

    @property
    def supports_partial_fit(self):
        """
        True if every step supports partial_fit (so a Pipeline can be a step of another).

        A step counts as streaming if its supports_partial_fit flag is True, or, for an
        object without the flag, if it has a partial_fit method. An empty Pipeline is True.
        """
        return all(_supports_partial_fit(step) for _, step in self.steps)

    def partial_fit(self, X, y=None):
        """
        Update every step with one chunk, in order (streaming).

        Each transformer is updated with partial_fit(X, y) and then transforms the chunk for
        the next step; the final step gets partial_fit(X, y). So the chunk is transformed
        with statistics that already include it, as in fit_transform.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), one chunk of the stream
        - y: array-like of shape (n_samples,), labels for the chunk (default None)

        Returns:
        -------
        - self

        Raises:
        ------
        - TypeError: if any step does not support partial_fit, checked for all steps before
        any step is updated
        - ValueError: if a transformer's output width differs from the first chunk (e.g. a
        OneHotEncoder finds a new category). Steps before it have already absorbed the chunk.
        Fit the encoder on all known categories first to keep the width fixed.
        - Anything a step raises for an invalid chunk
        """
        for name, step in self.steps:              # check all steps before touching any
            if not _supports_partial_fit(step):
                raise TypeError(f"Pipeline step {name!r} does not support partial_fit")
        if not self.steps:
            return self

        transformed = X
        for i, (name, transformer) in enumerate(self.steps[:-1]):
            transformer.partial_fit(transformed, y)
            transformed = transformer.transform(transformed)
            self._check_width(i, name, transformed)

        _, estimator = self.steps[-1]
        estimator.partial_fit(transformed, y)
        return self

    def predict_proba(self, X):
        """
        Transform the data through all transformers, then call the final predict_proba.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - np.ndarray of shape (n_samples, n_classes), from the final step

        Raises:
        ------
        - ValueError: if the pipeline has no steps
        - TypeError: if the final step does not implement predict_proba
        """
        if not self.steps:
            raise ValueError("Cannot predict with an empty pipeline")

        transformed = X
        for name, transformer in self.steps[:-1]:
            if not hasattr(transformer, "transform"):
                raise TypeError(f"Pipeline step {name!r} must implement transform")
            transformed = transformer.transform(transformed)

        name, estimator = self.steps[-1]
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(f"Final pipeline step {name!r} must implement predict_proba")
        return estimator.predict_proba(transformed)
