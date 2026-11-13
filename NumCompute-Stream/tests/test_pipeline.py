"""Tests for ``numcompute_stream.pipeline.Pipeline``.

Cover ``partial_fit()`` for chained models and transformers, and incremental
transformation + prediction through the pipeline.
"""

import numpy as np
import pytest

from numcompute_stream.pipeline import Estimator, Pipeline, Transformer
from numcompute_stream.preprocessing import OneHotEncoder, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.mark.parametrize("base", [Transformer, Estimator])
def test_base_partial_fit_raises_naming_subclass(base):
    """The default partial_fit names the concrete class that lacks it."""
    class NoStream(base):
        pass

    with pytest.raises(NotImplementedError, match="NoStream does not support partial_fit"):
        NoStream().partial_fit([[1.0]])


@pytest.mark.parametrize("base", [Transformer, Estimator])
def test_base_supports_partial_fit_defaults_false(base):
    """Base classes advertise no streaming support unless a subclass opts in."""
    assert base.supports_partial_fit is False

    class Streaming(base):
        supports_partial_fit = True

    assert Streaming.supports_partial_fit is True
    assert base.supports_partial_fit is False


# ---------------------------------------------------------------------
# Pipeline.partial_fit
# ---------------------------------------------------------------------

def _chunks(n=300, chunk_size=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=50, scale=10, size=(n, 3))
    y = np.where(X[:, 0] > X[:, 1], "a", "b")
    return X, y, [(X[s:s + chunk_size], y[s:s + chunk_size]) for s in range(0, n, chunk_size)]


def test_partial_fit_chain_matches_manual_steps():
    """Spec example: each chunk updates the scaler, is scaled, then goes to the tree."""
    X, y, chunks = _chunks()
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=3, buffer_size=50))])
    scaler, tree = StandardScaler(), DecisionTreeClassifier(max_depth=3, buffer_size=50)
    assert pipe.supports_partial_fit is True

    for X_chunk, y_chunk in chunks:
        assert pipe.partial_fit(X_chunk, y_chunk) is pipe
        scaler.partial_fit(X_chunk)
        tree.partial_fit(scaler.transform(X_chunk), y_chunk)

    assert pipe.steps[1][1].n_retrains_ == tree.n_retrains_ > 0
    np.testing.assert_array_equal(pipe.predict(X), tree.predict(scaler.transform(X)))
    np.testing.assert_array_equal(pipe.predict_proba(X),
                                  tree.predict_proba(scaler.transform(X)))


def test_partial_fit_checks_every_step_before_changing_state():
    class FitOnly:                                   # duck-typed, no partial_fit at all
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    class NoStream(Transformer):                     # inherits partial_fit, flag is False
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    X, y, chunks = _chunks()
    for bad in (FitOnly(), NoStream()):
        scaler = StandardScaler()
        pipe = Pipeline([("scale", scaler), ("bad", bad), ("model", DecisionTreeClassifier())])
        assert pipe.supports_partial_fit is False
        with pytest.raises(TypeError, match="step 'bad' does not support partial_fit"):
            pipe.partial_fit(*chunks[0])
        assert not hasattr(scaler, "n_features_in_")  # the scaler saw nothing

    class DuckStream(FitOnly):                       # partial_fit without the flag: accepted
        def partial_fit(self, X, y=None):
            return self

    Pipeline([("duck", DuckStream()), ("model", DecisionTreeClassifier())]).partial_fit(X, y)
    assert Pipeline([]).partial_fit(X, y).supports_partial_fit is True

    # X / y length mismatch: only the final step would notice, so check it up front
    scaler = StandardScaler()
    pipe = Pipeline([("scale", scaler), ("model", DecisionTreeClassifier())])
    with pytest.raises(ValueError, match="same number of samples. Got 20 and 19"):
        pipe.partial_fit(chunks[0][0], chunks[0][1][:-1])
    assert not hasattr(scaler, "n_features_in_")      # the scaler saw nothing


def test_partial_fit_transformer_output_width_change_raises_clearly():
    X1, X2 = np.array([[0.0], [1.0]] * 5), np.array([[0.0], [2.0]] * 5)
    y = np.array([0, 1] * 5)

    pipe = Pipeline([("encode", OneHotEncoder()),
                     ("model", DecisionTreeClassifier(buffer_size=100))])
    pipe.partial_fit(X1, y)
    # the new category 2.0 adds an output column; the tree's buffer can't mix widths
    with pytest.raises(ValueError, match="step 'encode' output changed from 2 to 3 columns"):
        pipe.partial_fit(X2, y)

    # fitting the encoder on all known categories first keeps the width fixed
    encoder = OneHotEncoder().fit([[0.0], [1.0], [2.0]])
    fixed = Pipeline([("encode", encoder), ("model", DecisionTreeClassifier(buffer_size=20))])
    fixed.partial_fit(X1, y).partial_fit(X2, y)
    np.testing.assert_array_equal(fixed.predict([[0.0], [1.0], [2.0]]), [0, 1, 1])

    # fit starts again, so a new width is allowed afterwards
    pipe.fit(np.array([[0.0], [1.0], [2.0]] * 2), [0, 1, 1] * 2)
    pipe.partial_fit(np.array([[0.0], [1.0], [2.0]]), [0, 1, 1])
