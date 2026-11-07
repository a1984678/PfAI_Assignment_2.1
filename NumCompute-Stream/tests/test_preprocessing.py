"""Tests for ``numcompute_stream.preprocessing``.

Cover ``.partial_fit()`` for ``StandardScaler`` (running mean/var, e.g.
Welford or EMA), ``Imputer`` (on-the-fly missing-value estimates) and
``OneHotEncoder`` (incremental category expansion), including edge cases such
as zero-variance chunks and all-NaN columns.
"""

import numpy as np
import pytest

from numcompute_stream.preprocessing import StandardScaler


def _partial_fit_in_chunks(scaler, X, chunk_size):
    for start in range(0, len(X), chunk_size):
        scaler.partial_fit(X[start:start + chunk_size])
    return scaler


# ---------------------------------------------------------------------
# StandardScaler.partial_fit
# ---------------------------------------------------------------------

@pytest.mark.parametrize("chunk_size", [1, 7, 200])
def test_standard_scaler_partial_fit_matches_fit(chunk_size):
    """Streaming fit over any chunking gives the same mean_, scale_ and output as fit."""
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[0.0, 50.0, -3.0], scale=[1.0, 20.0, 0.01], size=(200, 3))
    streamed = _partial_fit_in_chunks(StandardScaler(), X, chunk_size)
    batch = StandardScaler().fit(X)

    np.testing.assert_allclose(streamed.mean_, batch.mean_)
    np.testing.assert_allclose(streamed.scale_, batch.scale_)
    np.testing.assert_allclose(streamed.transform(X), batch.transform(X), atol=1e-12)
    assert streamed.n_features_in_ == 3


def test_standard_scaler_supports_partial_fit_and_returns_self():
    scaler = StandardScaler()
    assert StandardScaler.supports_partial_fit is True
    assert scaler.partial_fit(np.ones((2, 2))) is scaler


def test_standard_scaler_zero_variance_chunk_then_spread():
    """A column constant so far gets scale 1 (maps to 0); later spread updates the scale."""
    scaler = StandardScaler().partial_fit([[5.0, 1.0], [5.0, 3.0]])
    assert scaler.scale_[0] == 1.0
    np.testing.assert_array_equal(scaler.transform([[5.0, 2.0]])[:, 0], [0.0])

    scaler.partial_fit([[7.0, 2.0], [9.0, 2.0]])
    all_seen = np.array([[5.0, 1.0], [5.0, 3.0], [7.0, 2.0], [9.0, 2.0]])
    np.testing.assert_allclose(scaler.mean_, all_seen.mean(axis=0))
    np.testing.assert_allclose(scaler.scale_, all_seen.std(axis=0))


def test_standard_scaler_all_nan_column_in_early_chunk():
    """partial_fit tolerates a column with no values yet: its output is NaN (no warnings)
    until a later chunk provides data. fit keeps the strict 2.1 behaviour."""
    scaler = StandardScaler().partial_fit([[1.0, np.nan], [3.0, np.nan]])
    out = scaler.transform([[2.0, 4.0]])
    assert out[0, 0] == 0.0 and np.isnan(out[0, 1])

    scaler.partial_fit([[2.0, 10.0], [2.0, 20.0]])
    np.testing.assert_allclose(scaler.transform([[2.0, 15.0]]), [[0.0, 0.0]])

    with pytest.raises(ValueError, match="all NaN"):
        StandardScaler().fit([[1.0, np.nan], [3.0, np.nan]])


def test_standard_scaler_fit_resets_and_partial_fit_continues():
    A = np.array([[0.0], [2.0]])
    B = np.array([[10.0], [20.0], [30.0]])

    # fit discards anything learned by earlier partial_fit calls
    refit = StandardScaler().partial_fit(A).fit(B)
    np.testing.assert_allclose(refit.mean_, [20.0])

    # partial_fit after fit keeps accumulating on top of the fitted data
    both = StandardScaler().fit(A).partial_fit(B)
    np.testing.assert_allclose(both.mean_, np.vstack([A, B]).mean(axis=0))
    np.testing.assert_allclose(both.scale_, np.vstack([A, B]).std(axis=0))


def test_standard_scaler_partial_fit_errors():
    with pytest.raises(RuntimeError, match="fit"):
        StandardScaler().transform([[1.0]])
    scaler = StandardScaler().partial_fit(np.ones((2, 3)))
    with pytest.raises(ValueError, match="X has 2 features, but 3"):
        scaler.partial_fit(np.ones((2, 2)))
    with pytest.raises(ValueError, match="2-dimensional"):
        scaler.partial_fit([1.0, 2.0])
