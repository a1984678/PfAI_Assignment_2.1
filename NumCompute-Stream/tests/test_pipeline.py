"""Tests for ``numcompute_stream.pipeline.Pipeline``.

Cover ``partial_fit()`` for chained models and transformers, and incremental
transformation + prediction through the pipeline.
"""

import pytest

from numcompute_stream.pipeline import Estimator, Transformer


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
