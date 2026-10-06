import numpy as np
import pytest

from numcompute.pipeline import Pipeline, Transformer, Estimator
from numcompute.preprocessing import StandardScaler, SimpleImputer


def test_imputer_to_scailer_manual():
    X = np.array([[1, np.nan], [3, 30]])

    X_imp = SimpleImputer().fit_transform(X)
    np.testing.assert_almost_equal(X_imp, ([1, 0], [3, 30]))

    X_sc = StandardScaler().fit_transform(X_imp)
    np.testing.assert_almost_equal(X_sc, ([-1, -1], [1, 1]))


def test_imputer_to_scailer_pipeline():
    X = np.array([[1, np.nan], [3, 30]])

    pipe = Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler())
    ])

    X_tr = pipe.fit_transform(X)
    np.testing.assert_almost_equal(X_tr, ([-1, -1], [1, 1]))


def test_imputer_to_scailer_pipeline_transform():
    X = np.array([[1, np.nan], [3, 30]])
    X_new = np.array([[np.nan, 45], [2, np.nan]])

    pipe = Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler())
    ])

    X_tr = pipe.fit_transform(X)
    np.testing.assert_almost_equal(X_tr, ([-1, -1], [1, 1]))

    X_new_tr = pipe.transform(X_new)
    np.testing.assert_almost_equal(X_new_tr, ([-2, 2], [0, -1]))
