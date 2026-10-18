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


def test_pipeline_rejects_non_transformer_intermediate_step():
    with pytest.raises(TypeError):
        Pipeline([
            ("impute", "not a transformer"),
            ("scale", StandardScaler())
        ])


def test_pipeline_rejects_final_step_without_fit():
    with pytest.raises(TypeError):
        Pipeline([
            ("scale", StandardScaler()),
            ("estimator", "not an estimator")
        ])


def test_pipeline_empty():
    pipe = Pipeline([])
    X = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(pipe.fit_transform(X), X)
    np.testing.assert_array_equal(pipe.transform(X), X)


# ---------------------------------------------------------------------
# Dummy class testing
# ---------------------------------------------------------------------

class DummyEstimator(Estimator):
    """Minimal estimator: predicts the mean of each row."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.mean(X, axis=1)


class MeanTargetEstimator(Estimator):
    """Learns the mean of y; predicts row sum of X plus that mean."""

    def fit(self, X, y=None):
        self.y_mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.sum(X, axis=1) + self.y_mean_

class DummyTransformer(Transformer):
    """Adds the mean of y to each feature."""

    def fit(self, X, y=None):
        self.y_mean_ = np.mean(y)
        return self

    def transform(self, X):
        return X + self.y_mean_


def test_predict_with_dummy_estimator():
    X = np.array([[1, 2], [3, 4]])
    pipe = Pipeline([
        ("estimator", DummyEstimator())
    ])
    pipe.fit(X)
    predictions = pipe.predict(X)
    np.testing.assert_array_equal(predictions, np.array([1.5, 3.5]))


def test_pipeline_with_dummy_estimator_and_transformer():
    X = np.array([[1, 2], [3, 4]])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("estimator", DummyEstimator())
    ])
    pipe.fit(X)
    predictions = pipe.predict(X)
    # Scaled X is [[-1, -1], [1, 1]], so row means are [-1, 1]
    np.testing.assert_array_almost_equal(predictions, np.array([-1.0, 1.0]))


def test_pipeline_passes_y_to_estimator():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([10, 20])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("estimator", MeanTargetEstimator())
    ])
    pipe.fit(X, y)

    # Scaled X is [[-1, -1], [1, 1]] -> row sums [-2, 2]; mean(y) = 15
    np.testing.assert_array_almost_equal(pipe.predict(X), [13.0, 17.0])

    # New data uses the fitted scaler (mean [2, 3], std [1, 1]):
    # [[2, 3]] -> [[0, 0]] -> row sum 0; 0 + 15 = 15
    X_new = np.array([[2, 3]])
    np.testing.assert_array_almost_equal(pipe.predict(X_new), [15.0])


def test_pipeline_passes_y_to_transformer():

    X = np.array([[1, 2], [3, 4]])
    y = np.array([10, 20])
    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("estimator", DummyEstimator())
    ])
    pipe.fit(X, y)

    # The transformer adds mean(y) = 15 to each feature:
    # [[1+15, 2+15], [3+15, 4+15]] -> [[16, 17], [18, 19]]
    # Row means: [16.5, 18.5]
    np.testing.assert_array_almost_equal(pipe.predict(X), [16.5, 18.5])
