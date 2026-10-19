import numpy as np

from numcompute import io, metrics, preprocessing, rank, stats
from numcompute.pipeline import Pipeline


def test_csv_with_text_column_goes_through_encoder(tmp_path):
    """io -> preprocessing: a text column read by load_csv can be one-hot encoded."""
    file = tmp_path / "data.csv"
    file.write_text("red,1\nblue,2\nred,3\n")
    data = io.load_csv(file, dtype=object)

    encoded = preprocessing.OneHotEncoder().fit_transform(data[:, :1])
    np.testing.assert_array_equal(encoded, [[0, 1], [1, 0], [0, 1]])


def test_csv_with_missing_values_through_a_pipeline(tmp_path):
    """io -> pipeline (imputer + scaler) -> stats: the demo flow end to end."""
    file = tmp_path / "data.csv"
    file.write_text("1,10\n3,\n5,30\n")
    X = io.load_csv(file)  # the blank cell becomes NaN

    pipe = Pipeline([("impute", preprocessing.SimpleImputer(fill_value=20)),
                     ("scale", preprocessing.StandardScaler())])
    X_scaled = pipe.fit_transform(X)

    assert X_scaled.shape == (3, 2)
    np.testing.assert_allclose(stats.mean(X_scaled, axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(stats.std(X_scaled, axis=0), [1.0, 1.0])


def test_pipeline_prediction_scored_with_metrics_and_ranked():
    """pipeline -> metrics and rank: model output feeds the evaluation functions."""
    class ThresholdModel:
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return (X[:, 0] > 0).astype(int)

    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 1, 1, 1])
    pipe = Pipeline([("scale", preprocessing.StandardScaler()), ("model", ThresholdModel())])
    y_pred = pipe.fit(X, y).predict(X)

    assert metrics.accuracy(y, y_pred) == 0.75
    np.testing.assert_array_equal(rank.rank(y_pred, "dense"), [1, 1, 2, 2])
