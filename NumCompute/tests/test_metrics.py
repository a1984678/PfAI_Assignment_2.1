import numpy as np
import pytest

from numcompute import metrics


# ---------------------------------------------------------------------
# Contract tests (must hold for every metric that takes y_true, y_pred)
# ---------------------------------------------------------------------

PAIR_METRICS = [
    metrics.accuracy,
    metrics.precision,
    metrics.recall,
    metrics.f1,
    metrics.confusion_matrix,
    metrics.mse,
]


@pytest.mark.parametrize("metric", PAIR_METRICS, ids=lambda f: f.__name__)
def test_empty_input_raises(metric):
    with pytest.raises(ValueError, match="Inputs must not be empty."):
        metric(np.array([]), np.array([]))


@pytest.mark.parametrize("metric", PAIR_METRICS, ids=lambda f: f.__name__)
def test_length_mismatch_raises(metric):
    with pytest.raises(ValueError, match="Inputs must have the same length."):
        metric(np.array([0, 1, 1]), np.array([0, 1]))


@pytest.mark.parametrize("metric", PAIR_METRICS, ids=lambda f: f.__name__)
def test_2d_input_raises(metric):
    with pytest.raises(ValueError, match="Inputs must be 1-dimensional."):
        metric(np.array([[0, 1]]), np.array([[0, 1]]))


@pytest.mark.parametrize("metric", PAIR_METRICS, ids=lambda f: f.__name__)
def test_nan_input_raises(metric):
    with pytest.raises(ValueError, match="Inputs must not contain NaN values."):
        metric(np.array([0.0, np.nan]), np.array([0.0, 1.0]))


@pytest.mark.parametrize("metric", PAIR_METRICS, ids=lambda f: f.__name__)
def test_non_contiguous_input(metric):
    """Strided views (every second element) must give the same result as a copy."""
    y_true = np.array([0, 9, 1, 9, 1, 9, 0, 9])[::2]   # [0, 1, 1, 0]
    y_pred = np.array([0, 9, 1, 9, 0, 9, 0, 9])[::2]   # [0, 1, 0, 0]
    expected = metric(y_true.copy(), y_pred.copy())
    np.testing.assert_array_equal(metric(y_true, y_pred), expected)


@pytest.mark.parametrize("metric", PAIR_METRICS, ids=lambda f: f.__name__)
def test_input_not_mutated(metric):
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    true_copy, pred_copy = y_true.copy(), y_pred.copy()
    metric(y_true, y_pred)
    np.testing.assert_array_equal(y_true, true_copy)
    np.testing.assert_array_equal(y_pred, pred_copy)


# ---------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------

class TestConfusionMatrix:

    def test_basic(self):
        cm = metrics.confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
        np.testing.assert_array_equal(cm, np.array([[2, 0], [1, 1]]))

    def test_multiclass(self):
        cm = metrics.confusion_matrix([0, 1, 2, 2, 1], [0, 2, 2, 2, 1])
        np.testing.assert_array_equal(cm, np.array([[1, 0, 0], [0, 1, 1], [0, 0, 2]]))

    def test_string_labels(self):
        cm = metrics.confusion_matrix(["cat", "dog", "dog"], ["cat", "cat", "dog"])
        np.testing.assert_array_equal(cm, np.array([[1, 0], [1, 1]]))  # rows: cat, dog

    def test_labels_order(self):
        cm = metrics.confusion_matrix([0, 1, 1], [0, 1, 0], labels=[1, 0])
        np.testing.assert_array_equal(cm, np.array([[1, 1], [0, 1]]))

    def test_counts_sum_to_n(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 4, size=500)
        y_pred = rng.integers(0, 4, size=500)
        assert metrics.confusion_matrix(y_true, y_pred).sum() == 500

    def test_single_class(self):
        cm = metrics.confusion_matrix([1, 1, 1], [1, 1, 1])
        np.testing.assert_array_equal(cm, np.array([[3]]))

    def test_label_missing_from_labels_raises(self):
        with pytest.raises(ValueError, match="not present in `labels`"):
            metrics.confusion_matrix([0, 1, 2], [0, 1, 1], labels=[0, 1])

    def test_duplicate_labels_raises(self):
        with pytest.raises(ValueError, match="labels must not contain duplicates."):
            metrics.confusion_matrix([0, 1], [0, 1], labels=[0, 1, 1])


# ---------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------

class TestAccuracy:

    def test_basic(self):
        assert metrics.accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.75)

    def test_perfect(self):
        assert metrics.accuracy([2, 0, 1], [2, 0, 1]) == 1.0

    def test_all_wrong(self):
        assert metrics.accuracy([0, 0, 1], [1, 1, 0]) == 0.0

    def test_string_labels(self):
        assert metrics.accuracy(["a", "b"], ["a", "a"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------
# precision / recall / f1
# ---------------------------------------------------------------------

class TestPrecisionRecallF1:

    # y_true = [0, 1, 1, 0], y_pred = [0, 1, 0, 0] -> TP=1, FP=0, FN=1
    Y_TRUE = [0, 1, 1, 0]
    Y_PRED = [0, 1, 0, 0]

    def test_precision_binary(self):
        assert metrics.precision(self.Y_TRUE, self.Y_PRED) == pytest.approx(1.0)

    def test_recall_binary(self):
        assert metrics.recall(self.Y_TRUE, self.Y_PRED) == pytest.approx(0.5)

    def test_f1_binary(self):
        assert metrics.f1(self.Y_TRUE, self.Y_PRED) == pytest.approx(2 / 3)

    def test_pos_label(self):
        # treat 0 as the positive class: TP=2, FP=1, FN=0
        assert metrics.precision(self.Y_TRUE, self.Y_PRED, pos_label=0) == pytest.approx(2 / 3)
        assert metrics.recall(self.Y_TRUE, self.Y_PRED, pos_label=0) == pytest.approx(1.0)

    def test_macro_average(self):
        y_true = [0, 1, 2, 2, 1]
        y_pred = [0, 2, 2, 2, 1]
        # per-class precision: [1, 1, 2/3]; recall: [1, 1/2, 1]
        expected_precision = (1 + 1 + 2 / 3) / 3
        result = metrics.precision(y_true, y_pred, average="macro")
        assert result == pytest.approx(expected_precision)
        assert metrics.recall(y_true, y_pred, average="macro") == pytest.approx((1 + 0.5 + 1) / 3)
        f1_per_class = [1.0, 2 / 3, 0.8]
        assert metrics.f1(y_true, y_pred, average="macro") == pytest.approx(np.mean(f1_per_class))

    def test_string_labels(self):
        y_true = ["spam", "ham", "spam"]
        y_pred = ["spam", "spam", "spam"]
        assert metrics.precision(y_true, y_pred, pos_label="spam") == pytest.approx(2 / 3)

    # Made the decision to return 0.0 instead of raising when a division would be 0 / 0.
    def test_no_predicted_positives_gives_zero(self):
        assert metrics.precision([1, 1, 0], [0, 0, 0]) == 0.0
        assert metrics.f1([1, 1, 0], [0, 0, 0]) == 0.0

    def test_all_negative_gives_zero(self):
        # pos_label 1 never appears, only the negative class is present
        assert metrics.recall([0, 0, 0], [0, 0, 0]) == 0.0

    def test_binary_with_three_classes_raises(self):
        with pytest.raises(ValueError, match="average='binary' needs at most 2 classes"):
            metrics.precision([0, 1, 2], [0, 1, 2])

    def test_invalid_pos_label_raises(self):
        with pytest.raises(ValueError, match="pos_label=1 is not a valid label"):
            metrics.recall(["cat", "dog"], ["cat", "dog"])

    def test_invalid_average_raises(self):
        with pytest.raises(ValueError, match="average must be 'binary' or 'macro'"):
            metrics.f1([0, 1], [0, 1], average="weighted")


# ---------------------------------------------------------------------
# mse
# ---------------------------------------------------------------------

class TestMSE:

    def test_basic(self):
        assert metrics.mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(4 / 3)

    def test_perfect(self):
        assert metrics.mse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_integer_input(self):
        assert metrics.mse([0, 0], [1, 3]) == pytest.approx(5.0)

    def test_large_values(self):
        # (1e10 + 1) - 1e10 must still be exactly 1
        assert metrics.mse([1e10], [1e10 + 1]) == pytest.approx(1.0)

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="y_true must be numeric"):
            metrics.mse(["a", "b"], [1, 2])

    def test_infinite_raises(self):
        with pytest.raises(ValueError, match="must contain only finite values"):
            metrics.mse([np.inf, 1.0], [1.0, 1.0])


# ---------------------------------------------------------------------
# roc_curve / auc
# ---------------------------------------------------------------------

class TestRocAuc:

    def test_roc_curve_basic(self):
        fpr, tpr, thresholds = metrics.roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
        np.testing.assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
        assert thresholds[0] == np.inf

    def test_auc_of_roc(self):
        fpr, tpr, _ = metrics.roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        assert metrics.auc(fpr, tpr) == pytest.approx(0.75)

    def test_perfect_classifier(self):
        fpr, tpr, _ = metrics.roc_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert metrics.auc(fpr, tpr) == pytest.approx(1.0)

    def test_all_scores_tied(self):
        # every score equal -> one threshold, a straight diagonal, AUC 0.5
        fpr, tpr, _ = metrics.roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(fpr, [0, 1])
        np.testing.assert_allclose(tpr, [0, 1])
        assert metrics.auc(fpr, tpr) == pytest.approx(0.5)

    def test_curve_starts_at_origin_and_ends_at_one(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=200)
        y_score = rng.random(200)
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        assert (fpr[0], tpr[0]) == (0.0, 0.0)
        assert (fpr[-1], tpr[-1]) == (1.0, 1.0)
        assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)

    def test_one_class_only_raises(self):
        with pytest.raises(ValueError, match="both positive"):
            metrics.roc_curve([1, 1, 1], [0.2, 0.5, 0.9])

    def test_multiclass_raises(self):
        with pytest.raises(ValueError, match="binary classification"):
            metrics.roc_curve([0, 1, 2], [0.2, 0.5, 0.9])

    def test_auc_triangle(self):
        assert metrics.auc([0, 1], [0, 1]) == pytest.approx(0.5)

    def test_auc_decreasing_x(self):
        assert metrics.auc([1, 0], [1, 0]) == pytest.approx(0.5)

    def test_auc_not_monotonic_raises(self):
        with pytest.raises(ValueError, match="x must be monotonic"):
            metrics.auc([0, 1, 0.5], [0, 1, 1])

    def test_auc_single_point_raises(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            metrics.auc([0.5], [0.5])
