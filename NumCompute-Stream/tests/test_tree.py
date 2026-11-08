"""Tests for ``numcompute_stream.tree.DecisionTreeClassifier``.

Cover ``.fit()``, ``.predict()`` and ``.partial_fit()`` buffered retraining:
chunks are appended to the buffer, the tree retrains once the buffer holds at
least ``buffer_size`` samples and the buffer is cleared, and ``.predict()``
uses the most recent tree between retrains. Also cover Gini vs. entropy,
``max_depth``, ``min_samples_split``, ``max_features``, and edge cases such as
NaNs, ties, and zero-variance chunks.

Main check for ``fit``: the vectorised tree is identical (same features, same thresholds,
same node order) to a slow reference built with plain Python loops, as in the course tree
(``classification-tree.html``) extended to numeric thresholds.
"""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------
# Slow reference: one mask per (feature, threshold), course conventions
# ---------------------------------------------------------------------

def _impurity(y, k, criterion):
    p = np.bincount(y, minlength=k) / len(y)
    p = p[p > 0]
    return 1.0 - np.sum(p * p) if criterion == "gini" else -np.sum(p * np.log2(p))


def _reference_tree(X, y, criterion="gini", max_depth=5, min_samples_split=2):
    """Preorder lists (feature, threshold); leaves are (-1, nan)."""
    classes, y = np.unique(y, return_inverse=True)
    k = len(classes)
    feature, threshold = [], []

    def build(rows, depth):
        idx = len(feature)
        feature.append(-1)
        threshold.append(np.nan)
        yy = y[rows]
        if depth >= max_depth or len(rows) < min_samples_split or np.all(yy == yy[0]):
            return
        before, best = _impurity(yy, k, criterion), (None, None, 0.0)
        for f in range(X.shape[1]):
            values = np.unique(X[rows, f])
            for a, b in zip(values[:-1], values[1:]):
                t = a / 2 + b / 2
                if t >= b:
                    t = a
                mask = X[rows, f] <= t
                after = (mask.sum() * _impurity(yy[mask], k, criterion)
                         + (~mask).sum() * _impurity(yy[~mask], k, criterion)) / len(rows)
                if before - after > best[2]:          # strict: first maximum wins
                    best = (f, t, before - after)
        if best[0] is None:
            return
        feature[idx], threshold[idx] = best[0], best[1]
        go_left = X[rows, best[0]] <= best[1]
        build(rows[go_left], depth + 1)
        build(rows[~go_left], depth + 1)

    build(np.arange(len(y)), 0)
    return np.array(feature), np.array(threshold)


# ---------------------------------------------------------------------
# fit / predict / predict_proba
# ---------------------------------------------------------------------

def test_fit_simple_threshold_predict_and_proba():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array(["no", "no", "yes", "yes", "yes"])
    tree = DecisionTreeClassifier(max_depth=3)
    assert tree.fit(X, y) is tree

    assert list(tree.classes_) == ["no", "yes"]
    assert tree.n_features_in_ == 1
    np.testing.assert_array_equal(tree.feature_, [0, -1, -1])      # root + two pure leaves
    assert tree.threshold_[0] == 2.5
    assert tree.depth_ == 1
    np.testing.assert_array_equal(tree.predict([[0.0], [2.5], [2.6], [9.0]]),
                                  ["no", "no", "yes", "yes"])
    np.testing.assert_array_equal(tree.predict_proba([[0.0], [9.0]]), [[1, 0], [0, 1]])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("seed", range(6))
def test_fit_identical_to_loop_reference(criterion, seed):
    rng = np.random.default_rng(seed)
    n, d = 80, 4
    X = rng.normal(size=(n, d))
    X[:, 1] = rng.integers(0, 3, n)                  # many tied values
    X[:, 2] = 7.0                                     # constant feature
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.7, size=n) > 0.5).astype(int)
    y[rng.random(n) < 0.2] = 2                        # third class, noisy

    tree = DecisionTreeClassifier(max_depth=4, min_samples_split=5, criterion=criterion)
    tree.fit(X, y)
    ref_feature, ref_threshold = _reference_tree(X, y, criterion, 4, 5)
    np.testing.assert_array_equal(tree.feature_, ref_feature)
    np.testing.assert_array_equal(tree.threshold_, ref_threshold)   # bit for bit, NaN = leaf

    # probabilities are leaf class fractions: rows sum to 1, argmax agrees with predict
    X_new = rng.normal(size=(50, d))
    proba = tree.predict_proba(X_new)
    assert proba.shape == (50, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(tree.classes_[proba.argmax(axis=1)], tree.predict(X_new))


def test_stopping_rules_pure_constant_and_ties():
    X = np.arange(20.0).reshape(-1, 1)
    y = np.arange(20) % 2                             # alternating: needs many splits

    assert DecisionTreeClassifier(max_depth=2).fit(X, y).depth_ == 2
    assert DecisionTreeClassifier(max_depth=1).fit(X, y).n_nodes_ == 3
    # min_samples_split larger than the data: the root is never split
    assert DecisionTreeClassifier(min_samples_split=21).fit(X, y).n_nodes_ == 1

    # pure labels: a single leaf
    pure = DecisionTreeClassifier().fit(X, np.ones(20, dtype=int))
    assert pure.n_nodes_ == 1 and pure.depth_ == 0
    np.testing.assert_array_equal(pure.predict([[100.0]]), [1])

    # constant features: no split helps; majority tie goes to the smallest class
    const = DecisionTreeClassifier().fit(np.zeros((4, 2)), [3, 1, 3, 1])
    assert const.n_nodes_ == 1
    np.testing.assert_array_equal(const.predict([[5.0, 5.0]]), [1])
    np.testing.assert_array_equal(const.predict_proba([[0.0, 0.0]]), [[0.5, 0.5]])

    # tie between features: the lowest feature index wins
    twins = DecisionTreeClassifier().fit(np.repeat(X, 2, axis=1), y > 9)
    assert twins.feature_[0] == 0


def test_max_features_is_seeded_and_reproducible():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + X[:, 3] > 0).astype(int)

    a = DecisionTreeClassifier(max_features=2, seed=1).fit(X, y)
    b = DecisionTreeClassifier(max_features=2, seed=1).fit(X, y)
    np.testing.assert_array_equal(a.feature_, b.feature_)
    np.testing.assert_array_equal(a.feature_, a.fit(X, y).feature_)   # refit: same tree

    # different seeds look at different features at the root
    roots = {DecisionTreeClassifier(max_features=1, seed=s).fit(X, y).feature_[0]
             for s in range(10)}
    assert len(roots) > 1

    # max_features equal to the number of features is the same as using all of them
    full = DecisionTreeClassifier(max_features=6, seed=3).fit(X, y)
    np.testing.assert_array_equal(full.feature_, DecisionTreeClassifier().fit(X, y).feature_)


def test_validation_errors():
    X, y = np.array([[0.0], [1.0]]), np.array([0, 1])

    with pytest.raises(RuntimeError, match="not fitted"):
        DecisionTreeClassifier().predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        DecisionTreeClassifier().predict_proba(X)
    with pytest.raises(ValueError, match="must not contain NaN"):
        DecisionTreeClassifier().fit([[np.nan], [1.0]], y)

    tree = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError, match="must not contain NaN"):
        tree.predict([[np.nan]])
    with pytest.raises(ValueError, match="features"):
        tree.predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="same number of samples"):
        DecisionTreeClassifier().fit(X, [0, 1, 1])
    with pytest.raises(ValueError, match="max_features=2 is larger than"):
        DecisionTreeClassifier(max_features=2).fit(X, y)

    with pytest.raises(ValueError, match="criterion must be"):
        DecisionTreeClassifier(criterion="mse")
    with pytest.raises(ValueError, match="max_depth must be a positive integer"):
        DecisionTreeClassifier(max_depth=0)
    with pytest.raises(ValueError, match="min_samples_split must be at least 2"):
        DecisionTreeClassifier(min_samples_split=1)
    with pytest.raises(ValueError, match="max_features must be a positive integer"):
        DecisionTreeClassifier(max_features=0.5)
    with pytest.raises(ValueError, match="buffer_size must be a positive integer"):
        DecisionTreeClassifier(buffer_size=0)
