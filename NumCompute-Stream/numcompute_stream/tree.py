"""Decision tree implemented from scratch with streaming support.

``DecisionTreeClassifier``: depth-limited, Gini or entropy-based.

- ``.fit(X, y)``: trains the tree from scratch on the given data
- ``.partial_fit(X_chunk, y_chunk)`` supports streaming via buffered retraining:

  - Each call appends the chunk to an internal data buffer
  - Once the buffer holds at least ``buffer_size`` samples, the tree is
    retrained on the buffered data and the buffer is cleared
  - Between retrains, ``.predict()`` uses the most recently trained tree

- Support config: ``max_depth``, ``min_samples_split``, ``max_features``,
  ``buffer_size``

Example
-------
>>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
>>> for X_chunk, y_chunk in stream:
...     tree.partial_fit(X_chunk, y_chunk)

See ``classification-tree.html`` (Canvas) for details on building the tree.
"""

from .pipeline import Estimator


class DecisionTreeClassifier(Estimator):
    """
    Classification tree with numeric threshold questions ``X[:, f] <= t``.

    Same conventions as the course tree (classification-tree.html): the gain is the
    impurity before minus the size-weighted impurity after; a split must have gain > 0;
    ties go to the lowest feature index, then the lowest threshold; a leaf predicts the
    majority class, a tie going to the smallest class. Thresholds are midpoints between
    neighbouring distinct values.

    Parameters:
    ----------
    - max_depth: int >= 1 or None, maximum number of splits on any path (None = until
    pure or no split helps; default 5)
    - min_samples_split: int >= 2, a node with fewer samples is a leaf (default 2)
    - max_features: int >= 1 or None, number of features tried at each node, chosen at
    random with ``seed`` (None = all; default None)
    - buffer_size: int >= 1, samples collected by partial_fit before retraining
    (default 500)
    - criterion: "gini" or "entropy" (default "gini")
    - seed: int, seed for the max_features choice (default 0)

    Attributes (after fit):
    ----------
    - classes_: np.ndarray of shape (n_classes,), sorted labels seen in training
    - n_features_in_: int
    - feature_, threshold_, left_, right_: np.ndarray of shape (n_nodes,), the tree in
    preorder; a leaf has feature -1, threshold NaN and children -1
    - value_: np.ndarray of shape (n_nodes, n_classes), training class counts per node
    - n_nodes_, depth_: int

    Raises:
    ------
    - ValueError: for invalid hyperparameters (at construction)
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 buffer_size=500, criterion="gini", seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.criterion = criterion
        self.seed = seed

    def fit(self, X, y):
        """
        Build the tree from scratch on X, y.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), numeric, no NaN or infinity
        - y: array-like of shape (n_samples,), labels (numbers or strings)

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X or y is invalid, or max_features > n_features
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """
        Class probabilities: the training class fractions of the leaf each row reaches.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features_in_)

        Returns:
        -------
        - np.ndarray of shape (n_samples, n_classes), columns in classes_ order

        Raises:
        ------
        - RuntimeError: if the tree is not fitted yet
        - ValueError: if X is invalid or has the wrong number of features
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predicted labels: the majority class of the leaf each row reaches.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features_in_)

        Returns:
        -------
        - np.ndarray of shape (n_samples,), values from classes_

        Raises:
        ------
        - RuntimeError: if the tree is not fitted yet
        - ValueError: if X is invalid or has the wrong number of features
        """
        raise NotImplementedError
