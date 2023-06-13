import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from scikeras.utils.transformers import ClassifierLabelEncoder, TargetReshaper


class TestTargetReshaper:
    @pytest.mark.parametrize(
        "X", [np.array([1, 1]), np.array([1, 1]).reshape(-1, 1), np.array([[[1], [1]]])]
    )
    def test_basic_func(self, X):
        tf = TargetReshaper()
        tf.fit(X)
        X_t = tf.transform(X)
        assert len(X_t.shape) >= 2
        X_inv = tf.inverse_transform(X_t)
        assert X_inv.shape == X.shape

    def test_almost_1d(self):
        tf = TargetReshaper()
        X = np.random.uniform(size=(100, 1))
        tf.fit_transform(X)

    @pytest.mark.parametrize(
        "X",
        [
            np.random.uniform(size=(100,)),  # 1D
            np.random.uniform(size=(100,)).reshape(-1, 1),  # 2D
        ],
    )
    def test_inverse_transform_not_fitted(self, X):
        tf = TargetReshaper()
        with pytest.raises(NotFittedError, match="is not initialized"):
            tf.inverse_transform(X)


class TestClassifierLabelEncoder:
    @pytest.mark.parametrize(
        "y, y_pred",
        [
            ([0, 1, 0], [[1, 0], [0.49, 0.51], [0.55, 0.54]]),
            ([0, 1, 0], [0, 0.51, 0.50]),
            ([0, 1, 0], [[0], [0.51], [0.50]]),
            ([0, 1, 2], [[1, 0, 0], [0.25, 0.75, 0.25], [0.33, 0.33, 0.34]]),
            (
                [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                [[0.1, 0.8, 0.1], [0.34, 0.33, 0.33], [0, 0, 1]],
            ),
            (
                [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
                [[0.50001, 0.50001, 0.50000], [0, 0.75, 0.51], [1, 0, 1]],
            ),
        ],
        ids=[
            "binary",
            "binary-1d",
            "binary-2d-singleton-dim",
            "multiclass",
            "multiclass-one-hot",
            "multilabel-indicator",
        ],
    )
    @pytest.mark.parametrize(
        "loss", ("categorical_crossentropy", "sparse_categorical_crossentropy", None)
    )
    def test_supported_tasks(self, y, y_pred, loss):
        y, y_pred = np.array(y), np.array(y_pred)
        c = ClassifierLabelEncoder(loss=loss).fit(y)
        y_inv = c.inverse_transform(y_pred)
        np.testing.assert_equal(y, y_inv)

    def test_multiclass_multioutput(self):
        c = ClassifierLabelEncoder()
        y = np.column_stack(np.array([1, 2, 3]))
        c.fit(y)
        # classes_ and n_classes_ should be undefined/None
        assert c.classes_ is None
        assert c.n_classes_ is None
        # transform works and returns input array
        y_tf = c.transform(y)
        np.testing.assert_equal(y_tf, y)
        # inverse_tf with return_proba = True should work and return
        # the input array untouched
        y_probs = np.random.random(size=(3, 6))
        y_probs_new = c.inverse_transform(y_probs, return_proba=True)
        np.testing.assert_equal(y_probs, y_probs_new)
        # inverse_tf with return_proba = False raises a NotImplementedError
        with pytest.raises(
            NotImplementedError, match="Class-predictions are not clearly"
        ):
            c.inverse_transform(y_probs)
