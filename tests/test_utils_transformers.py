import numpy as np
import pytest

from scikeras.utils.transformers import ClassifierLabelEncoder, Ensure2DTransformer


class TestEnsure2DTransformer:
    @pytest.mark.parametrize(
        "X", [np.array([1, 1]), np.array([1, 1]).reshape(-1, 1), np.array([[[1], [1]]])]
    )
    def test_basic_func(self, X):
        tf = Ensure2DTransformer()
        tf.fit(X)
        X_t = tf.transform(X)
        assert len(X_t.shape) >= 2
        X_inv = tf.inverse_transform(X_t)
        assert X_inv.shape == X.shape

    @pytest.mark.parametrize(
        "X_new", [np.array([1, 1]).reshape(-1, 1), np.array([[[1], [1]]])]
    )
    def test_invalid_input_transform(self, X_new):
        # Fit with 1D then pass 2D or 3D
        X = np.random.uniform(size=100)
        tf = Ensure2DTransformer().fit(X)

        if X.ndim == 1:
            with pytest.raises(ValueError, match="Expected a 1D array"):
                tf.transform(X_new)
        else:
            tf.transform(X_new)

        # Fit with NOT 1D
        # Then no errors are raised regardless of X_new's dimensionality
        tf.fit(X.reshape(-1, 1))
        tf.transform(X_new)

    def test_almost_1d(self):
        tf = Ensure2DTransformer()
        X = np.random.uniform(size=(100, 1))
        tf.fit_transform(X)

    def test_invalid_input_inverse_transform(self):
        tf = Ensure2DTransformer()
        X = np.array([1, 1])
        # Fit with 1D then pass 1D or 3D
        # Raises ValueError
        tf.fit(X)
        with pytest.raises(ValueError, match="Expected a 2D array"):
            tf.inverse_transform(np.array([1, 1]))
        with pytest.raises(ValueError, match="Expected a 2D array"):
            tf.inverse_transform(np.array([[[1], [1]]]))
        # 2D with shape[1] != 1 raises a ValueError as well
        with pytest.raises(ValueError, match="Expected `X.shape"):
            tf.inverse_transform(np.array([1, 2, 3]).reshape(-1, 1).T)
        # Fit with NOT 1D
        # Then no errors are raised regardless of X_new's dimensionality
        tf.fit(X.reshape(-1, 1))
        tf.inverse_transform(np.array([1, 1]))
        tf.inverse_transform(np.array([[[1], [1]]]))
        # 2D with shape[1] != 1 raises an error as well
        tf.inverse_transform(np.array([1, 2, 3]).T)


class TestClassifierLabelEncoder:
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
