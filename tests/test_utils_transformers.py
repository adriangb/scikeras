import numpy as np
import pytest

from scikeras.utils import Ensure2DTransformer


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

        if X.ndim == 2:
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
