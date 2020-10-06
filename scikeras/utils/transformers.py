import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class Ensure2DTransformer(TransformerMixin, BaseEstimator):
    """Transforms from 1D -> 2D and back.
    """

    def fit(self, X: np.ndarray) -> BaseEstimator:
        self.should_transform_ = X.ndim == 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.should_transform_:
            if not (X.ndim == 1 or (np.prod(X.shape) == X.size)):
                raise ValueError(
                    "Expected a 1D array for `X` "
                    "or for X to have one meaningful dimension (e.g, "
                    "`X.shape == (1, 400, 1)`). Instead, got a"
                    f" {X.ndim}D array instead with shape={X.shape}."
                )
            X = X.reshape(-1, 1)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.should_transform_:
            if X.ndim != 2:
                raise ValueError(
                    f"Expected a 2D array of shape (n_samples, 1)"
                    " for `X`, but got a"
                    f" {X.ndim}D array instead."
                )
            if X.shape[1] != 1:
                raise ValueError(
                    f"Expected `X.shape[1] == 1`"
                    f" but got `X.shape[1] == {X.shape}` instead."
                )
            X = np.squeeze(X, axis=1)
        return X
