import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class Ensure2DTransformer(TransformerMixin, BaseEstimator):
    """Transforms from 1D -> 2D and back.
    """

    def fit(self, X: np.ndarray) -> None:
        if len(X.shape) == 1:
            self.should_transform_ = True
        else:
            self.should_transform_ = False
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.should_transform_:
            if len(X.shape) != 1:
                raise ValueError(
                    "Expected a 1D array for `X`, got a"
                    f" {len(X.shape)}D array instead."
                )
            X = X.reshape(-1, 1)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.should_transform_:
            if len(X.shape) != 2:
                raise ValueError(
                    f"Expected a 2D array of shape (n_samples, 1)"
                    " for `X`, but got a"
                    f" {len(X.shape)}D array instead."
                )
            if X.shape[1] != 1:
                raise ValueError(
                    f"Expected `X.shape[1] == 1`"
                    f" but got `X.shape[1] == {X.shape}` instead."
                )
            X = np.squeeze(X, axis=1)
        return X
