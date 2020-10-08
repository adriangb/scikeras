from typing import List

import numpy as np

from scikeras.utils.transformers import KerasClassifierTargetTransformer
from scikeras.wrappers import KerasClassifier


class MultiLabelTransformer(KerasClassifierTargetTransformer):
    def fit(self, X: np.ndarray) -> "MultiLabelTransformer":
        if self.target_type != "multilabel-indicator":
            return super().fit(X)
        # y = array([1, 1, 1, 0], [0, 0, 1, 1])
        # each col will be processed as multiple binary classifications
        self.n_outputs_ = self.model_n_outputs_ = X.shape[1]
        self.y_dtype_ = X.dtype
        self.classes_ = [np.array([0, 1])] * X.shape[1]
        self.n_classes_ = [c.size for c in self.classes_]
        return self

    def transform(self, X: np.ndarray) -> List[np.ndarray]:
        if self.target_type != "multilabel-indicator":
            return super().transform(X)
        return np.split(X, X.shape[1], axis=1)

    def inverse_transform(
        self, X: List[np.ndarray], return_proba: bool = False
    ) -> np.ndarray:
        if self.target_type != "multilabel-indicator":
            return super().inverse_transform(X, return_proba=return_proba)
        if not return_proba:
            X = [np.argmax(X_, axis=1).astype(self.y_dtype_, copy=False) for X_ in X]
        return np.squeeze(np.column_stack(X))


class MultiOutputClassifier(KerasClassifier):
    """Extend KerasClassifier with the ability to process
    "multilabel-indicator" and "multiclass-multioutput"
    by mapping them to multiple Keras outputs.
    """

    @property
    def target_encoder(self) -> MultiLabelTransformer:
        return MultiLabelTransformer(target_type=self.target_type_)

    def score(self, X, y):
        """Taken from sklearn.multiouput.MultiOutputClassifier
        """
        if self.target_type_ != "multilabel-indicator":
            return super().score(X, y)
        return np.mean(np.all(y == self.predict(X), axis=1))
