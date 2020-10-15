from typing import List

import numpy as np

from sklearn.utils.multiclass import type_of_target

from scikeras.utils.transformers import ClassifierLabelEncoder
from scikeras.wrappers import KerasClassifier


class MultiLabelTransformer(ClassifierLabelEncoder):
    def fit(self, y: np.ndarray) -> "MultiLabelTransformer":
        self._target_type = type_of_target(y)
        if self._target_type != "multilabel-indicator":
            return super().fit(y)
        # y = array([1, 1, 1, 0], [0, 0, 1, 1])
        # each col will be processed as multiple binary classifications
        self.n_outputs_ = self.n_outputs_expected_ = y.shape[1]
        self.y_dtype_ = y.dtype
        self.classes_ = [np.array([0, 1])] * y.shape[1]
        self.n_classes_ = [2] * y.shape[1]
        return self

    def transform(self, y: np.ndarray) -> List[np.ndarray]:
        if self._target_type != "multilabel-indicator":
            return super().transform(y)
        return np.split(y, y.shape[1], axis=1)

    def inverse_transform(
        self, y: List[np.ndarray], return_proba: bool = False
    ) -> np.ndarray:
        if self._target_type != "multilabel-indicator":
            return super().inverse_transform(y, return_proba=return_proba)
        if not return_proba:
            y = [np.argmax(y_, axis=1).astype(self.y_dtype_, copy=False) for y_ in y]
        return np.squeeze(np.column_stack(y))


class MultiOutputClassifier(KerasClassifier):
    """Extend KerasClassifier with the ability to process
    "multilabel-indicator" by mapping to multiple Keras outputs.
    """

    @property
    def target_encoder(self) -> MultiLabelTransformer:
        return MultiLabelTransformer()

    def score(self, X, y):
        """Taken from sklearn.multiouput.MultiOutputClassifier
        """
        if self.target_type_ != "multilabel-indicator":
            return super().score(X, y)
        return np.mean(np.all(y == self.predict(X), axis=1))
