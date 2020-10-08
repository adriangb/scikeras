from typing import List

import numpy as np

from sklearn.preprocessing import FunctionTransformer

from scikeras.utils.transformers import KerasClassifierTargetTransformer
from scikeras.wrappers import KerasClassifier


class MultiLabelTransformer(KerasClassifierTargetTransformer):
    def fit(self, X: np.ndarray) -> "KerasClassifierTargetTransformer":
        y = X  # rename for clarity, the input is always expected to be a target `y`

        if self.target_type != "multilabel-indicator":
            return super().fit(y)

        # y = array([1, 1, 1, 0], [0, 0, 1, 1])
        # each col will be processed as multiple binary classifications
        self.n_outputs_ = y.shape[1]
        self.model_n_outputs_ = y.shape[1]
        self.y_dtype_ = y.dtype
        y = np.split(y, y.shape[1], axis=1)
        self.final_encoder_ = [FunctionTransformer().fit(y_) for y_ in y]
        self.classes_ = [np.array([0, 1])] * len(y)
        self.n_classes_ = [c.size for c in self.classes_]
        return self

    def transform(self, X: np.ndarray) -> List[np.ndarray]:
        if self.target_type != "multilabel-indicator":
            return super().transform(X)
        y = X  # rename for clarity, the input is always expected to be a target `y`
        y = np.split(y, y.shape[1], axis=1)
        return [encoder.transform(y_) for encoder, y_ in zip(self.final_encoder_, y)]

    def inverse_transform(
        self, X: List[np.ndarray], return_proba: bool = False
    ) -> np.ndarray:
        y = X  # rename for clarity, the input is always expected to be a target `y`

        if self.target_type != "multilabel-indicator":
            return super().inverse_transform(y, return_proba=return_proba)

        class_predictions = [np.argmax(y_, axis=1) for y_ in y]

        if return_proba:
            return np.squeeze(np.column_stack(y))
        else:
            return np.squeeze(np.column_stack(class_predictions)).astype(
                self.y_dtype_, copy=False
            )


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
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))
