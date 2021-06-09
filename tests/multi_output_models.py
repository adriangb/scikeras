from typing import List

import numpy as np

from sklearn.utils.multiclass import type_of_target
from tensorflow.keras.backend import floatx as tf_floatx

from scikeras.utils.transformers import ClassifierLabelEncoder
from scikeras.wrappers import KerasClassifier


class MultiLabelTransformer(ClassifierLabelEncoder):
    def __init__(
        self, split: bool = True,
    ):
        super().__init__()
        self.split = split

    def fit(self, y: np.ndarray) -> "MultiLabelTransformer":
        self._target_type = type_of_target(y)
        if self._target_type not in ("multilabel-indicator", "multiclass-multioutput"):
            return super().fit(y)
        # y = array([1, 1, 1, 0], [0, 0, 1, 1])
        # each col will be processed as multiple binary classifications
        self.n_outputs_ = y.shape[1]
        self.n_outputs_expected_ = None if not self.split else self.n_outputs_
        self._y_dtype = y.dtype
        self.classes_ = [np.array([0, 1])] * y.shape[1]
        self.n_classes_ = [2] * y.shape[1]
        return self

    def transform(self, y: np.ndarray) -> List[np.ndarray]:
        if self._target_type not in ("multilabel-indicator", "multiclass-multioutput"):
            return super().transform(y)
        y = y.astype(tf_floatx())
        if self.split:
            return np.split(y, y.shape[1], axis=1)
        return y

    def inverse_transform(
        self, y: List[np.ndarray], return_proba: bool = False
    ) -> np.ndarray:
        if self._target_type not in ("multilabel-indicator", "multiclass-multioutput"):
            return super().inverse_transform(y, return_proba=return_proba)
        if return_proba:
            return y
        if self._target_type == "multilabel-indicator":
            if self.split:
                y = np.column_stack(y)
            # RandomForestClassifier and sklearn's MultiOutputClassifier always return int64
            # for multilabel-indicator
            y = np.around(y).astype(int, copy=False)
        else:  # mutlitclass-multioutput
            if self.split:
                y = np.column_stack(
                    [
                        np.argmax(y_, axis=1).astype(self._y_dtype, copy=False)
                        for y_ in y
                    ]
                )
            else:
                raise NotImplementedError(
                    "multiclass-multioutput must be handled by a multi-output Model"
                )
        return y


class MultiOutputClassifier(KerasClassifier):
    """Extend KerasClassifier with the ability to process
    "multilabel-indicator" by mapping to multiple Keras outputs.
    """

    def __init__(self, model=None, loss=None, split: bool = True, **kwargs):
        super().__init__(model=model, loss=loss, **kwargs)
        self.split = split

    @property
    def target_encoder(self) -> MultiLabelTransformer:
        return MultiLabelTransformer(split=self.split)

    def score(self, X, y):
        """Taken from sklearn.multiouput.MultiOutputClassifier
        """
        if self.target_type_ != "multilabel-indicator":
            return super().score(X, y)
        return np.mean(np.all(y == self.predict(X), axis=1))
