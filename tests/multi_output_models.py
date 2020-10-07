from typing import Any, Dict

import numpy as np

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from tensorflow.python.keras.losses import is_categorical_crossentropy

from scikeras.utils import Ensure2DTransformer, KerasClassifierTargetTransformer
from scikeras.wrappers import BaseWrapper, KerasClassifier


class MultiOutputKerasClassifierTargetTransformer(KerasClassifierTargetTransformer):
    def fit(self, X: np.ndarray) -> "KerasClassifierTargetTransformer":
        y = X  # rename for clarity, the input is always expected to be a target `y`

        if self.target_type not in ("multilabel-indicator", "multiclass-multioutput"):
            return super().fit(y)

        self.n_outputs_ = y.shape[1]
        self.model_n_outputs_ = y.shape[1]
        self.y_dtype_ = y.dtype

        y = np.split(y, y.shape[1], axis=1)
        if self.target_type == "multilabel-indicator":
            # y = array([1, 1, 1, 0], [0, 0, 1, 1])
            # each col will be processed as multiple binary classifications
            self.final_encoder_ = [FunctionTransformer().fit(y_) for y_ in y]
            self.classes_ = [np.array([0, 1])] * len(y)
        else:  # multiclass-multioutput
            # y = array([1, 0, 5], [2, 1, 3])
            # each col be processed as a separate multiclass problem
            if is_categorical_crossentropy(self.loss):
                encoder = make_pipeline(
                    Ensure2DTransformer(), OneHotEncoder(sparse=False),
                )
            else:
                encoder = make_pipeline(
                    Ensure2DTransformer(), OrdinalEncoder(dtype=np.float32)
                )
            self.final_encoder_ = [clone(encoder).fit(y_) for y_ in y]
            self.classes_ = [
                encoder[1].categories_[0] for encoder in self.final_encoder_
            ]

        self.n_classes_ = [c.size for c in self.classes_]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.target_type not in ("multilabel-indicator", "multiclass-multioutput"):
            return super().transform(X)
        y = X  # rename for clarity, the input is always expected to be a target `y`
        # TODO: validate self.classes_ and self.n_classes_, n_outputs_, model_n_outputs_
        y = np.split(y, y.shape[1], axis=1)
        return [encoder.transform(y_) for encoder, y_ in zip(self.final_encoder_, y)]

    def inverse_transform(
        self, X: np.ndarray, return_proba: bool = False
    ) -> np.ndarray:
        y = X  # rename for clarity, the input is always expected to be a target `y`

        if self.target_type not in ("multilabel-indicator", "multiclass-multioutput"):
            return super().inverse_transform(X, return_proba=return_proba)

        class_predictions = []

        for i in range(self.n_outputs_):
            if self.target_type == "multilabel-indicator":
                class_predictions.append(np.argmax(y[i], axis=1))
            else:  # multiclass-multioutput
                # array([0.8, 0.1, 0.1], [.1, .8, .1]) ->
                # array(['apple', 'orange'])
                idx = np.argmax(y[i], axis=-1)
                if not is_categorical_crossentropy(self.loss):
                    y_ = idx.reshape(-1, 1)
                else:
                    y_ = np.zeros(y[i].shape, dtype=int)
                    y_[np.arange(y[i].shape[0]), idx] = 1
                class_predictions.append(self.target_encoder_[i].inverse_transform(y_))

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
    def target_encoder(self) -> MultiOutputKerasClassifierTargetTransformer:
        return MultiOutputKerasClassifierTargetTransformer(
            loss=self.loss, target_type=self.target_type_
        )
