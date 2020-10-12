from typing import Union

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.utils.multiclass import type_of_target
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.losses import is_categorical_crossentropy


class TargetReshaper(BaseEstimator, TransformerMixin):
    """Convert 1D targets to 2D and back.

    For use in pipelines with transformers that only accept
    2D inputs, like OneHotEncoder and OrdinalEncoder.

    Attributes
    ----------
    ndim_ : int
        Dimensions of `X` that the transformer was trained on.
    """

    def fit(self, X):
        self.ndim_ = X.ndim
        return self

    @staticmethod
    def transform(X):
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    def inverse_transform(self, X):
        if not hasattr(self, "ndim_"):
            raise NotFittedError(
                f"This {self.__class__.__name__} is not initialized."
                " You must call `fit` before using `inverse_transform`."
            )
        if self.ndim_ == 1 and X.ndim == 2:
            return np.squeeze(X, axis=1)
        return X


class ClassifierLabelEncoder(BaseEstimator, TransformerMixin):
    """Default target transformer for KerasClassifier.
    """

    def __init__(self, loss: Union[None, str, Loss] = None):
        self.loss = loss

    def fit(self, X: np.ndarray) -> "ClassifierLabelEncoder":
        y = X  # rename for clarity, the input is always expected to be a target `y`
        target_type = type_of_target(y)
        keras_dtype = np.dtype(tf.keras.backend.floatx())
        encoders = {
            "binary": make_pipeline(
                TargetReshaper(), OrdinalEncoder(dtype=keras_dtype),
            ),
            "multiclass": make_pipeline(
                TargetReshaper(), OrdinalEncoder(dtype=keras_dtype),
            ),
            "multiclass-multioutput": FunctionTransformer(),
            "multilabel-indicator": FunctionTransformer(),
        }
        if is_categorical_crossentropy(self.loss):
            encoders["multiclass"] = make_pipeline(
                TargetReshaper(), OneHotEncoder(sparse=False, dtype=keras_dtype),
            )
        if target_type not in encoders:
            raise ValueError(
                f"Unknown label type: {target_type}."
                "\n\nTo implement support, subclass KerasClassifier and override"
                " `target_transformer` with a transformer that supports this"
                " label type."
                "\n\nFor information on sklearn target types, see:"
                " * https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html"
                " * https://scikit-learn.org/stable/modules/multiclass.html"
                "\n\nFor information on the SciKeras data transformation interface, see:"
                " * https://scikeras.readthedocs.io/en/latest/advanced.html#data-transforms"
            )
        self.final_encoder_ = encoders[target_type].fit(y)

        if (
            target_type == "multilabel-indicator"
            and y.min() == 0
            and (y.sum(axis=1) == 1).all()
        ):
            target_type = "multiclass-onehot"

        self.n_outputs_ = 1
        self.n_outputs_expected_ = 1
        self._y_dtype = y.dtype
        self._target_type = target_type

        if target_type in ("binary", "multiclass"):
            self.classes_ = self.final_encoder_[1].categories_[0]
            self.n_classes_ = self.classes_.size
        elif target_type in ("multiclass-onehot", "multilabel-indicator"):
            self.classes_ = np.arange(0, y.shape[1])
            self.n_classes_ = y.shape[1]
        elif target_type == "multiclass-multioutput":
            self.classes_ = None
            self.n_classes_ = None

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        y = X  # rename for clarity, the input is always expected to be a target `y`
        # no need to validate n_outputs_ or n_outputs_expected_, those are hardcoded
        # self.classes_ and self.n_classes_ are validated by the transformers themselves
        return self.final_encoder_.transform(y)

    def inverse_transform(
        self, X: np.ndarray, return_proba: bool = False
    ) -> np.ndarray:
        y = X  # rename for clarity, the input is always expected to be a target `y`
        if self._target_type == "binary":
            # array([0.9, 0.1], [.2, .8]) -> array(['yes', 'no'])
            if y.ndim == 1 or (y.shape[1] == 1 and self.n_classes_ == 2):
                # result from a single sigmoid output
                # reformat so that we have 2 columns
                y = np.column_stack([1 - y, y])
            class_predictions = np.argmax(y, axis=1).reshape(-1, 1)
            class_predictions = self.final_encoder_.inverse_transform(class_predictions)
        elif self._target_type == "multiclass":
            # array([0.8, 0.1, 0.1], [.1, .8, .1]) ->
            # array(['apple', 'orange'])
            idx = np.argmax(y, axis=-1)
            if not is_categorical_crossentropy(self.loss):
                class_predictions = idx.reshape(-1, 1)
            else:
                class_predictions = np.zeros(y.shape, dtype=int)
                class_predictions[:, idx] = 1
            class_predictions = self.final_encoder_.inverse_transform(class_predictions)
        elif self._target_type == "multiclass-onehot":
            # array([.8, .1, .1], [.1, .8, .1]) ->
            # array([[1, 0, 0], [0, 1, 0]])
            idx = np.argmax(y, axis=-1)
            class_predictions = np.zeros(y.shape, dtype=int)
            class_predictions[:, idx] = 1
        elif self._target_type == "multilabel-indicator":
            class_predictions = np.around(y)
        else:
            if not return_proba:
                raise NotImplementedError(
                    f"Class-predictions are not clearly defined for"
                    " 'multiclass-multioutput' target types."
                    "\n\nTo implement support, subclass KerasClassifier and override"
                    " `target_transformer` with a transformer that supports this"
                    " label type."
                    "\n\nFor information on sklearn target types, see:"
                    " * https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html"
                    " * https://scikit-learn.org/stable/modules/multiclass.html"
                    "\n\nFor information on the SciKeras data transformation interface, see:"
                    " * https://scikeras.readthedocs.io/en/latest/advanced.html#data-transforms"
                )

        if return_proba:
            return y
        return np.squeeze(np.column_stack(class_predictions)).astype(
            self._y_dtype, copy=False
        )

    def get_metadata(self):
        return {
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
            "n_outputs_": self.n_outputs_,
            "n_outputs_expected_": self.n_outputs_expected_,
        }


class RegressorTargetEncoder(BaseEstimator, TransformerMixin):
    """Default target transformer for KerasRegressor.
    """

    def fit(self, X: np.ndarray) -> "RegressorTargetEncoder":
        y = X  # rename for clarity, the input is always expected to be a target `y`
        self._y_dtype = y.dtype
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        self.n_outputs_expected_ = 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        y = X
        n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        if n_outputs_ != self.n_outputs_:
            raise ValueError(
                f"Detected `y` to map to {n_outputs_} outputs"
                f" with `y.shape = {y.shape}",
                f" but this {self.__class__.__name__} has"
                f" {self.n_outputs_} outputs.",
            )
        return y

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        y = X  # rename for clarity, the input is always expected to be a target `y`
        if self._y_dtype == np.float64 and y.dtype == np.float32:
            return np.squeeze(y.astype(np.float64, copy=False))
        return np.squeeze(y)

    def get_metadata(self):
        return {
            "n_outputs_": self.n_outputs_,
            "n_outputs_expected_": self.n_outputs_expected_,
        }
