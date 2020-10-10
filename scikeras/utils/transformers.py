from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from tensorflow.python.keras.losses import is_categorical_crossentropy


class Ensure2DTransformer(TransformerMixin, BaseEstimator):
    """Transforms from 1D -> 2D and back.
    """

    def fit(self, X: np.ndarray) -> "Ensure2DTransformer":
        self.should_transform_ = X.ndim == 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.should_transform_:
            if X.ndim != 1:
                raise ValueError(
                    "Expected a 1D array for `X`." f" Got a {X.ndim}D array instead."
                )
            X = X.reshape(-1, 1)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.should_transform_:
            if X.ndim != 2:
                raise ValueError(
                    f"Expected a 2D array for `X`, but got a"
                    f" {X.ndim}D array instead."
                )
            if X.shape[1] != 1:
                raise ValueError(
                    f"Expected `X.shape[1] == 1`"
                    f" but got `X.shape[1] == {X.shape}` instead."
                )
            X = np.squeeze(X, axis=1)
        return X


class BaseKerasTransformer(TransformerMixin, BaseEstimator, ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseKerasTransformer":
        """Fit this transformer using `X`.
        """

    @abstractmethod
    def transform(
        self, X: np.ndarray
    ) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        """Convert input numpy array to the format expected by the
        Keras Model this is being used with.

        Parameters
        ----------
        X : np.ndarray
            Numpy array.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
            Numpy array, list of arrays or dict of arrays mapping
            to the Model's outputs.
        """

    @abstractmethod
    def inverse_transform(self, X: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """Invert the transformation.

        Parameters
        ----------
        X : Union[List[np.ndarray], np.ndarray]
            Output from Keras Model.

        Returns
        -------
        np.ndarray
            Numpy array.
        """

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Retrieve the meta parameters of this transformer.

        Returns
        -------
        Dict[str, Any]
            Dictionary of of format parameter: value.
        """


class ClassifierLabelEncoder(BaseKerasTransformer):
    """Default target transformer for KerasClassifier.
    """

    def __init__(self, target_type="unknown", loss=None):
        self.target_type = target_type
        self.loss = loss

    def fit(self, X: np.ndarray) -> "ClassifierLabelEncoder":
        y = X  # rename for clarity, the input is always expected to be a target `y`
        encoders = {
            "binary": make_pipeline(
                Ensure2DTransformer(), OrdinalEncoder(dtype=np.float32),
            ),
            "multiclass": make_pipeline(
                Ensure2DTransformer(), OrdinalEncoder(dtype=np.float32),
            ),
            "multiclass-one-hot": FunctionTransformer(),
        }
        if is_categorical_crossentropy(self.loss):
            encoders["multiclass"] = make_pipeline(
                Ensure2DTransformer(), OneHotEncoder(sparse=False, dtype=np.float32),
            )
        if self.target_type not in encoders:
            raise ValueError(
                f"Unknown label type: {self.target_type}."
                "\n\nTo implement support, subclass KerasClassifier and override"
                " `target_transformer` with a transformer that supports this"
                " label type."
            )
        self.final_encoder_ = encoders[self.target_type].fit(y)
        if self.target_type in ["binary", "multiclass"]:
            self.classes_ = self.final_encoder_[1].categories_[0]
            self.n_classes_ = self.classes_.size
        elif self.target_type == "multiclass-one-hot":
            self.classes_ = np.arange(0, y.shape[1])
            self.n_classes_ = y.shape[1]
        self.n_outputs_ = 1
        self.model_n_outputs_ = 1
        self.y_dtype_ = y.dtype
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        y = X  # rename for clarity, the input is always expected to be a target `y`
        # no need to validate n_outputs_ or model_n_outputs_, those are hardcoded
        # self.classes_ and self.n_classes_ are validated by the transformers themselves
        return self.final_encoder_.transform(y)

    def inverse_transform(
        self, X: np.ndarray, return_proba: bool = False
    ) -> np.ndarray:
        y = X  # rename for clarity, the input is always expected to be a target `y`
        if self.target_type == "binary":
            # array([0.9, 0.1], [.2, .8]) -> array(['yes', 'no'])
            if self.n_classes_ == 1:
                # special case: single input label for sigmoid output
                # may give more predicted classes than inputs for
                # small sample sizes!
                # don't even bother inverse transforming, just fill.
                class_predictions = np.full(
                    shape=(y.shape[0], 1), fill_value=self.classes_[0]
                )
            else:
                if y.shape == 1 or (y.shape[1] == 1 and self.n_classes_ == 2):
                    # result from a single sigmoid output
                    # reformat so that we have 2 columns
                    y = np.column_stack([1 - y, y])
                y_ = np.argmax(y, axis=1).reshape(-1, 1)
                class_predictions = self.final_encoder_.inverse_transform(y_)
        elif self.target_type == "multiclass":
            # array([0.8, 0.1, 0.1], [.1, .8, .1]) ->
            # array(['apple', 'orange'])
            idx = np.argmax(y, axis=-1)
            if not is_categorical_crossentropy(self.loss):
                y_ = idx.reshape(-1, 1)
            else:
                y_ = np.zeros(y.shape, dtype=int)
                y_[np.arange(y.shape[0]), idx] = 1
            class_predictions = self.final_encoder_.inverse_transform(y_)
        else:  # "multiclass-one-hot"
            # array([0.8, 0.1, 0.1], [.1, .8, .1]) ->
            # array([[1, 0, 0], [0, 1, 0]])
            idx = np.argmax(y, axis=-1)
            y_ = np.zeros(y.shape, dtype=int)
            y_[np.arange(y.shape[0]), idx] = 1
            class_predictions = y_

        if return_proba:
            return y
        return np.squeeze(np.column_stack(class_predictions)).astype(
            self.y_dtype_, copy=False
        )

    def get_metadata(self):
        return {
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
            "n_outputs_": self.n_outputs_,
            "model_n_outputs_": self.model_n_outputs_,
        }


class RegressorTargetEncoder(BaseKerasTransformer):
    """Default target transformer for KerasRegressor.
    """

    def fit(self, X: np.ndarray) -> "RegressorTargetEncoder":
        y = X  # rename for clarity, the input is always expected to be a target `y`
        self.y_dtype_ = y.dtype
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        self.model_n_outputs_ = 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        y = X
        n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        if n_outputs_ != self.n_outputs_:
            raise ValueError(
                f"Detected `y` to have {n_outputs_},"
                f" but this {self} expects"
                f" {self.n_outputs_} for `y`."
            )
        return y

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        y = X  # rename for clarity, the input is always expected to be a target `y`
        if np.can_cast(self.y_dtype_, np.float32):
            return np.squeeze(y.astype(np.float32, copy=False))
        else:
            return np.squeeze(y.astype(np.float64, copy=False))

    def get_metadata(self):
        return {
            "n_outputs_": self.n_outputs_,
            "model_n_outputs_": self.model_n_outputs_,
        }
