from typing import Any, Dict, List, Union

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.utils.multiclass import type_of_target
from tensorflow.keras.losses import (
    CategoricalCrossentropy,
    Loss,
    categorical_crossentropy,
)


def _is_categorical_crossentropy(loss):
    return (
        isinstance(loss, CategoricalCrossentropy)
        or loss == categorical_crossentropy
        or getattr(loss, "__name__", None) == "categorical_crossentropy"
        or loss in ("categorical_crossentropy", "cce", "CategoricalCrossentropy")
    )


class TargetReshaper(BaseEstimator, TransformerMixin):
    """Convert 1D targets to 2D and back.

    For use in pipelines with transformers that only accept
    2D inputs, like OneHotEncoder and OrdinalEncoder.

    Attributes
    ----------
    ndim_ : int
        Dimensions of y that the transformer was trained on.
    """

    def fit(self, y: np.ndarray) -> "TargetReshaper":
        """Fit the transformer to a target y.

        Returns
        -------
        TargetReshaper
            A reference to the current instance of TargetReshaper.
        """
        self.ndim_ = y.ndim
        return self

    @staticmethod
    def transform(y: np.ndarray) -> np.ndarray:
        """Makes 1D y 2D.

        Parameters
        ----------
        y : np.ndarray
            Target y to be transformed.

        Returns
        -------
        np.ndarray
            A numpy array, of dimension at least 2.
        """
        if y.ndim == 1:
            return y.reshape(-1, 1)
        return y

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Revert the transformation of transform.

        Parameters
        ----------
        y : np.ndarray
            Transformed numpy array.

        Returns
        -------
        np.ndarray
            If the transformer was fit to a 1D numpy array,
            and a 2D numpy array with a singleton second dimension
            is passed, it will be squeezed back to 1D. Otherwise, it
            will eb left untouched.
        """
        if not hasattr(self, "ndim_"):
            raise NotFittedError(
                f"This {self.__class__.__name__} is not initialized."
                " You must call ``fit`` before using ``inverse_transform``."
            )
        if self.ndim_ == 1 and y.ndim == 2:
            return np.squeeze(y, axis=1)
        return y


class ClassifierLabelEncoder(BaseEstimator, TransformerMixin):
    """Default target transformer for KerasClassifier.

    Parameters
    ----------
    loss : Union[None, str, Loss], default None
        Keras Model's loss function. Used to automatically
        one-hot encode the target if the loss function is
        categorical crossentropy.
    categories : Union[str, List[np.ndarray]], default "auto"
        All of the categories present in the target for the entire
        dataset. "auto" will infer the categories from the
        data passed to fit.

    Attributes
    ----------
    classes_ : Iterable
        The classes seen during fit.
    n_classes_ : int
        The number of classes seen during fit.
    n_outputs_ : int
        Dimensions of y that the transformer was trained on.
    n_outputs_expected_ : int
        Number of outputs the Keras Model is expected to have.
    """

    def __init__(
        self,
        loss: Union[None, str, Loss] = None,
        categories: Union[str, List[np.ndarray]] = "auto",
    ):
        self.loss = loss
        self.categories = categories

    def _type_of_target(self, y: np.ndarray) -> str:
        """Determine the type of target accounting for the self.categories param."""
        target_type = type_of_target(y)
        if target_type == "binary" and self.categories != "auto":
            # check that this is not a multiclass problem missing categories
            # if not "auto", categories is expected to be a list with a single np.ndarray
            target_type = type_of_target(self.categories[0])
        return target_type

    def fit(self, y: np.ndarray) -> "ClassifierLabelEncoder":
        """Fit the estimator to the target y.

        For all targets, this transforms classes into ordinal numbers.
        If the loss function is categorical_crossentropy, the target
        will be one-hot encoded.

        Parameters
        ----------
        y : np.ndarray
            The target data to be transformed.

        Returns
        -------
        ClassifierLabelEncoder
            A reference to the current instance of ClassifierLabelEncoder.
        """
        target_type = self._type_of_target(y)
        keras_dtype = np.dtype(tf.keras.backend.floatx())
        self._y_shape = y.shape
        encoders = {
            "binary": make_pipeline(
                TargetReshaper(),
                OrdinalEncoder(dtype=keras_dtype, categories=self.categories),
            ),
            "multiclass": make_pipeline(
                TargetReshaper(),
                OrdinalEncoder(dtype=keras_dtype, categories=self.categories),
            ),
            "multiclass-multioutput": FunctionTransformer(),
            "multilabel-indicator": FunctionTransformer(),
        }
        if _is_categorical_crossentropy(self.loss):
            encoders["multiclass"] = make_pipeline(
                TargetReshaper(),
                OneHotEncoder(
                    sparse=False, dtype=keras_dtype, categories=self.categories
                ),
            )
        if target_type not in encoders:
            raise ValueError(
                f"Unknown label type: {target_type}."
                "\n\nTo implement support, subclass KerasClassifier and override"
                " ``target_encoder`` with a transformer that supports this"
                " label type."
                "\n\nFor information on sklearn target types, see:"
                " * https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html"
                " * https://scikit-learn.org/stable/modules/multiclass.html"
                "\n\nFor information on the SciKeras data transformation interface, see:"
                " * https://www.adriangb.com/scikeras/stable/advanced.html#data-transformers"
            )
        self._final_encoder = encoders[target_type].fit(y)

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
            self.classes_ = self._final_encoder[1].categories_[0]
            self.n_classes_ = self.classes_.size
        elif target_type in ("multiclass-onehot", "multilabel-indicator"):
            self.classes_ = np.arange(0, y.shape[1])
            self.n_classes_ = y.shape[1]
        elif target_type == "multiclass-multioutput":
            self.classes_ = None
            self.n_classes_ = None

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform the target y to the format expected by the Keras Model.

        If the loss function is categorical_crossentropy, the target
        will be one-hot encoded.
        For other types of target, this transforms classes into ordinal numbers.

        Returns
        -------
        np.ndarray
            Transformed target.
        """
        # no need to validate n_outputs_ or n_outputs_expected_, those are hardcoded
        # self.classes_ and self.n_classes_ are validated by the transformers themselves
        return self._final_encoder.transform(y)

    def inverse_transform(
        self, y: np.ndarray, return_proba: bool = False
    ) -> np.ndarray:
        """Restore the data types, shape and classes of the input y
        to the output of the Keras Model.

        Parameters
        ----------
        y : np.ndarray
            Raw probability predictions from the Keras Model.
        return_proba : bool, default False
            If True, return the prediction probabilites themselves.
            If False, return the class predictions.

        Returns
        -------
        np.ndarray
            Class predictions (of the same shape as the y to fit/transform), \
            or class prediction probabilities.
        """
        if self._target_type == "binary":
            # array([0.9, 0.1], [.2, .8]) -> array(['yes', 'no'])
            if y.ndim == 1 or (y.shape[1] == 1 and self.n_classes_ == 2):
                # result from a single sigmoid output
                # reformat so that we have 2 columns
                y = np.column_stack([1 - y, y])
            class_predictions = np.argmax(y, axis=1).reshape(-1, 1)
            class_predictions = self._final_encoder.inverse_transform(class_predictions)
        elif self._target_type == "multiclass":
            # array([0.8, 0.1, 0.1], [.1, .8, .1]) -> array(['apple', 'orange'])
            idx = np.argmax(y, axis=-1)
            if not _is_categorical_crossentropy(self.loss):
                class_predictions = idx.reshape(-1, 1)
            else:
                class_predictions = np.zeros(y.shape, dtype=int)
                class_predictions[np.arange(len(idx)), idx] = 1
            class_predictions = self._final_encoder.inverse_transform(class_predictions)
        elif self._target_type == "multiclass-onehot":
            # array([.8, .1, .1], [.1, .8, .1]) -> array([[1, 0, 0], [0, 1, 0]])
            idx = np.argmax(y, axis=-1)
            class_predictions = np.zeros(y.shape, dtype=int)
            class_predictions[np.arange(len(idx)), idx] = 1
        elif self._target_type == "multilabel-indicator":
            class_predictions = np.around(y).astype(int, copy=False)
        else:
            if not return_proba:
                raise NotImplementedError(
                    "Class-predictions are not clearly defined for"
                    " 'multiclass-multioutput' target types."
                    "\n\nTo implement support, subclass KerasClassifier and override"
                    " ``target_encoder`` with a transformer that supports this"
                    " label type."
                    "\n\nFor information on sklearn target types, see:"
                    " * https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html"
                    " * https://scikit-learn.org/stable/modules/multiclass.html"
                    "\n\nFor information on the SciKeras data transformation interface, see:"
                    " * https://www.adriangb.com/scikeras/stable/advanced.html#data-transformers"
                )

        if return_proba:
            return np.squeeze(y)
        return class_predictions.reshape(-1, *self._y_shape[1:])

    def get_metadata(self) -> Dict[str, Any]:
        """Returns a dictionary of meta-parameters generated when this transfromer
        was fitted.

        Used by SciKeras to bind these parameters to the SciKeras estimator itself
        and make them available as inputs to the Keras model.

        Returns
        -------
        Dict[str, Any]
            Dictionary of meta-parameters generated when this transfromer
            was fitted.
        """
        return {
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
            "n_outputs_": self.n_outputs_,
            "n_outputs_expected_": self.n_outputs_expected_,
        }


class RegressorTargetEncoder(BaseEstimator, TransformerMixin):
    """Default target transformer for KerasRegressor.

    Attributes
    ----------
    n_outputs_ : int
        Dimensions of y that the transformer was trained on.
    n_outputs_expected_ : int
        Number of outputs the Keras Model is expected to have.
    """

    def fit(self, y: np.ndarray) -> "RegressorTargetEncoder":
        """Fit the transformer to the target y.

        For RegressorTargetEncoder, this just records the dimensions
        of y as the expected number of outputs and saves the dtype.

        Returns
        -------
        RegressorTargetEncoder
            A reference to the current instance of RegressorTargetEncoder.
        """
        self._y_dtype = y.dtype
        self._y_shape = y.shape
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        self.n_outputs_expected_ = 1
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform the target y to the format expected by the Keras Model.

        For RegressorTargetEncoder, this simply checks that the shape passed to
        fit matches the shape passed to transform.

        Returns
        -------
        np.ndarray
            Untouched input y.
        """
        n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        if n_outputs_ != self.n_outputs_:
            raise ValueError(
                f"Detected ``y`` to have {n_outputs_} outputs"
                f" with ``y.shape = {y.shape}``",
                f" but this {self.__class__.__name__} has"
                f" {self.n_outputs_} outputs.",
            )
        return y

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Restore the data types and shape of the input y
        to the output of the Keras Model.

        Parameters
        ----------
        y : np.ndarray
            Raw predictions from the Keras Model.

        Returns
        -------
        np.ndarray
            Keras Model predictions cast to the dtype and shape of the input
            targets.
        """
        y = y.reshape(-1, *self._y_shape[1:])
        return y

    def get_metadata(self):
        """Returns a dictionary of meta-parameters generated when this transfromer
        was fitted.

        Used by SciKeras to bind these parameters to the SciKeras estimator itself
        and make them available as inputs to the Keras model.

        Returns
        -------
        Dict[str, Any]
            Dictionary of meta-parameters generated when this transfromer
            was fitted.
        """
        return {
            "n_outputs_": self.n_outputs_,
            "n_outputs_expected_": self.n_outputs_expected_,
        }
