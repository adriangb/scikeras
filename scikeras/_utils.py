import warnings
import random
import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras.layers import (
    deserialize as deserialize_layer,
    serialize as serialize_layer,
)
from tensorflow.keras.models import Model
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.metrics import deserialize as deserialize_metric


class TFRandomState:
    def __init__(self, seed):
        self.seed = seed
        self._not_found = object()

    def __enter__(self):
        warnings.warn(
            "SciKeras: Setting the random state for TF involves"
            " irreversibly re-setting the random seed. "
            "This may have unintended side effects."
        )

        # Save values
        self.origin_hashseed = os.environ.get(
            "PYTHONHASHSEED", self._not_found
        )
        self.origin_gpu_det = os.environ.get(
            "TF_DETERMINISTIC_OPS", self._not_found
        )
        self.orig_random_state = random.getstate()
        self.orig_np_random_state = np.random.get_state()

        # Set values
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __exit__(self, type, value, traceback):
        if self.origin_hashseed is not self._not_found:
            os.environ["PYTHONHASHSEED"] = self.origin_hashseed
        else:
            os.environ.pop("PYTHONHASHSEED")
        if self.origin_gpu_det is not self._not_found:
            os.environ["TF_DETERMINISTIC_OPS"] = self.origin_gpu_det
        else:
            os.environ.pop("TF_DETERMINISTIC_OPS")
        random.setstate(self.orig_random_state)
        np.random.set_state(self.orig_np_random_state)
        tf.random.set_seed(None)  # TODO: can we revert instead of unset?


class LabelDimensionTransformer(TransformerMixin, BaseEstimator):
    """Transforms from 1D -> 2D and back.

    Used when applying LabelTransformer -> OneHotEncoder.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return X

    def inverse_transform(self, X):
        if X.shape[1] == 1:
            X = np.squeeze(X, axis=1)
        return X


def unpack_keras_model(model, training_config, weights):
    """Creates a new Keras model object using the input
    parameters.

    Returns
    -------
    Model
        A copy of the input Keras Model,
        compiled if the original was compiled.
    """
    restored_model = deserialize_layer(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(training_config)
        )
    restored_model.set_weights(weights)
    restored_model.__reduce_ex__ = pack_keras_model.__get__(restored_model)
    return restored_model


def pack_keras_model(model_obj, protocol):
    """Pickle a Keras Model.

    Arguments:
        model_obj: an instance of a Keras Model.
        protocol: pickle protocol version, ignored.

    Returns
    -------
    Pickled model
        A tuple following the pickle protocol.
    """

    if not isinstance(model_obj, Model):
        raise TypeError("`model_obj` must be an instance of a Keras Model")
    # pack up model
    model_metadata = saving_utils.model_metadata(model_obj)
    training_config = model_metadata.get("training_config", None)
    model = serialize_layer(model_obj)
    weights = model_obj.get_weights()
    return (unpack_keras_model, (model, training_config, weights))


def make_model_picklable(model_obj):
    """Makes a Keras Model object picklable without cloning.

    Arguments:
        model_obj: an instance of a Keras Model.

    Returns
    -------
    Model
        The input model, but directly picklable.
    """
    if not isinstance(model_obj, Model):
        raise TypeError("`model_obj` must be an instance of a Keras Model")
    model_obj.__reduce_ex__ = pack_keras_model.__get__(model_obj)


def get_metric_full_name(name: str) -> str:
    """Get aliases for Keras losses and metrics.

    See: https://github.com/tensorflow/tensorflow/pull/42097

    Parameters
    ----------
    name : str
        Full name or shorthand for Keras metric. Ex: "mse".

    Returns
    -------
    str
        Full name for Keras metric. Ex: "mean_squared_error".
    """
    # deserialize returns the actual function, then get it's name
    # to keep a single consistent name for the metric
    if name == "loss":
        # may be passed "loss" from thre training history
        return name
    return getattr(deserialize_metric(name), "__name__")
