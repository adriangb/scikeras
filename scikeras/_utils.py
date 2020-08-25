import inspect
import os
import random
import warnings

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.keras.layers import serialize as serialize_layer
from tensorflow.keras.metrics import deserialize as deserialize_metric
from tensorflow.keras.models import Model
from tensorflow.python.keras.saving import saving_utils


class TFRandomState:
    def __init__(self, seed):
        self.seed = seed
        self._not_found = object()

    def __enter__(self):
        warnings.warn(
            "Setting the random state for TF involves "
            "irreversibly re-setting the random seed. "
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


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def _windows_upcast_ints(
    arr: Union[List[np.ndarray], np.ndarray]
) -> Union[List[np.ndarray], np.ndarray]:
    # see tensorflow/probability#886
    def _upcast(x):
        return x.astype("int64") if x.dtype == np.int32 else x

    if isinstance(arr, np.ndarray):
        return _upcast(arr)
    else:
        return [_upcast(x_) for x_ in arr]


def route_params(
    params: Dict[str, Any],
    destination: Union[str, None],
    pass_filter: Union[Iterable[str], None] = None,
) -> Dict[str, Any]:
    """Route and trim parameter names.

    Parameters
    ----------
    params : Dict[str, Any]
        Parameters to route/filter.
    destination : str, default "any"
        Destination to route to, ex: `build` or `compile`.
        If "any" all routed parameters are removed.
    pass_filter: Union[Iterable[str], None], default None
        If None, all non-routing `params` are passed. If an iterable,
        only keys from `params` that are in the iterable are passed.
        This does not affect routed parameters.

    Returns
    -------
    Dict[str, Any]
        Filtered parameters, with any routing prefixes removed.
    """
    res = {
        key: val
        for key, val in params.items()
        if pass_filter is None or key in pass_filter
    }
    for key, val in params.items():
        if (
            "__" in key
            and destination is not None
            and key.startswith(destination)
        ):
            res[key.replace(destination.strip("__") + "__", "")] = val
    return res
