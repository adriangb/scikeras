import inspect
import os
import random
import warnings

from inspect import isclass
from typing import Any, Callable, Dict, Iterable, List, Type, Union

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module
from tensorflow.keras import optimizers as optimizers_module
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
        # may be passed "loss" from training history
        return name
    return getattr(deserialize_metric(name), "__name__")


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
    destination: str,
    pass_filter: Union[None, Iterable[str]],
    strict: bool = False,
) -> Dict[str, Any]:
    """Route and trim parameter names.

    Parameters
    ----------
    params : Dict[str, Any]
        Parameters to route/filter.
    destination : str
        Destination to route to, ex: `build` or `compile`.
    pass_filter: Iterable[str]
        Only keys from `params` that are in the iterable are passed.
        This does not affect routed parameters.

    Returns
    -------
    Dict[str, Any]
        Filtered parameters, with any routing prefixes removed.
    """
    res = dict()
    for key, val in params.items():
        if "__" in key:
            # routed param
            if key.startswith(destination + "__"):
                new_key = key[len(destination + "__") :]
                res[new_key] = val
        else:
            # non routed
            if pass_filter is None or key in pass_filter:
                res[key] = val
    if strict:
        res = {k: v for k, v in res.items() if "__" not in k}
    return res


def has_param(func: Callable, param: str) -> bool:
    """[summary]

    Parameters
    ----------
    func : Callable
        [description]
    param : str
        [description]

    Returns
    -------
    bool
        [description]
    """
    return any(
        p.name == param
        for p in inspect.signature(func).parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    )


def accepts_kwargs(func: Callable) -> bool:
    """Check if ``func`` accepts kwargs.
    """
    return any(
        True
        for param in inspect.signature(func).parameters.values()
        if param.kind == param.VAR_KEYWORD
    )


def compile_with_params(items, params, base_params=None):
    """Recursively compile nested structures of classes
    using parameters from params.
    """
    if isclass(items):
        item = items
        new_base_params = {p: v for p, v in params.items() if "__" not in p}
        base_params = base_params or dict()
        kwargs = {**base_params, **new_base_params}
        for p, v in kwargs.items():
            kwargs[p] = compile_with_params(
                items=v,
                params=route_params(
                    params=params,
                    destination=f"{p}",
                    pass_filter=set(),
                    strict=False,
                ),
            )
        return item(**kwargs)
    if isinstance(items, (list, tuple)):
        iter_type_ = type(items)
        res = list()
        new_base_params = {p: v for p, v in params.items() if "__" not in p}
        for idx, item in enumerate(items):
            item_params = route_params(
                params=params,
                destination=f"{idx}",
                pass_filter=set(),
                strict=False,
            )
            for p, v in item_params.items():
                item_params[p] = compile_with_params(
                    items=v,
                    params=route_params(
                        params={
                            k: v_ for k, v_ in item_params.items() if p != k
                        },
                        destination=f"{p}",
                        pass_filter=set(),
                        strict=False,
                    ),
                )
            res.append(
                compile_with_params(
                    items=item, params=item_params, base_params=new_base_params
                )
            )
        return iter_type_(res)
    if isinstance(items, (dict,)):
        res = dict()
        new_base_params = {p: v for p, v in params.items() if "__" not in p}
        for key, item in items.items():
            item_params = route_params(
                params=params,
                destination=f"{key}",
                pass_filter=set(),
                strict=False,
            )
            for p, v in item_params.items():
                item_params[p] = compile_with_params(
                    items=v,
                    params=route_params(
                        params={
                            k: v_ for k, v_ in item_params.items() if p != k
                        },
                        destination=f"{p}",
                        pass_filter=set(),
                        strict=False,
                    ),
                )
            res[key] = compile_with_params(
                items=item, params=item_params, base_params=new_base_params,
            )
        return res
    # non-compilable item, check if it has any routed parameters
    item = items
    new_base_params = {p: v for p, v in params.items() if "__" not in p}
    base_params = base_params or dict()
    kwargs = {**base_params, **new_base_params}
    if kwargs:
        warnings.warn(
            message=f"SciKeras does not know how to compile {item}"
            f" but {item} was passed the following parameters:"
            f" {kwargs}.",
            category=UserWarning,
        )
    return item


def _class_from_strings(items, item_types: str):
    """Convert shorthand optimizer/loss/metric names to classes.
    """
    if isinstance(items, str):
        item = items
        try:
            if item_types == "optimizer":
                got = optimizers_module.get(item)
                if (
                    hasattr(got, "__class__")
                    and type(got).__module__ != "builtins"
                ):
                    return got.__class__
            if item_types == "loss":
                got = losses_module.get(item)
                if (
                    hasattr(got, "__class__")
                    and type(got).__module__ != "builtins"
                ):
                    return got.__class__
                else:
                    return got
            if item_types == "metrics":
                got = metrics_module.get(item)
                if (
                    hasattr(got, "__class__")
                    and type(got).__module__ != "builtins"
                ):
                    return got.__class__
                else:
                    return got
        except ValueError:
            # string not found
            return item
    elif isinstance(items, (list, tuple)):
        return type(items)(
            [_class_from_strings(item, item_types) for item in items]
        )
    elif isinstance(items, dict):
        return {
            k: _class_from_strings(item, item_types)
            for k, item in items.items()
        }
    else:
        return items
