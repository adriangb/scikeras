import inspect
import os
import random
import warnings

from inspect import isclass
from typing import Any, Callable, Dict, Iterable, Union

import numpy as np
import tensorflow as tf


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
        self.origin_hashseed = os.environ.get("PYTHONHASHSEED", self._not_found)
        self.origin_gpu_det = os.environ.get("TF_DETERMINISTIC_OPS", self._not_found)
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
    """Check if func has a parameter named param.

    Parameters
    ----------
    func : Callable
        Function to inspect.
    param : str
        Parameter name.

    Returns
    -------
    bool
        True if the parameter is part of func's signature,
        False otherwise.
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


def unflatten_params(items, params, base_params=None):
    """Recursively compile nested structures of classes
    using parameters from params.
    """
    if isclass(items):
        item = items
        new_base_params = {p: v for p, v in params.items() if "__" not in p}
        base_params = base_params or dict()
        kwargs = {**base_params, **new_base_params}
        for p, v in kwargs.items():
            kwargs[p] = unflatten_params(
                items=v,
                params=route_params(
                    params=params, destination=f"{p}", pass_filter=set(), strict=False,
                ),
            )
        return item(**kwargs)
    if isinstance(items, (list, tuple)):
        iter_type_ = type(items)
        res = list()
        new_base_params = {p: v for p, v in params.items() if "__" not in p}
        for idx, item in enumerate(items):
            item_params = route_params(
                params=params, destination=f"{idx}", pass_filter=set(), strict=False,
            )
            res.append(
                unflatten_params(
                    items=item, params=item_params, base_params=new_base_params
                )
            )
        return iter_type_(res)
    if isinstance(items, (dict,)):
        res = dict()
        new_base_params = {p: v for p, v in params.items() if "__" not in p}
        for key, item in items.items():
            item_params = route_params(
                params=params, destination=f"{key}", pass_filter=set(), strict=False,
            )
            res[key] = unflatten_params(
                items=item, params=item_params, base_params=new_base_params,
            )
        return res
    # non-compilable item, check if it has any routed parameters
    item = items
    new_base_params = {p: v for p, v in params.items() if "__" not in p}
    base_params = base_params or dict()
    kwargs = {**base_params, **new_base_params}
    if kwargs:
        raise TypeError(
            f'TypeError: "{str(item)}" object of type "{type(item)}"'
            "does not accept parameters because it's not a class."
            f' However, it received parameters "{kwargs}"'
        )
    return item


def _class_from_strings(items: Union[str, dict, tuple, list], class_getter: Callable):
    """Convert shorthand optimizer/loss/metric names to classes.
    """
    if isinstance(items, str):
        item = items
        if class_getter is tf.keras.metrics.get and item in (
            "acc",
            "accuracy",
            "ce",
            "crossentropy",
        ):
            # Keras matches "acc" and others in this list to the right function
            # based on the Model's loss function, output shape, etc.
            # We pass them through here to let Keras deal with these.
            return item
        got = class_getter(item)
        if hasattr(got, "__class__") and type(got).__module__.startswith("tensorflow"):
            # optimizers.get returns instances instead of classes
            got = got.__class__
        return got
    elif isinstance(items, (list, tuple)):
        return type(items)([_class_from_strings(item, class_getter) for item in items])
    elif isinstance(items, dict):
        return {k: _class_from_strings(item, class_getter) for k, item in items.items()}
    else:
        return items
