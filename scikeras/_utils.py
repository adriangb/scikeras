import inspect

from types import FunctionType
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Type, Union

from tensorflow.keras import losses as losses_mod
from tensorflow.keras import metrics as metrics_mod
from tensorflow.keras import optimizers as optimizers_mod


DIGITS = frozenset(str(i) for i in range(10))


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
    strict: bool
        Only include routed parameters target at `destination__...`
        and not any further routing (i.e. exclude `destination__inner__...`).

    Returns
    -------
    Dict[str, Any]
        Filtered parameters, with any routing prefixes removed.
    """
    res = dict()
    routed = {k: v for k, v in params.items() if "__" in k}
    non_routed = {k: params[k] for k in (params.keys() - routed.keys())}
    for key, val in non_routed.items():
        if pass_filter is None or key in pass_filter:
            res[key] = val
    for key, val in routed.items():
        prefix = destination + "__"
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            if strict and "__" in new_key:
                continue
            res[new_key] = val
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
    if inspect.isclass(items):
        item = items
        new_base_params = {p: v for p, v in params.items() if "__" not in p}
        base_params = base_params or dict()
        args_and_kwargs = {**base_params, **new_base_params}
        for p, v in args_and_kwargs.items():
            args_and_kwargs[p] = unflatten_params(
                items=v,
                params=route_params(
                    params=params, destination=f"{p}", pass_filter=set(), strict=False,
                ),
            )
        kwargs = {k: v for k, v in args_and_kwargs.items() if k[0] not in DIGITS}
        args = [(int(k), v) for k, v in args_and_kwargs.items() if k not in kwargs]
        args = (v for _, v in sorted(args))  # sorts by key / arg num
        return item(*args, **kwargs)
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


def get_optimizer_class(
    optimizer: Union[str, optimizers_mod.Optimizer, Type[optimizers_mod.Optimizer]]
) -> optimizers_mod.Optimizer:
    return type(
        optimizers_mod.get(optimizer)
    )  # optimizers.get returns instances instead of classes


def get_metric_class(
    metric: Union[str, metrics_mod.Metric, Type[metrics_mod.Metric]]
) -> Union[metrics_mod.Metric, str]:
    if metric in ("acc", "accuracy", "ce", "crossentropy"):
        # Keras matches "acc" and others in this list to the right function
        # based on the Model's loss function, output shape, etc.
        # We pass them through here to let Keras deal with these.
        return metric
    return metrics_mod.get(metric)  # always returns a class


def get_loss_class_function_or_string(loss: str) -> Union[losses_mod.Loss, Callable]:
    got = losses_mod.get(loss)
    if type(got) == FunctionType:
        return got
    return type(got)  # a class, e.g. if loss="BinaryCrossentropy"


def try_to_convert_strings_to_classes(
    items: Union[str, dict, tuple, list], class_getter: Callable
):
    """Convert shorthand optimizer/loss/metric names to classes.
    """
    if isinstance(items, str):
        return class_getter(items)  # single item, despite parameter name
    elif isinstance(items, Sequence):
        return type(items)(
            [try_to_convert_strings_to_classes(item, class_getter) for item in items]
        )
    elif isinstance(items, Mapping):
        return {
            k: try_to_convert_strings_to_classes(item, class_getter)
            for k, item in items.items()
        }
    else:
        return items  # not a string or known collection
