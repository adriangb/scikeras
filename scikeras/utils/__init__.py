from inspect import isclass
from typing import Callable, Union

import numpy as np

from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import get as get_loss
from tensorflow.keras.losses import serialize as serialize_loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import get as get_metric
from tensorflow.keras.metrics import serialize as serialize_metric


def loss_name(loss: Union[str, Loss, Callable]) -> str:
    """Retrieves a loss's full name (eg: "mean_squared_error").

    Parameters
    ----------
    loss : Union[str, Loss, Callable]
        Instance of Keras Loss, loss callable or string
        shorthand (eg: "mse") or full name ("mean_squared_error").

    Returns
    -------
    str
        [description]

    Examples
    --------
    >>> loss_name("BinaryCrossentropy")
    'BinaryCrossentropy'
    >>> loss_name("binary_crossentropy")
    'binary_crossentropy'
    >>> import tensorflow.keras.losses as losses
    >>> loss_name(losses.BinaryCrossentropy)
    'BinaryCrossentropy'
    >>> loss_name(losses.binary_crossentropy)
    'binary_crossentropy'

    Raises
    ------
    ValueError
        In case of an unknown loss.
    """
    if isclass(loss):
        loss = loss()  # get_loss accepts instances, not classes
    try:
        loss = serialize_loss(get_loss(loss))
    except ValueError:
        # Error messages change slightly across TF versions
        # And errors are different for unknown strings vs. unknown objects
        # We homogenize them to a single error message
        raise ValueError(f"Unable to determine name for loss: {loss}")
    if isinstance(loss, dict):
        # classes are serialized as dicts
        return loss["class_name"]
    return loss  # for functions (serialize returns a string)


def metric_name(metric: Union[str, Metric, Callable]) -> str:
    """Retrieves a metric's full name (eg: "mean_squared_error").

    Parameters
    ----------
    metrics : Union[str, Metric, Callable]
        Instance of Keras Metric, metric callable or string
        shorthand (eg: "mse") or full name ("mean_squared_error").

    Returns
    -------
    str
        Full name for Keras metric. Ex: "mean_squared_error".

    Examples
    --------
    >>> metric_name("BinaryCrossentropy")
    'BinaryCrossentropy'
    >>> metric_name("binary_crossentropy")
    'binary_crossentropy'
    >>> import tensorflow.keras.metrics as metrics
    >>> metric_name(metrics.BinaryCrossentropy)
    'BinaryCrossentropy'
    >>> metric_name(metrics.binary_crossentropy)
    'binary_crossentropy'

    Raises
    ------
    ValueError
        In case of an unknown metric.
    """
    if isclass(metric):
        metric = metric()  # get_metric accepts instances, not classes
    try:
        metric = serialize_metric(get_metric(metric))
    except ValueError:
        # Error messages change slightly across TF versions
        # And errors are different for unknown strings vs. unknown objects
        # We homogenize them to a single error message
        raise ValueError(f"Unable to determine name for metric: {metric}")
    if isinstance(metric, dict):
        # classes are serialized as dicts
        return metric["class_name"]
    return metric  # for functions (serialize returns a string)
