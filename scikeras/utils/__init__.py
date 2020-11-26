from inspect import isclass
from typing import Callable, Union

from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import get as keras_loss_get
from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import get as keras_metric_get


def _camel2snake(s: str) -> str:
    """from [1]
    [1]:https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    return "".join(["_" + c.lower() if c.isupper() else c for c in s]).lstrip("_")


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
        String name of the loss.

    Notes
    -----
    The result of this function will always be in snake case, not camel case.

    Examples
    --------
    >>> loss_name("BinaryCrossentropy")
    'binary_crossentropy'
    >>> loss_name("binary_crossentropy")
    'binary_crossentropy'
    >>> import tensorflow.keras.losses as losses
    >>> loss_name(losses.BinaryCrossentropy)
    'binary_crossentropy'
    >>> loss_name(losses.binary_crossentropy)
    'binary_crossentropy'

    Raises
    ------
    TypeError
        If loss is not a string, tf.keras.losses.Loss instance or a callable.
    """
    if isclass(loss):
        loss = loss()
    if not (isinstance(loss, (str, Loss)) or callable(loss)):
        raise TypeError(
            "``loss`` must be a string, a function, an instance of ``tf.keras.losses.Loss``"
            " or a type inheriting from ``tf.keras.losses.Loss``"
        )
    fn_or_cls = keras_loss_get(loss)
    if isinstance(fn_or_cls, Loss):
        return _camel2snake(fn_or_cls.__class__.__name__)
    return fn_or_cls.__name__


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

    Notes
    -----
    The result of this function will always be in snake case, not camel case.

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
    TypeError
        If metric is not a string, a tf.keras.metrics.Metric instance a class
        inheriting from tf.keras.metrics.Metric.
    """
    if isclass(metric):
        metric = metric()  # get_metric accepts instances, not classes
    if not (isinstance(metric, (str, Metric)) or callable(metric)):
        raise TypeError(
            "``metric`` must be a string, a function, an instance of"
            " ``tf.keras.metrics.Metric`` or a type inheriting from"
            " ``tf.keras.metrics.Metric``"
        )
    fn_or_cls = keras_metric_get(metric)
    if isinstance(fn_or_cls, Metric):
        return _camel2snake(fn_or_cls.__class__.__name__)
    return fn_or_cls.__name__
