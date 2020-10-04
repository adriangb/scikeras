from inspect import isclass
from typing import Callable, Type, Union

import numpy as np

from sklearn import metrics
from sklearn.utils.multiclass import type_of_target as _type_of_target
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
        try:
            loss = loss()  # get_loss accepts instances, not classes
        except TypeError:
            # convert TypeError -> ValueError for consistency with errors
            # raised by get_loss
            raise ValueError(f"Unknown loss: {loss}")
    try:
        loss = serialize_loss(get_loss(loss))
    except ValueError:
        # Error messages are different for metrics or losses; homogenize them
        raise ValueError(f"Unknown loss: {loss}")
    if isinstance(loss, dict):
        # for classes
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
        try:
            metric = metric()  # get_metric accepts instances, not classes
        except TypeError:
            # convert TypeError -> ValueError for consistency with errors
            # raised by get_metric
            raise ValueError(f"Unknown metric: {metric}")
    try:
        metric = serialize_metric(get_metric(metric))
    except ValueError:
        # Error messages are different for metrics or losses; homogenize them
        raise ValueError(f"Unknown metric: {metric}")
    if isinstance(metric, dict):
        # for classes
        return metric["class_name"]
    return metric  # for functions (serialize returns a string)


def type_of_target(y):
    """Identical to sklearn.utils.multiclass.type_of_target,
    but recognizes "multiclass-one-hot" as a sub-category of
    "multilabel-indicator" when each sample has only one
    class assigned.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.
        * ``multiclass-one-hot`` is more specific but compatible with
          ``multilabel-indicator``.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    target_type : string
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'multiclass-one-hot': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values where each sample has only 1 class assigned.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    Examples
    --------
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multilabel-indicator'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    >>> type_of_target(np.array([[0, 1], [1, 0]]))
    'multiclass-one-hot'
    """
    type_ = _type_of_target(y)
    if type_ == "multilabel-indicator":
        if len(y.shape) == 2 and np.all(np.sum(y, axis=1) == 1):
            # one-hot encoded target
            # this is a subset of multilabel-indicator
            # but SciKeras differentiates it since it is
            # a common use case in Keras to have a single multiclass
            # target one-hot encoded + categorical_crossentropy as a loss
            type_ = "multiclass-one-hot"
    return type_
