from typing import Union, List, Type, Callable

import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.optimizers.Optimizer as TF_Optimizer
import tensorflow.keras.losses.Loss as TF_Loss
import tensorflow.keras.metrics.Metric as TF_Metric
import tensorflow.keras.callbacks.Callback as TF_Callback
import numpy as np


Model = Union[Callable[..., keras.Model], keras.Model]
RandomState = Union[int, np.random.RandomState]
Optimizer = Union[str, TF_Optimizer, Type[TF_Optimizer]]
Loss = Union[str, TF_Loss, Type[TF_Loss], Callable]
Metrics = Union[List[Union[str, TF_Metric, Type[TF_Metric], Callable]]]
Callbacks = Union[List[Union[TF_Callback, Type[TF_Callback]]]]
