from io import BytesIO
from typing import Any, Callable, Dict, Hashable, List, Tuple

import keras as keras
import keras.saving
import numpy as np
from keras.src.saving.saving_lib import load_model, save_model


def unpack_keras_model(
    packed_keras_model: np.ndarray,
):
    """Reconstruct a model from the result of __reduce__"""
    b = BytesIO(packed_keras_model)
    return load_model(b, compile=True)


def pack_keras_model(
    model: keras.Model,
) -> Tuple[
    Callable[[np.ndarray, List[np.ndarray]], keras.Model],
    Tuple[np.ndarray, List[np.ndarray]],
]:
    """Support for Pythons's Pickle protocol."""
    tp = type(model)
    out = BytesIO()
    if tp not in keras.saving.get_custom_objects():
        module = ".".join(tp.__qualname__.split(".")[:-1])
        name = tp.__qualname__.split(".")[-1]
        keras.saving.register_keras_serializable(module, name)(tp)
    save_model(model, out)
    model_bytes = np.asarray(memoryview(out.getvalue()))
    return (unpack_keras_model, (model_bytes,))


def deepcopy_model(model: keras.Model, memo: Dict[Hashable, Any]) -> keras.Model:
    _, (model_bytes,) = pack_keras_model(model)
    new_model = unpack_keras_model(model_bytes)
    memo[model] = new_model
    return new_model


def unpack_keras_optimizer(
    opt_serialized: Dict[str, Any]
) -> keras.optimizers.Optimizer:
    """Reconstruct optimizer."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.deserialize(opt_serialized)
    return optimizer


def pack_keras_optimizer(
    optimizer: keras.optimizers.Optimizer,
) -> Tuple[
    Callable[
        [Dict[str, Any], keras.optimizers.Optimizer],
        Tuple[Dict[str, Any], List[np.ndarray]],
    ],
]:
    """Support for Pythons's Pickle protocol in Keras Optimizers."""
    opt_serialized = keras.optimizers.serialize(optimizer)
    return unpack_keras_optimizer, (opt_serialized,)


def unpack_keras_metric(metric_serialized: Dict[str, Any]) -> keras.metrics.Metric:
    """Reconstruct metric."""
    metric: keras.metrics.Metric = keras.metrics.deserialize(metric_serialized)
    return metric


def pack_keras_metric(
    metric: keras.metrics.Metric,
) -> Tuple[Callable[[Dict[str, Any], keras.metrics.Metric], Tuple[Dict[str, Any]]]]:
    """Support for Pythons's Pickle protocol in Keras Metrics."""
    metric_serialized = keras.metrics.serialize(metric)
    return unpack_keras_metric, (metric_serialized,)


def unpack_keras_loss(loss_serialized):
    """Reconstruct loss."""
    loss: keras.losses.Loss = keras.losses.deserialize(loss_serialized)
    return loss


def pack_keras_loss(loss: keras.losses.Loss):
    """Support for Pythons's Pickle protocol in Keras Losses."""
    loss_serialized = keras.losses.serialize(loss)
    return unpack_keras_loss, (loss_serialized,)
