import os
import tarfile
import tempfile

from io import BytesIO
from types import MethodType

import numpy as np

from tensorflow import io as tf_io
from tensorflow import keras
from tensorflow.keras.models import load_model


def _temp_create_all_weights(self, var_list):
    """A hack to restore weights in optimizers that use slots.

    See https://tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#slots_2
    """
    self._create_all_weights_orig(var_list)
    try:
        self.set_weights(self._restored_weights)
    except ValueError:
        # Weights don't match, eg. when optimizer was pickled before any training
        pass
    delattr(self, "_restored_weights")
    self._create_all_weights = self._create_all_weights_orig


def _restore_optimizer_weights(optimizer, weights) -> None:
    optimizer._restored_weights = weights
    optimizer._create_all_weights_orig = optimizer._create_all_weights
    optimizer._create_all_weights = MethodType(_temp_create_all_weights, optimizer)


def unpack_keras_model(packed_keras_model, optimizer_weights):
    """Reconstruct a model from the result of __reduce__
    """
    b = BytesIO(packed_keras_model)
    with tempfile.TemporaryDirectory() as tmpdirname:
        with tarfile.open(fileobj=b, mode="r:gz") as tar:
            tar.extractall(tmpdirname)
            model: keras.Model = load_model(tmpdirname)
    _restore_optimizer_weights(model.optimizer, optimizer_weights)
    return model


def pack_keras_model(model):
    """Support for Pythons's Pickle protocol.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # directory and contents have been removed
        model.save(tmpdirname)
        b = BytesIO()
        with tarfile.open(fileobj=b, mode="w:gz") as tar:
            tar.add(tmpdirname, arcname=os.path.sep)
        b.seek(0)
    model_np = np.asarray(memoryview(b.read()))
    optimizer_weights = model.optimizer.get_weights()
    return (
        unpack_keras_model,
        (model_np, optimizer_weights),
    )


def unpack_keras_optimizer(opt_serialized, weights):
    """Reconstruct optimizer.
    """
    optimizer: keras.optimizers.Optimizer = keras.optimizers.deserialize(opt_serialized)
    _restore_optimizer_weights(optimizer, weights)
    return optimizer


def pack_keras_optimizer(optimizer: keras.optimizers.Optimizer):
    """Support for Pythons's Pickle protocol in Keras Optimizers.
    """
    opt_serialized = keras.optimizers.serialize(optimizer)
    weights = optimizer.get_weights()
    return unpack_keras_optimizer, (opt_serialized, weights)


def unpack_keras_metric(metric_serialized):
    """Reconstruct metric.
    """
    metric: keras.metrics.Metric = keras.metrics.deserialize(metric_serialized)
    return metric


def pack_keras_metric(metric: keras.metrics.Metric):
    """Support for Pythons's Pickle protocol in Keras Metrics.
    """
    metric_serialized = keras.metrics.serialize(metric)
    return unpack_keras_metric, (metric_serialized,)


def unpack_keras_loss(loss_serialized):
    """Reconstruct loss.
    """
    loss: keras.losses.Loss = keras.losses.deserialize(loss_serialized)
    return loss


def pack_keras_loss(loss: keras.losses.Loss):
    """Support for Pythons's Pickle protocol in Keras Losses.
    """
    loss_serialized = keras.losses.serialize(loss)
    return unpack_keras_loss, (loss_serialized,)
