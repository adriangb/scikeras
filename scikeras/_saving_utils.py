import os
import tempfile
import zipfile

from io import BytesIO
from types import MethodType
from uuid import uuid4 as uuid

import numpy as np

from tensorflow import io as tf_io
from tensorflow import keras
from tensorflow.keras.models import load_model


def _get_temp_folder():
    if os.name == "nt":
        # the RAM-based filesystem is not fully supported on
        # Windows yet, we save to a temp folder on disk instead
        return tempfile.mkdtemp()
    else:
        return f"ram://{tempfile.mkdtemp()}"


def _temp_create_all_weights(self, var_list):
    """A hack to restore weights in optimizers that use slots.

    See https://tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#slots_2
    """
    self._create_all_weights_orig(var_list)
    try:
        self.set_weights(self._restored_weights)
    except ValueError:
        # Weights don't match, eg. when optimizer was pickled before any training
        # or a completely new dataset is being used right after pickling
        pass
    delattr(self, "_restored_weights")
    self._create_all_weights = self._create_all_weights_orig


def _restore_optimizer_weights(optimizer, weights) -> None:
    optimizer._restored_weights = weights
    optimizer._create_all_weights_orig = optimizer._create_all_weights
    # MethodType is used to "bind" the _temp_create_all_weights method
    # to the "live" optimizer object
    optimizer._create_all_weights = MethodType(_temp_create_all_weights, optimizer)


def unpack_keras_model(packed_keras_model, optimizer_weights):
    """Reconstruct a model from the result of __reduce__"""
    temp_dir = _get_temp_folder()
    b = BytesIO(packed_keras_model)
    with zipfile.ZipFile(b, "r", zipfile.ZIP_DEFLATED) as zf:
        for path in zf.namelist():
            dest = os.path.join(temp_dir, path)
            tf_io.gfile.makedirs(os.path.dirname(dest))
            with tf_io.gfile.GFile(dest, "wb") as f:
                f.write(zf.read(path))
    model: keras.Model = load_model(temp_dir)
    for root, _, filenames in tf_io.gfile.walk(temp_dir):
        for filename in filenames:
            if filename.startswith("ram://"):
                # Currently, tf.io.gfile.walk returns
                # the entire path for the ram:// filesystem
                dest = filename
            else:
                dest = os.path.join(root, filename)
            tf_io.gfile.remove(dest)
    _restore_optimizer_weights(model.optimizer, optimizer_weights)
    return model


def pack_keras_model(model):
    """Support for Pythons's Pickle protocol."""
    temp_dir = _get_temp_folder()
    model.save(temp_dir)
    b = BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, filenames in tf_io.gfile.walk(temp_dir):
            for filename in filenames:
                if filename.startswith("ram://"):
                    # Currently, tf.io.gfile.walk returns
                    # the entire path for the ram:// filesystem
                    dest = filename
                else:
                    dest = os.path.join(root, filename)
                with tf_io.gfile.GFile(dest, "rb") as f:
                    zf.writestr(os.path.relpath(dest, temp_dir), f.read())
                tf_io.gfile.remove(dest)
    b.seek(0)
    return (
        unpack_keras_model,
        (np.asarray(memoryview(b.read())), model.optimizer.get_weights()),
    )


def unpack_keras_optimizer(opt_serialized, weights):
    """Reconstruct optimizer."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.deserialize(opt_serialized)
    _restore_optimizer_weights(optimizer, weights)
    return optimizer


def pack_keras_optimizer(optimizer: keras.optimizers.Optimizer):
    """Support for Pythons's Pickle protocol in Keras Optimizers."""
    opt_serialized = keras.optimizers.serialize(optimizer)
    weights = optimizer.get_weights()
    return unpack_keras_optimizer, (opt_serialized, weights)


def unpack_keras_metric(metric_serialized):
    """Reconstruct metric."""
    metric: keras.metrics.Metric = keras.metrics.deserialize(metric_serialized)
    return metric


def pack_keras_metric(metric: keras.metrics.Metric):
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
