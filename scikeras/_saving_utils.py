import os
import zipfile

from io import BytesIO
from types import MethodType
from uuid import uuid4 as uuid

import numpy as np

from tensorflow import io as tf_io
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.saved_model import loader_impl


ram_prefix = "ram://"


def new_parse_saved_model_with_debug_info(export_dir):
    """Reads the savedmodel as well as the graph debug info.

    Args:
    export_dir: Directory containing the SavedModel and GraphDebugInfo files.

    Returns:
    `SavedModel` and `GraphDebugInfo` protocol buffers.

    Raises:
    IOError: If the saved model file does not exist, or cannot be successfully
    parsed. Missing graph debug info file is fine.
    """
    saved_model = loader_impl._parse_saved_model(export_dir)

    debug_info_path = os.path.join(
        loader_impl.saved_model_utils.get_debug_dir(export_dir),
        loader_impl.constants.DEBUG_INFO_FILENAME_PB,
    )
    debug_info = loader_impl.graph_debug_info_pb2.GraphDebugInfo()
    if loader_impl.file_io.file_exists(debug_info_path):
        with loader_impl.file_io.FileIO(debug_info_path, "rb") as debug_file:
            try:
                debug_info.ParseFromString(debug_file.read())
            except loader_impl.message.DecodeError as e:
                raise IOError("Cannot parse file %s: %s." % (debug_info_path, str(e)))
            except NotFoundError:
                # No debug info, loader_impl.file_io.file_exists is broken for ram://
                pass
    return (saved_model, debug_info)


loader_impl.parse_saved_model_with_debug_info = new_parse_saved_model_with_debug_info


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
    del self._restored_weights
    self._create_all_weights = self._create_all_weights_orig


def _restore_optimizer_weights(optimizer, weights) -> None:
    optimizer._restored_weights = weights
    optimizer._create_all_weights_orig = optimizer._create_all_weights
    optimizer._create_all_weights = MethodType(_temp_create_all_weights, optimizer)


def unpack_keras_model(packed_keras_model, optimizer_weights):
    """Reconstruct a model from the result of __reduce__
    """
    save_folder = str(uuid())
    temp_ram_location = ram_prefix + save_folder
    b = BytesIO(packed_keras_model)
    with zipfile.ZipFile(b, "r", zipfile.ZIP_DEFLATED) as zf:
        for path in zf.namelist():
            dest = temp_ram_location + os.path.sep + path
            tf_io.gfile.makedirs(os.path.dirname(dest))
            with tf_io.gfile.GFile(dest, "wb") as f:
                f.write(zf.read(path))
    model: keras.Model = load_model(temp_ram_location)
    # tf_io.gfile.rmtree(temp_ram_location)
    _restore_optimizer_weights(model.optimizer, optimizer_weights)
    return model


def pack_keras_model(model):
    """Support for Pythons's Pickle protocol.
    """
    save_folder = str(uuid())
    temp_ram_location = ram_prefix + save_folder
    model.save(temp_ram_location)
    b = BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, _, filenames in tf_io.gfile.walk(temp_ram_location):
            for filename in filenames:
                with tf_io.gfile.GFile(filename, "rb") as f:
                    zf.writestr(os.path.relpath(filename, temp_ram_location), f.read())
    b.seek(0)
    # tf_io.gfile.rmtree(temp_ram_location)
    return (
        unpack_keras_model,
        (np.asarray(memoryview(b.read())), model.optimizer.get_weights()),
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
