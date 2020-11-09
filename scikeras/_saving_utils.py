import os
import zipfile

from io import BytesIO
from pathlib import Path

import numpy as np

from tensorflow import io as tf_io
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.python.lib import io as tf_io
from tensorflow.python.platform import gfile


def walk(dir):
    """Recursive directory tree generator for directories.

  Similar to os.walk, but supports TF RAM based filesystems.

  Backported from TF Nightly > 2.3.1; this func is broken in <=2.3.1.
  """

    def _make_full_path(parent, item):
        # Since `os.path.join` discards paths before one that starts with the path
        # separator (https://docs.python.org/3/library/os.path.html#os.path.join),
        # we have to manually handle that case as `/` is a valid character on GCS.
        if item[0] == os.sep:
            return "".join([os.path.join(parent, ""), item])
        return os.path.join(parent, item)

    top = str(Path(dir))
    listing = tf_io.gfile.listdir(top)

    files = []
    subdirs = []
    for item in listing:
        full_path = _make_full_path(top, item)
        if tf_io.gfile.isdir(full_path):
            subdirs.append(item)
        else:
            files.append(item)

    here = (top, subdirs, files)

    yield here

    for subdir in subdirs:
        for subitem in walk(_make_full_path(top, subdir)):
            yield subitem


ram_prefix = "ram://"


def unpack_keras_model(packed_keras_model):
    """Reconstruct a model from the result of __reduce__
    """
    save_folder = f"tmp/saving/{id(packed_keras_model)}"
    temp_ram_location = os.path.join(ram_prefix, save_folder)
    b = BytesIO(packed_keras_model)
    with zipfile.ZipFile(b, "r", zipfile.ZIP_DEFLATED) as zf:
        for path in zf.namelist():
            if not tf_io.file_io.file_exists_v2(
                os.path.dirname(os.path.join(temp_ram_location, path))
            ):
                tf_io.file_io.recursive_create_dir_v2(
                    os.path.dirname(os.path.join(temp_ram_location, path))
                )
            with gfile.GFile(os.path.join(temp_ram_location, path), "wb+") as f:
                f.write(zf.read(path))
    return load_model(temp_ram_location)


def pack_keras_model(model):
    """Support for Pythons's Pickle protocol.
    """
    save_folder = f"tmp/saving/{id(model)}"
    temp_ram_location = os.path.join(ram_prefix, save_folder)
    model.save(temp_ram_location)
    b = BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, _, filenames in walk(temp_ram_location):
            for filename in filenames:
                with gfile.GFile(filename, "rb") as f:
                    zf.writestr(os.path.relpath(filename, temp_ram_location), f.read())
    b.seek(0)
    return unpack_keras_model, (np.asarray(memoryview(b.read())),)


def unpack_keras_optimizer(opt_serialized):
    """Reconstruct optimizer.
    """
    optimizer: keras.optimizers.Optimizer = keras.optimizers.deserialize(opt_serialized)
    return optimizer


def pack_keras_optimizer(optimizer: keras.optimizers.Optimizer):
    """Support for Pythons's Pickle protocol in Keras Optimizers.
    """
    opt_serialized = keras.optimizers.serialize(optimizer)
    return unpack_keras_optimizer, (opt_serialized,)


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
