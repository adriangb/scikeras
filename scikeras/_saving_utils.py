import os
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from io import BytesIO
from typing import Any, Callable, Dict, Hashable, Iterator, List, Tuple
from uuid import uuid4

import numpy as np
import tensorflow.keras as keras
from tensorflow import io as tf_io
from tensorflow.keras.models import load_model


@contextmanager
def _get_temp_folder() -> Iterator[str]:
    if os.name == "nt":
        # the RAM-based filesystem is not fully supported on
        # Windows yet, we save to a temp folder on disk instead
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        temp_dir = f"ram://{uuid4().hex}"
        try:
            yield temp_dir
        finally:
            for root, _, filenames in tf_io.gfile.walk(temp_dir):
                for filename in filenames:
                    dest = os.path.join(root, filename)
                    tf_io.gfile.remove(dest)


def unpack_keras_model(
    packed_keras_model: np.ndarray,
):
    """Reconstruct a model from the result of __reduce__"""
    with _get_temp_folder() as temp_dir:
        b = BytesIO(packed_keras_model)
        with tarfile.open(fileobj=b, mode="r") as archive:
            for fname in archive.getnames():
                dest = os.path.join(temp_dir, fname)
                tf_io.gfile.makedirs(os.path.dirname(dest))
                with tf_io.gfile.GFile(dest, "wb") as f:
                    f.write(archive.extractfile(fname).read())
        model: keras.Model = load_model(temp_dir)
        model.load_weights(temp_dir)
        model.optimizer.build(model.trainable_variables)
        return model


def pack_keras_model(
    model: keras.Model,
) -> Tuple[
    Callable[[np.ndarray, List[np.ndarray]], keras.Model],
    Tuple[np.ndarray, List[np.ndarray]],
]:
    """Support for Pythons's Pickle protocol."""
    with _get_temp_folder() as temp_dir:
        model.save(temp_dir)
        b = BytesIO()
        with tarfile.open(fileobj=b, mode="w") as archive:
            for root, _, filenames in tf_io.gfile.walk(temp_dir):
                for filename in filenames:
                    dest = os.path.join(root, filename)
                    with tf_io.gfile.GFile(dest, "rb") as f:
                        info = tarfile.TarInfo(name=os.path.relpath(dest, temp_dir))
                        info.size = f.size()
                        archive.addfile(tarinfo=info, fileobj=f)
                    tf_io.gfile.remove(dest)
        b.seek(0)
        model_bytes = np.asarray(memoryview(b.read()))
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
