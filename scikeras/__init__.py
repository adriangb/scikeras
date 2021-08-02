"""Top-level package for Scikit-Learn Wrapper for Keras."""

__author__ = """Adrian Garcia Badaracco"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata  # python <3.8

__version__ = importlib_metadata.version("scikeras")


MIN_TF_VERSION = "2.4.0"
TF_VERSION_ERR = f"SciKeras requires TensorFlow >= {MIN_TF_VERSION}."

from packaging import version


try:
    from tensorflow import __version__ as tf_version
except ImportError:  # pragma: no cover
    raise ImportError("TensorFlow is not installed. " + TF_VERSION_ERR)
else:
    if version.parse(tf_version) < version.parse(MIN_TF_VERSION):  # pragma: no cover
        raise ImportError(TF_VERSION_ERR)

from tensorflow import keras as _keras

from scikeras import _saving_utils


_keras.Model.__reduce__ = _saving_utils.pack_keras_model
_keras.losses.Loss.__reduce__ = _saving_utils.pack_keras_loss
_keras.metrics.Metric.__reduce__ = _saving_utils.pack_keras_metric
_keras.optimizers.Optimizer.__reduce__ = _saving_utils.pack_keras_optimizer
