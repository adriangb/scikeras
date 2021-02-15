"""Top-level package for Scikit-Learn Wrapper for Keras."""

__author__ = """Adrian Garcia Badaracco"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata  # python <3.8

__version__ = importlib_metadata.version("scikeras")

from tensorflow import keras as _keras

from scikeras import _saving_utils


_keras.Model.__reduce__ = _saving_utils.pack_keras_model
_keras.losses.Loss.__reduce__ = _saving_utils.pack_keras_loss
_keras.metrics.Metric.__reduce__ = _saving_utils.pack_keras_metric
_keras.optimizers.Optimizer.__reduce__ = _saving_utils.pack_keras_optimizer
