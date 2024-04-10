"""Top-level package for Scikit-Learn Wrapper for Keras."""

__author__ = """Adrian Garcia Badaracco"""

import importlib.metadata as importlib_metadata

__version__ = importlib_metadata.version("scikeras")  # type: ignore

import keras as _keras

from scikeras import _saving_utils

_keras.Model.__reduce__ = _saving_utils.pack_keras_model
_keras.Model.__deepcopy__ = _saving_utils.deepcopy_model
_keras.losses.Loss.__reduce__ = _saving_utils.pack_keras_loss
_keras.metrics.Metric.__reduce__ = _saving_utils.pack_keras_metric
_keras.optimizers.Optimizer.__reduce__ = _saving_utils.pack_keras_optimizer
