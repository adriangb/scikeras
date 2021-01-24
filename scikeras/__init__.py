"""Top-level package for Scikit-Learn Wrapper for Keras."""

__author__ = """Adrian Garcia Badaracco"""
__version__ = "0.2.1"

from tensorflow import keras as _keras

from scikeras import _saving_utils


_keras.Model.__reduce__ = _saving_utils.pack_keras_model
_keras.losses.Loss.__reduce__ = _saving_utils.pack_keras_loss
_keras.metrics.Metric.__reduce__ = _saving_utils.pack_keras_metric
_keras.optimizers.Optimizer.__reduce__ = _saving_utils.pack_keras_optimizer
