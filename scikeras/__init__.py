"""Top-level package for Scikit-Learn Wrapper for Keras."""

__author__ = """Adrian Garcia Badaracco"""
__version__ = "0.2.0"

from tensorflow import keras
from tensorflow.python.keras import metrics as _metrics  # noqa


_metrics.log_cosh = (
    _metrics.metrics.logcosh
)  # See https://github.com/tensorflow/tensorflow/pull/42097

from scikeras import _saving_utils as saving_utils  # noqa


keras.Model.__reduce__ = saving_utils.pack_keras_model
keras.losses.Loss.__reduce__ = saving_utils.pack_keras_loss
keras.metrics.Metric.__reduce__ = saving_utils.pack_keras_metric
keras.optimizers.Optimizer.__reduce__ = saving_utils.pack_keras_optimizer
