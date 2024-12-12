"""Top-level package for Scikit-Learn Wrapper for Keras."""

__author__ = """Adrian Garcia Badaracco"""

import importlib.metadata as importlib_metadata
from warnings import warn

__version__ = importlib_metadata.version("scikeras")  # type: ignore

import keras as _keras

from scikeras import _saving_utils

_keras.Model.__reduce__ = _saving_utils.pack_keras_model
_keras.Model.__deepcopy__ = _saving_utils.deepcopy_model
_keras.losses.Loss.__reduce__ = _saving_utils.pack_keras_loss
_keras.metrics.Metric.__reduce__ = _saving_utils.pack_keras_metric
_keras.optimizers.Optimizer.__reduce__ = _saving_utils.pack_keras_optimizer

warn(
    """
    This project is now deprecated. Keras has re-introduced wrappers with a similar API to those in SciKeras, but they will be better maintained.
    SciKeras was a project to meet a specific need that was developed by a single developer.
    I no longer use Keras nor do I have the time to maintain this project, which became increasingly difficult with multiple versions of Keras and Scikit-Learn to support.
    I thank all of the users and contributors over the years and hope that the new Keras wrappers will meet your needs.
    TODO: add link to Keras docs and release here.
    """,
    stacklevel=1,
)
