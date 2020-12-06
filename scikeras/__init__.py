"""Top-level package for Scikit-Learn Wrapper for Keras."""

__author__ = """Adrian Garcia Badaracco"""
__version__ = "0.2.1"


# Monkey patch log_cosh reference
# See https://github.com/tensorflow/tensorflow/pull/42097
# Will be removed whenever the
# min supported version of tf incorporates the fix
from tensorflow.python import keras  # noqa


keras.metrics.log_cosh = keras.metrics.logcosh
