"""Top-level package for Scikit-Learn Wrapper for Keras."""

__author__ = """Adrian Garcia Badaracco"""
__version__ = "0.1.8"


# Monkey patch log_cosh reference
# See https://github.com/tensorflow/tensorflow/pull/42097
# Will be removed whenever the
# min supported version of tf incorporates the fix
import tensorflow.python.keras.metrics  # noqa


tensorflow.python.keras.metrics.log_cosh = (
    tensorflow.python.keras.metrics.logcosh
)
