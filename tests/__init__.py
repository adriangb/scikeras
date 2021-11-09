"""Unit test package for scikeras."""
import pytest
import tensorflow as tf


# Force data conversion warnings to be come errors
pytestmark = pytest.mark.filterwarnings(
    "error::sklearn.exceptions.DataConversionWarning"
)


# Don't filter tensorflow tracebacks
tf.debugging.disable_traceback_filtering()
