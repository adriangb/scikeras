"""Unit test package for scikeras."""
import os

import pytest
import tensorflow as tf


# Force data conversion warnings to be come errors
pytestmark = pytest.mark.filterwarnings(
    "error::sklearn.exceptions.DataConversionWarning"
)

# Don't filter tensorflow tracebacks
tf.debugging.disable_traceback_filtering()

# Disable TensorFlow warnings
# most of them are about missing CUDA libs and such
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
