import pickle

import numpy as np
import pytest

from sklearn.datasets import load_boston
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

from scikeras.wrappers import KerasRegressor

from .mlp_models import dynamic_regressor


def check_pickle(estimator, loader):
    """Run basic checks (fit, score, pickle) on estimator."""
    data = loader()
    # limit to 100 data points to speed up testing
    X, y = data.data[:100], data.target[:100]
    estimator.fit(X, y)
    estimator.predict(X)
    score = estimator.score(X, y)
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    deserialized_estimator.predict(X)
    score_new = deserialized_estimator.score(X, y)
    np.testing.assert_almost_equal(score, score_new)


# ---------------------- Custom Loss Test ----------------------


@keras.utils.generic_utils.register_keras_serializable()
class CustomLoss(keras.losses.MeanSquaredError):
    """Dummy custom loss."""

    pass


def test_custom_loss_function():
    """Test that a custom loss function can be serialized.
    """
    estimator = KerasRegressor(build_fn=dynamic_regressor, loss=CustomLoss(),)
    check_pickle(estimator, load_boston)


# ---------------------- Subclassed Model Tests ------------------


def build_fn_custom_model_registered(n_features_in_, n_outputs_):
    """Dummy custom Model subclass that is registered to be serializable.
    """

    @keras.utils.generic_utils.register_keras_serializable()
    class CustomModelRegistered(Model):
        pass

    inp = Input(shape=n_features_in_)
    x1 = Dense(n_features_in_, activation="relu")(inp)
    out = Dense(n_outputs_, activation="linear")(x1)
    model = CustomModelRegistered(inp, out)
    model.compile("adam", loss="mean_squared_error")
    return model


def test_custom_model_registered():
    """Test that a registered subclassed Model can be serialized.
    """
    estimator = KerasRegressor(build_fn=build_fn_custom_model_registered)
    check_pickle(estimator, load_boston)


def build_fn_custom_model_unregistered(n_features_in_, n_outputs_):
    """Dummy custom Model subclass that is not registed to be serializable.
    """

    class CustomModelUnregistered(Model):
        pass

    inp = Input(shape=n_features_in_)
    x1 = Dense(n_features_in_, activation="relu")(inp)
    out = Dense(n_outputs_, activation="linear")(x1)
    model = CustomModelUnregistered(inp, out)
    model.compile("adam", loss="mean_squared_error")
    return model


def test_custom_model_unregistered():
    """Test that an unregistered subclassed Model raises an error.
    """
    estimator = KerasRegressor(build_fn=build_fn_custom_model_unregistered)
    with pytest.raises(ValueError, match="Unknown layer"):
        check_pickle(estimator, load_boston)


# ---------------- Model Compiled with `run_eagerly` --------------------


def test_run_eagerly():
    """Test that models compiled with run_eagerly can be serialized.
    """
    estimator = KerasRegressor(
        build_fn=dynamic_regressor,
        run_eagerly=True,
        loss=KerasRegressor.r_squared,
    )
    check_pickle(estimator, load_boston)
