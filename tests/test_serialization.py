import pickle

from typing import Any, Dict

import numpy as np
import pytest

from sklearn.base import clone
import tensorflow as tf
from sklearn.datasets import load_boston, make_regression
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Input
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
    np.testing.assert_almost_equal(score, score_new, decimal=2)


# ---------------------- Custom Loss Test ----------------------


@keras.utils.generic_utils.register_keras_serializable()
class CustomLoss(keras.losses.MeanSquaredError):
    """Dummy custom loss."""

    pass


def test_custom_loss_function():
    """Test that a custom loss function can be serialized.
    """
    estimator = KerasRegressor(
        model=dynamic_regressor, loss=CustomLoss(), model__hidden_layer_sizes=(100,),
    )
    check_pickle(estimator, load_boston)


# ---------------------- Subclassed Model Tests ------------------


def build_fn_custom_model_registered(
    meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Dummy custom Model subclass that is registered to be serializable.
    """

    @keras.utils.generic_utils.register_keras_serializable()
    class CustomModelRegistered(Model):
        pass

    # get parameters
    n_features_in_ = meta["n_features_in_"]
    n_outputs_ = meta["n_outputs_"]

    inp = Input(shape=n_features_in_)
    x1 = Dense(n_features_in_, activation="relu")(inp)
    out = Dense(n_outputs_, activation="linear")(x1)
    model = CustomModelRegistered(inp, out)
    model.compile("adam", loss="mean_squared_error")
    return model


def test_custom_model_registered():
    """Test that a registered subclassed Model can be serialized.
    """
    estimator = KerasRegressor(model=build_fn_custom_model_registered)
    check_pickle(estimator, load_boston)


def build_fn_custom_model_unregistered(
    meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Dummy custom Model subclass that is not registed to be serializable.
    """

    class CustomModelUnregistered(Model):
        pass

    # get parameters
    n_features_in_ = meta["n_features_in_"]
    n_outputs_ = meta["n_outputs_"]

    inp = Input(shape=n_features_in_)
    x1 = Dense(n_features_in_, activation="relu")(inp)
    out = Dense(n_outputs_, activation="linear")(x1)
    model = CustomModelUnregistered(inp, out)
    model.compile("adam", loss="mean_squared_error")
    return model


def test_custom_model_unregistered():
    """Test that an unregistered subclassed Model raises an error.
    """
    estimator = KerasRegressor(model=build_fn_custom_model_unregistered)
    with pytest.raises(ValueError, match="Unknown layer"):
        check_pickle(estimator, load_boston)


# ---------------- Model Compiled with `run_eagerly` --------------------


def test_run_eagerly():
    """Test that models compiled with run_eagerly can be serialized.
    """
    estimator = KerasRegressor(
        model=dynamic_regressor,
        run_eagerly=True,
        loss=KerasRegressor.r_squared,
        model__hidden_layer_sizes=(100,),
    )
    check_pickle(estimator, load_boston)


def _weights_close(model1, model2):
    w1 = [x.numpy() for x in model1.model_.weights]
    w2 = [x.numpy() for x in model2.model_.weights]
    assert len(w1) == len(w2)
    for _1, _2 in zip(w1, w2):
        assert np.allclose(_1, _2, rtol=0.01)
    return True


def _reload(model, epoch=None):
    tmp = pickle.loads(pickle.dumps(model))
    assert tmp is not model
    if epoch:
        assert tmp.current_epoch == model.current_epoch == epoch
    return tmp


def test_partial_fit_pickle(optim="adam"):
    """
    This test is implemented to make sure model pickling does not affect
    training.

    (this is essentially what Dask-ML does for search)
    """
    from scikeras._utils import TFRandomState
    X, y = make_regression(n_features=8, n_samples=100)

    m1 = KerasRegressor(model=dynamic_regressor, optimizer=optim)
    m2 = clone(m1)

    # Make sure start from same model
    with TFRandomState(42):
        m1.partial_fit(X, y)
    with TFRandomState(42):
        m2.partial_fit(X, y)
    assert _weights_close(m1, m2)

    # Train; make sure pickling doesn't affect it
    for k in range(4):
        with TFRandomState(42):
            m1.partial_fit(X, y)
        with TFRandomState(42):
            m2 = _reload(m2, epoch=k + 1).partial_fit(X, y)

        # Make sure the same model is produced
        assert _weights_close(m1, m2)

        # Make sure predictions are the same
        assert np.allclose(m1.predict(X), m2.predict(X))
