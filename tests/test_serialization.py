import pickle

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import tensorflow as tf

from sklearn.base import clone
from sklearn.datasets import load_boston, make_regression
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

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


@keras.utils.register_keras_serializable()
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

    @keras.utils.register_keras_serializable()
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


@pytest.mark.parametrize(
    "optim",
    [
        pytest.param(
            "adam",
            marks=pytest.mark.xfail(
                reason="https://github.com/tensorflow/tensorflow/issues/44670",
                raises=AssertionError,
            ),
        ),
        "sgd",
    ],
)
def test_partial_fit_pickle(optim):
    """
    This test is implemented to make sure model pickling does not affect
    training.

    (this is essentially what Dask-ML does for search)
    """
    X, y = make_regression(n_features=8, n_samples=100)

    m1 = KerasRegressor(model=dynamic_regressor, optimizer=optim, random_state=42)
    m2 = clone(m1)

    # Make sure start from same model
    m1.partial_fit(X, y)
    m2.partial_fit(X, y)
    assert _weights_close(m1, m2)

    # Train; make sure pickling doesn't affect it
    for k in range(4):
        m1.partial_fit(X, y)
        m2 = _reload(m2, epoch=k + 1).partial_fit(X, y)

        # Make sure the same model is produced
        assert _weights_close(m1, m2)

        # Make sure predictions are the same
        assert np.allclose(m1.predict(X), m2.predict(X))


@pytest.mark.parametrize(
    "loss",
    [
        keras.losses.binary_crossentropy,
        keras.losses.BinaryCrossentropy(),
        keras.losses.mean_absolute_error,
        keras.losses.MeanAbsoluteError(),
    ],
)
def test_pickle_loss(loss):
    y1 = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    y2 = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    v1 = loss(y1, y2)
    loss = pickle.loads(pickle.dumps(loss))
    v2 = loss(y1, y2)
    np.testing.assert_almost_equal(v1, v2)


@pytest.mark.parametrize(
    "metric",
    [
        keras.metrics.binary_crossentropy,
        keras.metrics.BinaryCrossentropy(),
        keras.metrics.mean_absolute_error,
        keras.metrics.MeanAbsoluteError(),
    ],
)
def test_pickle_loss(metric):
    y1 = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    y2 = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    v1 = metric(y1, y2)
    metric = pickle.loads(pickle.dumps(metric))
    v2 = metric(y1, y2)
    np.testing.assert_almost_equal(v1, v2)


@pytest.mark.parametrize(
    "opt_cls",
    [
        pytest.param(
            keras.optimizers.Adam,
            marks=pytest.mark.xfail(
                reason="https://github.com/tensorflow/tensorflow/issues/44670",
                raises=AssertionError,
            ),
        ),
        pytest.param(
            keras.optimizers.RMSprop,
            marks=pytest.mark.xfail(
                reason="https://github.com/tensorflow/tensorflow/issues/44670",
                raises=AssertionError,
            ),
        ),
        keras.optimizers.SGD,
    ],
)
def test_pickle_optimizer(opt_cls):
    # Minimize a variable subject to two different
    # loss functions
    opt = opt_cls()
    var1 = tf.Variable(10.0)
    loss1 = lambda: (var1 ** 2) / 2.0
    opt.minimize(loss1, [var1]).numpy()
    loss2 = lambda: (var1 ** 2) / 1.0
    opt.minimize(loss2, [var1]).numpy()
    val_no_pickle = var1.numpy()
    # Do the same with a roundtrip pickle in the middle
    opt = opt_cls()
    var1 = tf.Variable(10.0)
    loss1 = lambda: (var1 ** 2) / 2.0
    opt.minimize(loss1, [var1]).numpy()
    loss2 = lambda: (var1 ** 2) / 1.0
    opt = pickle.loads(pickle.dumps(opt))
    opt.minimize(loss2, [var1]).numpy()
    val_pickle = var1.numpy()
    # Check that the final values are the same
    np.testing.assert_equal(val_no_pickle, val_pickle)
