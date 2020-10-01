"""Misc. tests that only exist to verify errors are raised."""

import numpy as np
import pytest

from sklearn.exceptions import NotFittedError
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from scikeras.wrappers import KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor


def test_shape_change_error():
    """Tests that a ValueError is raised if the input
    changes shape in subsequent partial fit calls.
    """

    estimator = KerasRegressor(
        model=dynamic_regressor,
        loss=KerasRegressor.r_squared,
        hidden_layer_sizes=(100,),
    )
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0, 1, 0], [1, 0, 0]])

    estimator.fit(X=X, y=y)

    with pytest.raises(ValueError, match=r"but this [\w\d]+ is expecting "):
        # Calling with a different shape for X raises an error
        estimator.partial_fit(X=X[:, :1], y=y)


def test_not_fitted_error():
    """Tests error when trying to use predict before fit.
    """
    estimator = KerasClassifier(
        build_fn=dynamic_classifier, loss=KerasRegressor.r_squared,
    )
    X = np.random.rand(10, 20)
    with pytest.raises(NotFittedError):
        # This is in BaseWrapper so it covers
        # KerasRegressor as well
        estimator.predict(X)
    with pytest.raises(NotFittedError):
        estimator.predict_proba(X)


class TestInvalidBuildFn:
    """Tests various error cases for BuildFn.
    """

    def test_invalid_build_fn(self):
        class Model:
            pass

        clf = KerasClassifier(model=Model())
        with pytest.raises(TypeError, match="`model` must be"):
            clf.fit(np.array([[0], [1]]), np.array([0, 1]))

    def test_no_build_fn(self):
        class NoBuildFn(KerasClassifier):
            pass

        clf = NoBuildFn()

        with pytest.raises(ValueError, match="must implement `_keras_build_fn`"):
            clf.fit(np.array([[0], [1]]), np.array([0, 1]))

    def test_call_and_build_fn_function(self):
        class Clf(KerasClassifier):
            def _keras_build_fn(self, hidden_layer_sizes=(100,)):
                return dynamic_classifier(hidden_layer_sizes=hidden_layer_sizes)

        def dummy_func():
            return None

        clf = Clf(build_fn=dummy_func,)

        with pytest.raises(ValueError, match="cannot implement `_keras_build_fn`"):
            clf.fit(np.array([[0], [1]]), np.array([0, 1]))


def test_sample_weights_all_zero():
    """Checks for a user-friendly error when sample_weights
    are all zero.
    """
    # build estimator
    estimator = KerasClassifier(
        build_fn=dynamic_classifier,
        model__hidden_layer_sizes=(100,),
        epochs=10,
        random_state=0,
    )

    # we create 20 points
    n, d = 50, 4
    X = np.random.uniform(size=(n, d))
    y = np.random.uniform(size=n)
    sample_weight = np.zeros(y.shape)

    with pytest.raises(RuntimeError, match="no samples left"):
        estimator.fit(X, y, sample_weight=sample_weight)


def test_build_fn_deprecation():
    """An appropriate warning is raised when using the `build_fn`
    parameter instead of `model`.
    """
    clf = KerasClassifier(
        build_fn=dynamic_regressor, model__hidden_layer_sizes=(100,), loss="auto"
    )
    with pytest.warns(UserWarning, match="`build_fn` will be renamed to `model`"):
        clf.fit(np.array([[0], [1]]), np.array([0, 1]))


@pytest.mark.parametrize("wrapper", [KerasClassifier, KerasRegressor])
def test_build_fn_and_init_signature_do_not_agree(wrapper):
    """Test that passing a kwarg not present in the model
    building function's signature raises a TypeError.
    """

    def no_bar(foo=42):
        pass

    # all attempts to pass `bar` should fail
    est = wrapper(model=no_bar, model__bar=42)
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        est.fit(np.array([[0], [1]]), np.array([0, 1]))
    est = wrapper(model=no_bar, bar=42)
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        est.fit(np.array([[0], [1]]), np.array([0, 1]))
    est = wrapper(model=no_bar, model__bar=42, foo=43)
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        est.fit(np.array([[0], [1]]), np.array([0, 1]))
    est = wrapper(model=no_bar, bar=42, foo=43)
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        est.fit(np.array([[0], [1]]), np.array([0, 1]))


@pytest.mark.parametrize("loss", [None, [None]])
@pytest.mark.parametrize("compile", [True, False])
def test_no_loss(loss, compile):
    def get_model(compile, meta, compile_kwargs):
        inp = Input(shape=(meta["n_features_in_"],))
        hidden = Dense(10, activation="relu")(inp)
        out = [
            Dense(1, activation="sigmoid", name=f"out{i+1}")(hidden)
            for i in range(meta["n_outputs_"])
        ]
        model = Model(inp, out)
        if compile:
            model.compile(**compile_kwargs)
        return model

    est = KerasRegressor(model=get_model, loss=loss, compile=compile)
    with pytest.raises(ValueError, match="must provide a loss function"):
        est.fit([[1]], [1])


@pytest.mark.parametrize("compile", [True, False])
def test_no_optimizer(compile):
    def get_model(compile, meta, compile_kwargs):
        inp = Input(shape=(meta["n_features_in_"],))
        hidden = Dense(10, activation="relu")(inp)
        out = [
            Dense(1, activation="sigmoid", name=f"out{i+1}")(hidden)
            for i in range(meta["n_outputs_"])
        ]
        model = Model(inp, out)
        if compile:
            model.compile(**compile_kwargs)
        return model

    est = KerasRegressor(
        model=get_model,
        loss="categorical_crossentropy",
        compile=compile,
        optimizer=None,
    )
    with pytest.raises(
        ValueError, match="Could not interpret optimizer identifier"  # Keras error
    ):
        est.fit([[1]], [1])
