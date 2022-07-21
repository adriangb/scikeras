"""Misc. tests that only exist to verify errors are raised."""

import numpy as np
import pytest

from sklearn.exceptions import NotFittedError
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from scikeras.wrappers import BaseWrapper, KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor


def test_X_shape_change():
    """Tests that a ValueError is raised if the input
    changes shape in subsequent partial fit calls.
    """

    estimator = KerasRegressor(
        model=dynamic_regressor,
        hidden_layer_sizes=(100,),
    )
    X = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
    y = np.array([[0, 1, 0], [1, 0, 0]])

    estimator.fit(X=X, y=y)

    with pytest.raises(ValueError, match="dimensions in X"):
        # Calling with a different number of dimensions for X raises an error
        estimator.partial_fit(X=X.reshape(2, 2), y=y)


def test_unknown_param():
    """Test that setting a parameter unknown to set_params raises a
    friendly error message.
    """
    est = BaseWrapper()
    err = (
        r"Invalid parameter test for estimator [\S\s]*"
        "This issue can likely be resolved by setting this parameter"
    )
    with pytest.raises(ValueError, match=err):
        est.set_params(test=1)


def test_not_fitted_error():
    """Tests error when trying to use predict before fit."""
    estimator = KerasClassifier(dynamic_classifier)
    X = np.random.rand(10, 20)
    with pytest.raises(NotFittedError):
        # This is in BaseWrapper so it covers
        # KerasRegressor as well
        estimator.predict(X)
    with pytest.raises(NotFittedError):
        estimator.predict_proba(X)


class TestInvalidBuildFn:
    """Tests various error cases for BuildFn."""

    def test_invalid_build_fn(self):
        class Model:
            pass

        clf = KerasClassifier(model=Model())
        with pytest.raises(TypeError, match="``model`` must be"):
            clf.fit(np.array([[0], [1]]), np.array([0, 1]))

    def test_no_build_fn(self):
        class NoBuildFn(KerasClassifier):
            pass

        clf = NoBuildFn()

        with pytest.raises(ValueError, match="must implement ``_keras_build_fn``"):
            clf.fit(np.array([[0], [1]]), np.array([0, 1]))

    def test_call_and_build_fn_function(self):
        class Clf(KerasClassifier):
            def _keras_build_fn(self, hidden_layer_sizes=(100,)):
                return dynamic_classifier(hidden_layer_sizes=hidden_layer_sizes)

        def dummy_func():
            return None

        clf = Clf(
            model=dummy_func,
        )

        with pytest.raises(ValueError, match="cannot implement ``_keras_build_fn``"):
            clf.fit(np.array([[0], [1]]), np.array([0, 1]))


def test_sample_weights_all_zero():
    """Checks for a user-friendly error when sample_weights
    are all zero.
    """
    # build estimator
    estimator = KerasClassifier(
        model=dynamic_classifier,
        model__hidden_layer_sizes=(100,),
    )

    # we create 20 points
    n, d = 50, 4
    X = np.random.uniform(size=(n, d))
    y = np.random.choice(2, size=n).astype("uint8")
    sample_weight = np.zeros(y.shape)

    with pytest.raises(ValueError, match="only zeros were passed in sample_weight"):
        estimator.fit(X, y, sample_weight=sample_weight)


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
        est.fit([[0], [1]], [0, 1])
    est = wrapper(model=no_bar, bar=42)
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        est.fit([[0], [1]], [0, 1])
    est = wrapper(model=no_bar, model__bar=42, foo=43)
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        est.fit([[0], [1]], [0, 1])
    est = wrapper(model=no_bar, bar=42, foo=43)
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        est.fit([[0], [1]], [0, 1])


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
        est.fit([[0], [1]], [0, 1])


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
        loss="mse",
        compile=compile,
        optimizer=None,
    )
    with pytest.raises(
        ValueError, match="Could not interpret optimizer identifier"  # Keras error
    ):
        est.fit([[0], [1]], [0, 1])


def test_target_dtype_changes_incremental_fit():
    X = np.array([[1, 2], [2, 3]])
    y = np.array([1, 3])

    est = KerasClassifier(model=dynamic_classifier, hidden_layer_sizes=(100,))
    est.fit(X, y)
    est.partial_fit(X, y.astype(np.uint8))
    with pytest.raises(
        ValueError,
        match="Got y with dtype",
    ):
        est.partial_fit(X, y.astype(np.float64))


def test_target_dims_changes_incremental_fit():
    X = np.array([[1, 2], [2, 3]])
    y = np.array([1, 3])

    est = KerasClassifier(model=dynamic_classifier, hidden_layer_sizes=(100,))
    est.fit(X, y)
    y_new = y.reshape(-1, 1)
    with pytest.raises(
        ValueError,
        match="y has 2 dimensions, but this ",
    ):
        est.partial_fit(X, y_new)


def test_target_shape_changes_incremental_fit_clf():
    X = np.array([[1, 2], [2, 3]])
    y = np.array([1, 3]).reshape(-1, 1)

    est = KerasClassifier(model=dynamic_classifier, hidden_layer_sizes=(100,))
    est.fit(X, y)
    with pytest.raises(ValueError, match="features"):  # raised by transformers
        est.partial_fit(X, np.column_stack([y, y]))


def test_target_shape_changes_incremental_fit_reg():
    X = np.array([[1, 2], [2, 3]])
    y = np.array([1, 3]).reshape(-1, 1)

    est = KerasRegressor(model=dynamic_regressor, hidden_layer_sizes=(100,))
    est.fit(X, y)
    with pytest.raises(
        ValueError,
        match="Detected ``y`` to have ",
    ):
        est.partial_fit(X, np.column_stack([y, y]))


def test_X_dtype_changes_incremental_fit():
    X = np.array([[1, 2], [2, 3]])
    y = np.array([1, 3])

    est = KerasClassifier(model=dynamic_classifier, hidden_layer_sizes=(100,))
    est.fit(X, y)
    est.partial_fit(X.astype(np.uint8), y)
    with pytest.raises(
        ValueError,
        match="Got X with dtype",
    ):
        est.partial_fit(X.astype(np.float64), y)


def test_target_classes_change_incremental_fit():
    X = np.array([[1, 2], [2, 3]])
    y = np.array([1, 3])

    est = KerasClassifier(model=dynamic_classifier, hidden_layer_sizes=(100,))
    est.fit(X, y)
    est.partial_fit(X.astype(np.uint8), y)
    with pytest.raises(
        ValueError,
        match="Found unknown categories",
    ):
        y[0] = 10
        est.partial_fit(X, y)
