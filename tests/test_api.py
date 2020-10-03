"""Tests for Scikit-learn API wrapper."""
import pickle

from typing import Any, Dict

import numpy as np
import pytest

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_boston, load_digits, load_iris
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.np_utils import to_categorical

from scikeras.wrappers import KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor
from .testing_utils import basic_checks


def build_fn_clf(
    hidden_dim, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Builds a Sequential based classifier."""
    # extract parameters
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(hidden_dim))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(n_classes_))
    model.add(keras.layers.Activation("softmax"))
    model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_fn_reg(
    hidden_dim, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Builds a Sequential based regressor."""
    # extract parameters
    n_features_in_ = meta["n_features_in_"]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, input_shape=(n_features_in_,)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(hidden_dim))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("linear"))
    model.compile(optimizer="sgd", loss="mean_absolute_error", metrics=["accuracy"])
    return model


class InheritClassBuildFnClf(KerasClassifier):
    def _keras_build_fn(
        self, hidden_dim, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
    ) -> Model:
        return build_fn_clf(
            hidden_dim=hidden_dim, meta=meta, compile_kwargs=compile_kwargs,
        )


class InheritClassBuildFnReg(KerasRegressor):
    def _keras_build_fn(
        self, hidden_dim, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
    ) -> Model:
        return build_fn_reg(
            hidden_dim=hidden_dim, meta=meta, compile_kwargs=compile_kwargs,
        )


class TestBasicAPI:
    """Tests basic functionality."""

    def test_classify_build_fn(self):
        """Tests a classification task for errors."""
        clf = KerasClassifier(build_fn=build_fn_clf, hidden_dim=5)
        basic_checks(clf, load_iris)

    def test_classify_inherit_class_build_fn(self):
        """Tests for errors using an inherited class."""

        clf = InheritClassBuildFnClf(build_fn=None, hidden_dim=5)
        basic_checks(clf, load_iris)

    def test_regression_build_fn(self):
        """Tests for errors using KerasRegressor."""
        reg = KerasRegressor(build_fn=build_fn_reg, hidden_dim=5)
        basic_checks(reg, load_boston)

    def test_regression_inherit_class_build_fn(self):
        """Tests for errors using KerasRegressor inherited."""

        reg = InheritClassBuildFnReg(build_fn=None, hidden_dim=5,)
        basic_checks(reg, load_boston)


def load_digits8x8():
    """Load image 8x8 dataset."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    # Convert NCHW to NHWC
    # Convert back to numpy or sklearn funcs (GridSearchCV, etc.) WILL fail
    data.data = np.transpose(data.data, [0, 2, 3, 1])
    K.set_image_data_format("channels_last")
    return data


def build_fn_regs(
    hidden_layer_sizes, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Dynamically build regressor."""
    # get params
    X_shape_ = meta["X_shape_"]
    n_outputs_ = meta["n_outputs_"]

    model = Sequential()
    model.add(Dense(X_shape_[1], activation="relu", input_shape=X_shape_[1:]))
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation="relu"))
    model.add(Dense(n_outputs_))
    model.compile("adam", loss="mean_squared_error")
    return model


def build_fn_clss(
    hidden_layer_sizes, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Dynamically build classifier."""
    # get params
    X_shape_ = meta["X_shape_"]

    model = Sequential()
    model.add(Dense(X_shape_[1], activation="relu", input_shape=X_shape_[1:]))
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation="relu"))
    model.add(Dense(1, activation="softmax"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_fn_clscs(
    hidden_layer_sizes, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Dynamically build functional API regressor."""
    # get params
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]

    model = Sequential()
    model.add(Conv2D(3, (3, 3), input_shape=X_shape_[1:]))
    model.add(Flatten())
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation="relu"))
    model.add(Dense(n_classes_, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_fn_clscf(
    hidden_layer_sizes, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Dynamically build functional API classifier."""
    # get params
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]

    x = Input(shape=X_shape_[1:])
    z = Conv2D(3, (3, 3))(x)
    z = Flatten()(z)
    for size in hidden_layer_sizes:
        z = Dense(size, activation="relu")(z)
    y = Dense(n_classes_, activation="softmax")(z)
    model = Model(inputs=x, outputs=y)
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


CONFIG = {
    "MLPRegressor": (
        load_boston,
        KerasRegressor,
        dynamic_regressor,
        (BaggingRegressor, AdaBoostRegressor),
    ),
    "MLPClassifier": (
        load_iris,
        KerasClassifier,
        dynamic_classifier,
        (BaggingClassifier, AdaBoostClassifier),
    ),
    "CNNClassifier": (
        load_digits8x8,
        KerasClassifier,
        build_fn_clscs,
        (BaggingClassifier, AdaBoostClassifier),
    ),
    "CNNClassifierF": (
        load_digits8x8,
        KerasClassifier,
        build_fn_clscf,
        (BaggingClassifier, AdaBoostClassifier),
    ),
}


class TestAdvancedAPIFuncs:
    """Tests advanced features such as pipelines and hyperparameter tuning."""

    @pytest.mark.parametrize(
        "config", ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
    )
    def test_standalone(self, config):
        """Tests standalone estimator."""
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, n_iter=1, model__hidden_layer_sizes=[])
        basic_checks(estimator, loader)

    @pytest.mark.parametrize("config", ["MLPRegressor", "MLPClassifier"])
    def test_pipeline(self, config):
        """Tests compatibility with Scikit-learn's pipeline."""
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, n_iter=1, model__hidden_layer_sizes=[])
        estimator = Pipeline([("s", StandardScaler()), ("e", estimator)])
        basic_checks(estimator, loader)

    @pytest.mark.parametrize(
        "config", ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
    )
    def test_searchcv_init_params(self, config):
        """Tests compatibility with Scikit-learn's hyperparameter search CV."""
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(
            build_fn, n_iter=1, validation_split=0.1, model__hidden_layer_sizes=[],
        )
        basic_checks(
            GridSearchCV(estimator, {"model__hidden_layer_sizes": [[], [5]]}), loader,
        )
        basic_checks(
            RandomizedSearchCV(
                estimator,
                {"n_iter": [1, 2, 3], "optimizer": ["rmsprop", "sgd"]},
                n_iter=2,
            ),
            loader,
        )

    @pytest.mark.parametrize(
        "config", ["MLPClassifier"],
    )
    def test_searchcv_routed_params(self, config):
        """Tests compatibility with Scikit-learn's hyperparameter search CV."""
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, n_iter=1, model__hidden_layer_sizes=[])
        params = {
            "model__hidden_layer_sizes": [[], [5]],
            "optimizer": ["sgd", "adam"],
        }
        search = GridSearchCV(estimator, params)
        basic_checks(search, loader)
        assert search.best_estimator_.model_.optimizer._name.lower() in ("sgd", "adam",)

    @pytest.mark.parametrize("config", ["MLPRegressor", "MLPClassifier"])
    def test_ensemble(self, config):
        """Tests compatibility with Scikit-learn's ensembles."""
        loader, model, build_fn, ensembles = CONFIG[config]
        base_estimator = model(build_fn, n_iter=1, model__hidden_layer_sizes=[])
        for ensemble in ensembles:
            estimator = ensemble(base_estimator=base_estimator, n_estimators=2)
            basic_checks(estimator, loader)

    @pytest.mark.parametrize("config", ["MLPClassifier"])
    def test_calibratedclassifiercv(self, config):
        """Tests compatibility with Scikit-learn's calibrated classifier CV."""
        loader, _, build_fn, _ = CONFIG[config]
        base_estimator = KerasClassifier(
            build_fn, n_iter=1, model__hidden_layer_sizes=[]
        )
        estimator = CalibratedClassifierCV(base_estimator=base_estimator, cv=5)
        basic_checks(estimator, loader)


class TestPrebuiltModel:
    """Tests using a prebuilt model instance."""

    @pytest.mark.parametrize(
        "config", ["MLPRegressor", "MLPClassifier"],
    )
    def test_basic(self, config):
        """Tests using a prebuilt model."""
        loader, model, build_fn, _ = CONFIG[config]
        data = loader()
        x_train, y_train = data.data[:100], data.target[:100]

        n_classes_ = np.unique(y_train).size
        # make y the same shape as will be used by .fit
        if config != "MLPRegressor":
            y_train = to_categorical(y_train)
            meta = {
                "n_classes_": n_classes_,
                "target_type_": "multiclass",
                "n_features_in_": x_train.shape[1],
                "model_n_outputs_": 1,
            }
            keras_model = build_fn(
                meta=meta,
                hidden_layer_sizes=(100,),
                compile_kwargs={"optimizer": "adam", "loss": None, "metrics": None,},
            )
        else:
            meta = {
                "n_outputs_": 1,
                "n_features_in_": x_train.shape[1],
            }
            keras_model = build_fn(
                meta=meta,
                hidden_layer_sizes=(100,),
                compile_kwargs={"optimizer": "adam", "loss": None, "metrics": None,},
            )

        estimator = model(build_fn=keras_model)
        basic_checks(estimator, loader)

    @pytest.mark.parametrize("config", ["MLPRegressor", "MLPClassifier"])
    def test_ensemble(self, config):
        """Tests using a prebuilt model in an ensemble learner."""
        loader, model, build_fn, ensembles = CONFIG[config]
        data = loader()
        x_train, y_train = data.data[:100], data.target[:100]

        n_classes_ = np.unique(y_train).size
        # make y the same shape as will be used by .fit
        if config != "MLPRegressor":
            y_train = to_categorical(y_train)
            meta = {
                "n_classes_": n_classes_,
                "target_type_": "multiclass",
                "n_features_in_": x_train.shape[1],
                "model_n_outputs_": 1,
            }
            keras_model = build_fn(
                meta=meta,
                hidden_layer_sizes=(100,),
                compile_kwargs={"optimizer": "adam", "loss": None, "metrics": None,},
            )
        else:
            meta = {
                "n_outputs_": 1,
                "n_features_in_": x_train.shape[1],
            }
            keras_model = build_fn(
                meta=meta,
                hidden_layer_sizes=(100,),
                compile_kwargs={"optimizer": "adam", "loss": None, "metrics": None,},
            )

        base_estimator = model(build_fn=keras_model)
        for ensemble in ensembles:
            estimator = ensemble(base_estimator=base_estimator, n_estimators=2)
            basic_checks(estimator, loader)


def test_warm_start():
    """Test the warm start parameter."""
    # Load data
    data = load_boston()
    X, y = data.data[:100], data.target[:100]
    # Initial fit
    estimator = KerasRegressor(
        build_fn=dynamic_regressor,
        loss=KerasRegressor.r_squared,
        model__hidden_layer_sizes=(100,),
    )
    estimator.fit(X, y)
    model = estimator.model_

    # With warm start, successive calls to fit
    # should NOT create a new model
    estimator.set_params(warm_start=True)
    estimator.fit(X, y)
    assert model is estimator.model_

    # Without warm start, each call to fit
    # should create a new model instance
    estimator.set_params(warm_start=False)
    for _ in range(3):
        estimator.fit(X, y)
        assert model is not estimator.model_
        model = estimator.model_


class TestPartialFit:
    def test_partial_fit(self):
        data = load_boston()
        X, y = data.data[:100], data.target[:100]
        estimator = KerasRegressor(
            build_fn=dynamic_regressor,
            loss=KerasRegressor.r_squared,
            model__hidden_layer_sizes=[100,],
        )

        estimator.partial_fit(X, y)
        # Make sure loss history is incremented
        assert len(estimator.history_["loss"]) == 1
        estimator.partial_fit(X, y)
        assert len(estimator.history_["loss"]) == 2
        # Make sure new model not created
        model = estimator.model_
        estimator.partial_fit(X, y)
        assert estimator.model_ is model, "Model memory address should remain constant"

    def test_partial_fit_history_len(self):
        # history_ records the history from this partial_fit call
        # Make sure for each call to partial_fit a single entry
        # into the history is added
        # As per https://github.com/keras-team/keras/issues/1766,
        # there is no direct measure of epochs
        data = load_boston()
        X, y = data.data[:100], data.target[:100]
        estimator = KerasRegressor(
            build_fn=dynamic_regressor,
            loss=KerasRegressor.r_squared,
            metrics="mean_squared_error",
            model__hidden_layer_sizes=[100,],
        )

        for k in range(10):
            estimator = estimator.partial_fit(X, y)
            assert len(estimator.history_["loss"]) == k + 1
            assert set(estimator.history_.keys()) == {
                "loss",
                "mean_squared_error",
            }

    @pytest.mark.parametrize(
        "config", ["CNNClassifier", "CNNClassifierF"],
    )
    def test_pf_pickle_pf(self, config):
        loader, model, build_fn, _ = CONFIG[config]
        clf = model(build_fn, n_iter=1, model__hidden_layer_sizes=[])
        data = loader()

        X, y = data.data[:100], data.target[:100]
        clf.partial_fit(X, y)

        # Check that partial_fit -> pickle -> partial_fit
        # builds up the training
        # even after pickling by checking that
        # (1) the history_ attribute grows in length
        # (2) the loss value decreases
        clf2 = clf
        for k in range(2, 4):
            clf2 = pickle.loads(pickle.dumps(clf2))
            clf2.partial_fit(X, y)
            assert len(clf.history_["loss"]) == 1
            assert len(clf2.history_["loss"]) == k
            assert np.allclose(clf.history_["loss"][0], clf2.history_["loss"][0])

        weights1 = [w.numpy() for w in clf.model_.weights]
        weights2 = [w.numpy() for w in clf2.model_.weights]
        n_weights = [w1.size for w1 in weights1]

        # Make sure there's a decent number of weights
        # Also make sure that this network is "over-parameterized" (more
        # weights than examples)
        # (these numbers are empirical and depend on model__hidden_layer_sizes=[])
        assert 1000 <= sum(n_weights) <= 2000
        assert 200 <= np.mean(n_weights) <= 300
        assert max(n_weights) >= 1000
        assert len(n_weights) == 4, "At least 4 layers"

        rel_errors = [
            np.linalg.norm(w1 - w2) / np.linalg.norm((w1 + w2) / 2)
            for w1, w2 in zip(weights1, weights2)
        ]

        # Make sure the relative errors aren't too small, and at least one
        # layer is very different. Relative error is a normalized measure of
        # difference. I consider rel_error < 0.1 to be a good approximation,
        # and rel_error > 0.9 to be completely different.
        assert all(0.01 < x for x in rel_errors)
        assert any(x > 0.5 for x in rel_errors)
        # the rel_error is often higher than 0.5 but the tests are random


def test_history():
    """Test that history_'s keys are strings and values are lists.
    """
    data = load_boston()
    X, y = data.data[:100], data.target[:100]
    estimator = KerasRegressor(build_fn=dynamic_regressor, model__hidden_layer_sizes=[])

    estimator.partial_fit(X, y)

    assert isinstance(estimator.history_, dict)
    assert all(isinstance(k, str) for k in estimator.history_.keys())
    assert all(isinstance(v, list) for v in estimator.history_.values())


def test_compile_model_from_params():
    """Tests that if build_fn returns an un-compiled model,
    the __init__ parameters will be used to compile it
    and that if build_fn returns a compiled model
    it is not re-compiled.
    """
    # Load data
    data = load_boston()
    X, y = data.data[:100], data.target[:100]

    losses = ("mean_squared_error", "mean_absolute_error")

    # build_fn that does not compile
    def build_fn(compile_with_loss=None):
        model = Sequential()
        model.add(keras.layers.Dense(X.shape[1], input_shape=(X.shape[1],)))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.Activation("linear"))
        if compile_with_loss:
            model.compile(loss=compile_with_loss)
        return model

    for loss in losses:
        estimator = KerasRegressor(
            build_fn=build_fn,
            loss=loss,
            # compile_with_loss=None returns an un-compiled model
            compile_with_loss=None,
        )
        estimator.fit(X, y)
        assert estimator.model_.loss.__name__ == loss

    for myloss in losses:
        estimator = KerasRegressor(
            build_fn=build_fn,
            loss="binary_crossentropy",
            # compile_with_loss != None overrides loss
            compile_with_loss=myloss,
        )
        estimator.fit(X, y)
        assert estimator.model_.loss == myloss
