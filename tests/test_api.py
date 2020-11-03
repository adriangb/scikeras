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
from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable
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
        optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
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
        clf = KerasClassifier(model=build_fn_clf, hidden_dim=5)
        basic_checks(clf, load_iris)

    def test_classify_inherit_class_build_fn(self):
        """Tests for errors using an inherited class."""

        clf = InheritClassBuildFnClf(model=None, hidden_dim=5)
        basic_checks(clf, load_iris)

    def test_regression_build_fn(self):
        """Tests for errors using KerasRegressor."""
        reg = KerasRegressor(model=build_fn_reg, hidden_dim=5)
        basic_checks(reg, load_boston)

    def test_regression_inherit_class_build_fn(self):
        """Tests for errors using KerasRegressor inherited."""

        reg = InheritClassBuildFnReg(model=None, hidden_dim=5,)
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
    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
        estimator = model(build_fn, epochs=1, model__hidden_layer_sizes=[])
        basic_checks(estimator, loader)

    @pytest.mark.parametrize("config", ["MLPRegressor", "MLPClassifier"])
    def test_pipeline(self, config):
        """Tests compatibility with Scikit-learn's pipeline."""
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1, model__hidden_layer_sizes=[])
        estimator = Pipeline([("s", StandardScaler()), ("e", estimator)])
        basic_checks(estimator, loader)

    @pytest.mark.parametrize(
        "config", ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
    )
    def test_searchcv_init_params(self, config):
        """Tests compatibility with Scikit-learn's hyperparameter search CV."""
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(
            build_fn, epochs=1, validation_split=0.1, model__hidden_layer_sizes=[],
        )
        basic_checks(
            GridSearchCV(estimator, {"model__hidden_layer_sizes": [[], [5]]}), loader,
        )
        basic_checks(
            RandomizedSearchCV(
                estimator,
                {"epochs": [1, 2, 3], "optimizer": ["rmsprop", "sgd"]},
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
        estimator = model(build_fn, epochs=1, model__hidden_layer_sizes=[])
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
        base_estimator = model(build_fn, epochs=1, model__hidden_layer_sizes=[])
        for ensemble in ensembles:
            estimator = ensemble(base_estimator=base_estimator, n_estimators=2)
            basic_checks(estimator, loader)

    @pytest.mark.parametrize("config", ["MLPClassifier"])
    def test_calibratedclassifiercv(self, config):
        """Tests compatibility with Scikit-learn's calibrated classifier CV."""
        loader, _, build_fn, _ = CONFIG[config]
        base_estimator = KerasClassifier(
            build_fn, epochs=1, model__hidden_layer_sizes=[]
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
                "n_outputs_expected_": 1,
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

        estimator = model(model=keras_model)
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
                "n_outputs_expected_": 1,
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

        base_estimator = model(model=keras_model)
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
        model=dynamic_regressor,
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


@register_keras_serializable(name="CustomMetric")
class CustomMetric(metrics_module.MeanAbsoluteError):
    def __reduce__(self):
        return metrics_module.deserialize, (metrics_module.serialize(self),)


class TestPartialFit:
    def test_partial_fit(self):
        data = load_boston()
        X, y = data.data[:100], data.target[:100]
        estimator = KerasRegressor(
            model=dynamic_regressor,
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

    def test_partial_fit_history_metric_names(self):
        data = load_boston()
        X, y = data.data[:100], data.target[:100]
        estimator = KerasRegressor(
            model=dynamic_regressor,
            loss=KerasRegressor.r_squared,
            model__hidden_layer_sizes=[100,],
            metrics=["mse", CustomMetric(name="custom_metric")],
        )
        estimator.partial_fit(X, y)
        # Make custom metric names are preserved
        # and shorthand metric names are saved by their full name
        for _ in range(2):
            estimator = pickle.loads(pickle.dumps(estimator))
            estimator = estimator.partial_fit(X, y)
            assert set(estimator.history_.keys()) == {
                "loss",
                "mean_squared_error",
                "custom_metric",
            }

    def test_partial_fit_history_len(self):
        # history_ records the history from this partial_fit call
        # Make sure for each call to partial_fit a single entry
        # into the history is added
        # As per https://github.com/keras-team/keras/issues/1766,
        # there is no direct measure of epochs
        data = load_boston()
        X, y = data.data[:100], data.target[:100]
        estimator = KerasRegressor(
            model=dynamic_regressor,
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

    def test_partial_fit_single_epoch(self):
        """Test that partial_fit trains for a single epoch,
        independently of what epoch value is passed to the constructor.
        """
        data = load_boston()
        X, y = data.data[:100], data.target[:100]
        epochs = 9
        partial_fit_iter = 4

        estimator = KerasRegressor(
            model=dynamic_regressor,
            loss=KerasRegressor.r_squared,
            model__hidden_layer_sizes=[100,],
            epochs=epochs,
        )

        # Check that each partial_fit call trains for 1 epoch
        for k in range(1, partial_fit_iter):
            estimator = estimator.partial_fit(X, y)
            assert len(estimator.history_["loss"]) == k

        # Check that fit calls still train for the number of
        # epochs specified in the constructor
        estimator = estimator.fit(X, y)
        assert len(estimator.history_["loss"]) == epochs

    @pytest.mark.parametrize("warm_start", [True, False])
    def test_current_epoch_property(self, warm_start):
        """Test the public current_epoch property
        that tracks the overall training epochs.

        The warm_start parameter should have
        NO impact on this behavior.
        """
        data = load_boston()
        X, y = data.data[:100], data.target[:100]
        epochs = 2
        partial_fit_iter = 4

        estimator = KerasRegressor(
            model=dynamic_regressor,
            loss=KerasRegressor.r_squared,
            model__hidden_layer_sizes=[100,],
            epochs=epochs,
            warm_start=warm_start,
        )

        # Check that each partial_fit call trains for 1 epoch
        for k in range(1, partial_fit_iter):
            estimator = estimator.partial_fit(X, y)
            assert estimator.current_epoch == k

        # Check that fit calls still train for the number of
        # epochs specified in the constructor
        estimator = estimator.fit(X, y)
        assert estimator.current_epoch == epochs

    @pytest.mark.parametrize(
        "config", ["CNNClassifier", "CNNClassifierF"],
    )
    def test_pf_pickle_pf(self, config):
        loader, model, build_fn, _ = CONFIG[config]
        clf = model(build_fn, epochs=1, model__hidden_layer_sizes=[])
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

    def test_partial_fit_classes_param(self):
        """Test use of `partial_fit` with the `classes` parameter
        and incomplete classes in the first pass.
        """
        clf = KerasClassifier(
            model=dynamic_classifier,
            loss="sparse_categorical_crossentropy",
            model__hidden_layer_sizes=[100,],
        )
        X1 = np.array([[1, 2, 3], [4, 5, 6]]).T
        y1 = np.array([1, 2, 2])
        X2 = X1
        y2 = np.array([2, 3, 3])
        classes = np.unique(np.concatenate([y1, y2]))
        clf.partial_fit(X=X1, y=y1, classes=classes)
        clf.score(X1, y1)
        clf.score(X2, y2)
        clf.partial_fit(X=X2, y=y2)
        clf.score(X1, y1)
        clf.score(X2, y2)


def force_compile_shorthand(hidden_layer_sizes, meta, compile_kwargs, params):
    model = dynamic_regressor(
        hidden_layer_sizes=hidden_layer_sizes, meta=meta, compile_kwargs=compile_kwargs
    )
    model.compile(
        optimizer=compile_kwargs["optimizer"],
        loss=compile_kwargs["loss"],
        metrics=params["metrics"],
    )
    return model


class TestHistory:
    def test_history(self):
        """Test that history_'s keys are strings and values are lists.
        """
        data = load_boston()
        X, y = data.data[:100], data.target[:100]
        estimator = KerasRegressor(
            model=dynamic_regressor, model__hidden_layer_sizes=[]
        )

        estimator.partial_fit(X, y)

        assert isinstance(estimator.history_, dict)
        assert all(isinstance(k, str) for k in estimator.history_.keys())
        assert all(isinstance(v, list) for v in estimator.history_.values())

    def test_partial_fit_shorthand_metric_name(self):
        """Test that metrics get stored in the `history_` attribute
        by their long name (and not shorthand) even if the user
        compiles their model with a shorthand name.
        """
        est = KerasRegressor(
            model=force_compile_shorthand,
            loss=KerasRegressor.r_squared,
            model__hidden_layer_sizes=(100,),
            metrics=["mae"],  # shorthand
        )
        X, y = load_boston(return_X_y=True)
        X = X[:100]
        y = y[:100]
        est.fit(X, y)
        assert "mae" not in est.history_ and "mean_absolute_error" in est.history_


def test_compile_model_from_params():
    """Tests that if build_fn returns an un-compiled model,
    the __init__ parameters will be used to compile it
    and that if build_fn returns a compiled model
    it is not re-compiled.
    """
    # Load data
    data = load_boston()
    X, y = data.data[:100], data.target[:100]

    other_loss = losses_module.MeanAbsoluteError

    # build_fn that does not compile
    def build_fn(my_loss=None):
        model = Sequential()
        model.add(keras.layers.Dense(X.shape[1], input_shape=(X.shape[1],)))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.Activation("linear"))
        if my_loss is not None:
            model.compile(loss=my_loss)
        return model

    # Calling with loss=None (or other default)
    # and compiling within build_fn must work
    loss_obj = other_loss()
    estimator = KerasRegressor(
        model=build_fn,
        # compile_with_loss != None overrides loss
        my_loss=loss_obj,
    )
    estimator.fit(X, y)
    assert estimator.model_.loss is loss_obj

    # Passing a value for loss AND compiling with
    # the SAME loss should succeed, and the final loss
    # should be the user-supplied loss
    loss_obj = other_loss()
    estimator = KerasRegressor(
        model=build_fn,
        loss=other_loss(),
        # compile_with_loss != None overrides loss
        my_loss=loss_obj,
    )
    estimator.fit(X, y)
    assert estimator.model_.loss is loss_obj

    # Passing a non-default value for loss AND compiling with
    # a DIFFERENT loss should raise a ValueError
    loss_obj = other_loss()
    estimator = KerasRegressor(
        model=build_fn,
        loss=losses_module.CosineSimilarity(),
        # compile_with_loss != None overrides loss
        my_loss=loss_obj,
    )
    with pytest.raises(ValueError, match=" but model compiled with "):
        estimator.fit(X, y)

    # The ValueError should appear even if the default is != None
    class DefaultLossNotNone(KerasRegressor):
        def __init__(self, *args, loss=losses_module.CosineSimilarity(), **kwargs):
            super().__init__(*args, **kwargs, loss=loss)

    loss_obj = other_loss()
    estimator = DefaultLossNotNone(model=build_fn, my_loss=loss_obj,)
    estimator.fit(X, y)
    assert estimator.model_.loss is loss_obj

    loss_obj = other_loss()
    estimator = DefaultLossNotNone(
        model=build_fn, loss=losses_module.CategoricalHinge(), my_loss=loss_obj,
    )
    with pytest.raises(ValueError, match=" but model compiled with "):
        estimator.fit(X, y)


def test_subclassed_model_no_params():
    """Test that we can define a subclassed model with no `__init__` params
    (i.e., a fully self-contined sklearn style estimator)
    and that wrappers do not fail on introspection of the child class'
    `__init__`.
    """

    class MLPClassifier(KerasClassifier):
        def __init__(self):
            super().__init__()

        def _keras_build_fn(self):
            model = Sequential()
            model.add(Dense(2, activation="relu", input_shape=(2,)))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(loss="binary_crossentropy")
            return model

    clf = MLPClassifier()
    X = np.random.random(size=(100, 2))
    y = np.random.randint(0, 1, size=(100,))
    clf.fit(X, y)
    clf.predict(X)
