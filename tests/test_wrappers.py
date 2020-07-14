"""Tests for Scikit-learn API wrapper."""


import pickle

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_boston, load_digits, load_iris
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from tensorflow.python import keras
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
)
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical

from scikeras import wrappers
from scikeras.wrappers import KerasClassifier, KerasRegressor

INPUT_DIM = 5
HIDDEN_DIM = 5
TRAIN_SAMPLES = 10
TEST_SAMPLES = 5
NUM_CLASSES = 2
BATCH_SIZE = 5
EPOCHS = 1


def build_fn_clf(hidden_dim):
    """Builds a Sequential based classifier."""
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(INPUT_DIM, input_shape=(INPUT_DIM,)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(hidden_dim))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(NUM_CLASSES))
    model.add(keras.layers.Activation("softmax"))
    model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def assert_classification_works(clf):
    """Checks a classification task for errors."""
    np.random.seed(42)
    (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES,
    )

    clf.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    score = clf.score(x_train, y_train, batch_size=BATCH_SIZE)
    assert np.isscalar(score) and np.isfinite(score)

    preds = clf.predict(x_test, batch_size=BATCH_SIZE)
    assert preds.shape == (TEST_SAMPLES,)
    for prediction in np.unique(preds):
        assert prediction in range(NUM_CLASSES)

    proba = clf.predict_proba(x_test, batch_size=BATCH_SIZE)
    assert proba.shape == (TEST_SAMPLES, NUM_CLASSES)
    assert np.allclose(np.sum(proba, axis=1), np.ones(TEST_SAMPLES))


def build_fn_reg(hidden_dim):
    """Builds a Sequential based regressor."""
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(INPUT_DIM, input_shape=(INPUT_DIM,)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(hidden_dim))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("linear"))
    model.compile(
        optimizer="sgd", loss="mean_absolute_error", metrics=["accuracy"]
    )
    return model


def assert_regression_works(reg):
    """Checks a regression task for errors."""
    np.random.seed(42)
    (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES,
    )

    reg.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    score = reg.score(x_train, y_train, batch_size=BATCH_SIZE)
    assert np.isscalar(score) and np.isfinite(score)

    preds = reg.predict(x_test, batch_size=BATCH_SIZE)
    assert preds.shape == (TEST_SAMPLES,)


class TestBasicAPI:
    """Tests basic functionality."""

    def test_classify_build_fn(self):
        """Tests a classification task for errors."""
        clf = wrappers.KerasClassifier(
            build_fn=build_fn_clf,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

        assert_classification_works(clf)

    def test_classify_class_build_fn(self):
        """Tests for errors using a class implementing __call__."""

        class ClassBuildFnClf:
            def __call__(self, hidden_dim):
                return build_fn_clf(hidden_dim)

        clf = wrappers.KerasClassifier(
            build_fn=ClassBuildFnClf(),
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

        assert_classification_works(clf)

    def test_classify_inherit_class_build_fn(self):
        """Tests for errors using an inherited class."""

        class InheritClassBuildFnClf(wrappers.KerasClassifier):
            def __call__(self, hidden_dim):
                return build_fn_clf(hidden_dim)

        clf = InheritClassBuildFnClf(
            build_fn=None,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

        assert_classification_works(clf)

    def test_regression_build_fn(self):
        """Tests for errors using KerasRegressor."""
        reg = wrappers.KerasRegressor(
            build_fn=build_fn_reg,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

        assert_regression_works(reg)

    def test_regression_class_build_fn(self):
        """Tests for errors using KerasRegressor implementing __call__."""

        class ClassBuildFnReg:
            def __call__(self, hidden_dim):
                return build_fn_reg(hidden_dim)

        reg = wrappers.KerasRegressor(
            build_fn=ClassBuildFnReg(),
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

        assert_regression_works(reg)

    def test_regression_inherit_class_build_fn(self):
        """Tests for errors using KerasRegressor inherited."""

        class InheritClassBuildFnReg(wrappers.KerasRegressor):
            def __call__(self, hidden_dim):
                return build_fn_reg(hidden_dim)

        reg = InheritClassBuildFnReg(
            build_fn=None,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

        assert_regression_works(reg)


def load_digits8x8():
    """Load image 8x8 dataset."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    # Convert NCHW to NHWC
    # Convert back to numpy or sklearn funcs (GridSearchCV, etc.) WILL fail
    data.data = np.transpose(data.data, [0, 2, 3, 1])
    K.set_image_data_format("channels_last")
    return data


def check(estimator, loader):
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
    assert True


def build_fn_regs(X, n_outputs_, hidden_layer_sizes=None, n_classes_=None):
    """Dynamically build regressor."""
    if hidden_layer_sizes is None:
        hidden_layer_sizes = []
    model = Sequential()
    model.add(Dense(X.shape[1], activation="relu", input_shape=X.shape[1:]))
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation="relu"))
    model.add(Dense(n_outputs_))
    model.compile("adam", loss="mean_squared_error")
    return model


def build_fn_clss(X, n_outputs_, hidden_layer_sizes=None, n_classes_=None):
    """Dynamically build classifier."""
    if hidden_layer_sizes is None:
        hidden_layer_sizes = []
    model = Sequential()
    model.add(Dense(X.shape[1], activation="relu", input_shape=X.shape[1:]))
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation="relu"))
    model.add(Dense(1, activation="softmax"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_fn_clscs(X, n_outputs_, hidden_layer_sizes=None, n_classes_=None):
    """Dynamically build functional API regressor."""
    if hidden_layer_sizes is None:
        hidden_layer_sizes = []
    model = Sequential()
    model.add(Conv2D(3, (3, 3), input_shape=X.shape[1:]))
    model.add(Flatten())
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation="relu"))
    model.add(Dense(n_classes_, activation="softmax"))
    model.compile(
        "adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_fn_clscf(X, n_outputs_, hidden_layer_sizes=None, n_classes_=None):
    """Dynamically build functional API classifier."""
    if hidden_layer_sizes is None:
        hidden_layer_sizes = []
    x = Input(shape=X.shape[1:])
    z = Conv2D(3, (3, 3))(x)
    z = Flatten()(z)
    for size in hidden_layer_sizes:
        z = Dense(size, activation="relu")(z)
    y = Dense(n_classes_, activation="softmax")(z)
    model = Model(inputs=x, outputs=y)
    model.compile(
        "adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


CONFIG = {
    "MLPRegressor": (
        load_boston,
        KerasRegressor,
        build_fn_regs,
        (BaggingRegressor, AdaBoostRegressor),
    ),
    "MLPClassifier": (
        load_iris,
        KerasClassifier,
        build_fn_clss,
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
        "config",
        ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
    )
    def test_standalone(self, config):
        """Tests standalone estimator."""
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1)
        check(estimator, loader)

    @pytest.mark.parametrize("config", ["MLPRegressor", "MLPClassifier"])
    def test_pipeline(self, config):
        """Tests compatibility with Scikit-learn's pipeline."""
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1)
        estimator = Pipeline([("s", StandardScaler()), ("e", estimator)])
        check(estimator, loader)

    @pytest.mark.parametrize(
        "config",
        ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
    )
    def test_searchcv(self, config):
        """Tests compatibility with Scikit-learn's hyperparameter search CV."""
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(
            build_fn, epochs=1, validation_split=0.1, hidden_layer_sizes=[]
        )
        check(
            GridSearchCV(estimator, {"hidden_layer_sizes": [[], [5]]}), loader,
        )
        check(
            RandomizedSearchCV(
                estimator, {"epochs": np.random.randint(1, 5, 2)}, n_iter=2,
            ),
            loader,
        )

    @pytest.mark.parametrize("config", ["MLPRegressor", "MLPClassifier"])
    def test_ensemble(self, config):
        """Tests compatibility with Scikit-learn's ensembles."""
        loader, model, build_fn, ensembles = CONFIG[config]
        base_estimator = model(build_fn, epochs=1)
        for ensemble in ensembles:
            estimator = ensemble(base_estimator=base_estimator, n_estimators=2)
            check(estimator, loader)

    @pytest.mark.parametrize("config", ["MLPClassifier"])
    def test_calibratedclassifiercv(self, config):
        """Tests compatibility with Scikit-learn's calibrated classifier CV."""
        loader, _, build_fn, _ = CONFIG[config]
        base_estimator = KerasClassifier(build_fn, epochs=1)
        estimator = CalibratedClassifierCV(base_estimator=base_estimator, cv=5)
        check(estimator, loader)


class SentinalCallback(keras.callbacks.Callback):
    """
    Callback class that sets an internal value once it's been acessed.
    """

    called = 0

    def on_train_begin(self, logs=None):
        """Increments counter."""
        self.called += 1


class ClassWithCallback(wrappers.KerasClassifier):
    """Must be defined at top level to be picklable.
    """

    def __init__(self, **sk_params):
        self.callbacks = [SentinalCallback()]
        super().__init__(**sk_params)

    def __call__(self, hidden_dim):
        return build_fn_clf(hidden_dim)


class TestCallbacks:
    """Tests use of Callbacks."""

    @pytest.mark.parametrize(
        "config",
        ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
    )
    def test_callbacks_passed_as_arg(self, config):
        """Tests estimators created passing a callback to __init__."""
        loader, model, build_fn, _ = CONFIG[config]
        callback = SentinalCallback()
        estimator = model(build_fn, epochs=1, callbacks=[callback])
        # check that callback did not break estimator
        check(estimator, loader)
        # check that callback is preserved after pickling
        data = loader()
        X, y = data.data[:100], data.target[:100]
        estimator.fit(X, y)
        assert estimator.callbacks[0].called != SentinalCallback.called
        old_callback = estimator.callbacks[0]
        deserialized_estimator = pickle.loads(pickle.dumps(estimator))
        assert (
            deserialized_estimator.callbacks[0].called == old_callback.called
        )

    def test_callbacks_inherit(self):
        """Test estimators that inherit from KerasClassifier and implement
        their own callbacks in their __init___.
        """
        clf = ClassWithCallback(
            hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE, epochs=EPOCHS
        )

        assert_classification_works(clf)
        assert clf.callbacks[0].called != SentinalCallback.called
        serialized_estimator = pickle.dumps(clf)
        deserialized_estimator = pickle.loads(serialized_estimator)
        assert (
            deserialized_estimator.callbacks[0].called
            == clf.callbacks[0].called
        )
        assert_classification_works(deserialized_estimator)


class TestSampleWeights:
    """Tests involving the sample_weight parameter.
         TODO: fix warning regarding sample_weight shape coercing.
    """

    @staticmethod
    def check_sample_weights_work(estimator):
        """Checks that using the parameter sample_weight does not cause
        issues (it does not check if the parameter actually works as intended).
        """
        (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        s_w_train = np.random.randn(x_train.shape[0])

        # check that we can train with sample weights
        # TODO: how do we reliably check the effect of training with
        # sample_weights?
        estimator.fit(x_train, y_train, sample_weight=s_w_train)
        estimator.predict(x_test)

        # now train with no sample weights, test scoring
        estimator.fit(x_train, y_train, sample_weight=None)
        # re-use training data to try to get score > 0
        score_n_w = estimator.score(x_train, y_train)
        score_w = estimator.score(x_train, y_train, sample_weight=s_w_train)
        # check that sample weights did *something*
        try:
            np.testing.assert_array_almost_equal(score_n_w, score_w)
        except AssertionError:
            return

        raise AssertionError("`sample_weight` seemed to have no effect.")

    def test_classify_build_fn(self):
        clf = wrappers.KerasClassifier(
            build_fn=build_fn_clf,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )
        self.check_sample_weights_work(clf)

    def test_reg_build_fn(self):
        clf = wrappers.KerasRegressor(
            build_fn=build_fn_reg,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )
        self.check_sample_weights_work(clf)


def dynamic_classifier(X, cls_type_, n_classes_, keras_expected_n_ouputs_):
    """Creates a basic MLP classifier dynamically choosing binary/multiclass
    classification loss and ouput activations.
    """
    n_features = X.shape[1]

    inp = Input(shape=(n_features,))

    x1 = Dense(100)(inp)

    if cls_type_ == "binary":
        loss = "binary_crossentropy"
        out = [Dense(1, activation="sigmoid")(x1)]
    elif cls_type_ == "multilabel-indicator":
        loss = "binary_crossentropy"
        out = [
            Dense(1, activation="sigmoid")(x1)
            for _ in range(keras_expected_n_ouputs_)
        ]
    elif cls_type_ == "multiclass-multioutput":
        loss = "binary_crossentropy"
        out = [Dense(n, activation="softmax")(x1) for n in n_classes_]
    else:
        # multiclass
        loss = "categorical_crossentropy"
        out = [Dense(n_classes_, activation="softmax")(x1)]

    model = Model([inp], out)

    model.compile(optimizer="adam", loss=loss)

    return model


def dynamic_regressor(X, n_outputs_):
    """Creates a basic MLP regressor dynamically.
    """
    n_features = X.shape[1]

    inp = Input(shape=(n_features,))

    x1 = Dense(100)(inp)

    out = [Dense(n_outputs_)(x1)]

    model = Model([inp], out)

    model.compile(
        optimizer="adam", loss=wrappers.KerasRegressor.root_mean_squared_error,
    )
    return model


class FullyCompliantClassifier(wrappers.KerasClassifier):
    """A classifier that sets all parameters in __init__ and nothing more."""

    def __init__(
        self, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE, epochs=EPOCHS
    ):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        return super().__init__()

    def __call__(self, X, cls_type_, n_classes_, keras_expected_n_ouputs_):
        return dynamic_classifier(
            X, cls_type_, n_classes_, keras_expected_n_ouputs_
        )


class FullyCompliantRegressor(wrappers.KerasRegressor):
    """A classifier that sets all parameters in __init__ and nothing more."""

    def __init__(
        self, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE, epochs=EPOCHS
    ):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        return super().__init__()

    def __call__(self, X, n_outputs_):
        return dynamic_regressor(X, n_outputs_)


class TestFullyCompliantWrappers:
    """Tests wrappers that fully comply with the Scikit-Learn
        API by not using kwargs. Testing done with Scikit-Learn's
        internal model validation tool
    """

    @parametrize_with_checks([FullyCompliantClassifier()])
    def test_fully_compliant_classifier(self, estimator, check):
        check(estimator)

    @parametrize_with_checks([FullyCompliantRegressor()])
    def test_fully_compliant_regressor(self, estimator, check):
        check(estimator)


class TestOutputShapes:
    """Tests that compare output shapes to `MLPClassifier` from sklearn to
         check that ouput shapes respect sklearn API.
    """

    @classmethod
    def setup_class(cls):
        cls.keras_clf = KerasClassifier(build_fn=dynamic_classifier)
        cls.sklearn_clf = MLPClassifier()

    def test_1d_multiclass(self):
        """Compares KerasClassifier prediction output shape against
        sklearn.neural_net.MPLClassifier for 1D multi-class (n_samples,).
        """
        # crate 1D multiclass labels
        (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=4,
        )
        self.keras_clf.fit(X=x_train, y=y_train)
        self.sklearn_clf.fit(X=x_train, y=y_train)
        y_pred_keras = self.keras_clf.predict(X=x_test)
        y_pred_sklearn = self.sklearn_clf.predict(X=x_test)
        assert y_pred_keras.shape == y_pred_sklearn.shape
        y_pred_prob_keras = self.keras_clf.predict_proba(X=x_test)
        y_pred_prob_sklearn = self.sklearn_clf.predict_proba(X=x_test)
        assert y_pred_prob_keras.shape == y_pred_prob_sklearn.shape

    def test_2d_multiclass(self):
        """Compares KerasClassifier prediction output shape against
        sklearn.neural_net.MPLClassifier for 2D multi-class (n_samples,1).
        """
        # crate 2D multiclass labels
        (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=4,
        )
        y_train = y_train.reshape(-1, 1)
        self.keras_clf.fit(X=x_train, y=y_train)
        self.sklearn_clf.fit(X=x_train, y=y_train)
        y_pred_keras = self.keras_clf.predict(X=x_test)
        y_pred_sklearn = self.sklearn_clf.predict(X=x_test)
        assert y_pred_keras.shape == y_pred_sklearn.shape
        y_pred_prob_keras = self.keras_clf.predict_proba(X=x_test)
        y_pred_prob_sklearn = self.sklearn_clf.predict_proba(X=x_test)
        assert y_pred_prob_keras.shape == y_pred_prob_sklearn.shape

    def test_1d_binary(self):
        """Compares KerasClassifier prediction output shape against
        sklearn.neural_net.MPLClassifier for binary (n_samples,).
        """
        # create 1D binary labels
        (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=2,
        )
        self.keras_clf.fit(X=x_train, y=y_train)
        self.sklearn_clf.fit(X=x_train, y=y_train)
        y_pred_keras = self.keras_clf.predict(X=x_test)
        y_pred_sklearn = self.sklearn_clf.predict(X=x_test)
        assert y_pred_keras.shape == y_pred_sklearn.shape
        y_pred_prob_keras = self.keras_clf.predict_proba(X=x_test)
        y_pred_prob_sklearn = self.sklearn_clf.predict_proba(X=x_test)
        assert y_pred_prob_keras.shape == y_pred_prob_sklearn.shape

    def test_2d_binary(self):
        """Compares KerasClassifier prediction output shape against
        sklearn.neural_net.MPLClassifier for 2D binary (n_samples,1).
        """
        # create 2D binary labels
        (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=2,
        )
        y_train = y_train.reshape(-1, 1)
        self.keras_clf.fit(X=x_train, y=y_train)
        self.sklearn_clf.fit(X=x_train, y=y_train)
        y_pred_keras = self.keras_clf.predict(X=x_test)
        y_pred_sklearn = self.sklearn_clf.predict(X=x_test)
        assert y_pred_keras.shape == y_pred_sklearn.shape
        y_pred_prob_keras = self.keras_clf.predict_proba(X=x_test)
        y_pred_prob_sklearn = self.sklearn_clf.predict_proba(X=x_test)
        assert y_pred_prob_keras.shape == y_pred_prob_sklearn.shape


class TestPrebuiltModel:
    """Tests using a prebuilt model instance."""

    @pytest.mark.parametrize(
        "config",
        ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
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
            keras_model = build_fn(
                X=x_train, n_classes_=n_classes_, n_outputs_=1
            )
        else:
            keras_model = build_fn(X=x_train, n_outputs_=1)

        estimator = model(build_fn=keras_model)
        check(estimator, loader)

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
            keras_model = build_fn(
                X=x_train, n_classes_=n_classes_, n_outputs_=1
            )
        else:
            keras_model = build_fn(X=x_train, n_outputs_=1)

        base_estimator = model(build_fn=keras_model)
        for ensemble in ensembles:
            estimator = ensemble(base_estimator=base_estimator, n_estimators=2)
            check(estimator, loader)


class FunctionalAPIMultiInputClassifier(KerasClassifier):
    """Tests Functional API Classifier with 2 inputs.
    """

    def __call__(self, X, n_classes_):
        inp1 = Input((1,))
        inp2 = Input((3,))

        x1 = Dense(100)(inp1)
        x2 = Dense(100)(inp2)

        x3 = Concatenate(axis=-1)([x1, x2])

        cat_out = Dense(n_classes_, activation="softmax")(x3)

        model = Model([inp1, inp2], [cat_out])
        losses = ["categorical_crossentropy"]
        model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

        return model

    @staticmethod
    def _pre_process_X(X):
        """To support multiple inputs, a custom method must be defined.
        """
        return [X[:, 0], X[:, 1:4]], dict()


class FunctionalAPIMultiOutputClassifier(KerasClassifier):
    """Tests Functional API Classifier with 2 outputs of different type.
    """

    def __call__(self, X, n_classes_):
        inp = Input((4,))

        x1 = Dense(100)(inp)

        binary_out = Dense(1, activation="sigmoid")(x1)
        cat_out = Dense(n_classes_[1], activation="softmax")(x1)

        model = Model([inp], [binary_out, cat_out])
        losses = ["binary_crossentropy", "categorical_crossentropy"]
        model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

        return model

    def score(self, X, y):
        """Taken from sklearn.multiouput.MultiOutputClassifier
        """
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))


class FunctionAPIMultiLabelClassifier(KerasClassifier):
    """Tests Functional API Classifier with multiple binary outputs.
    """

    def __call__(self, X, n_outputs_):
        inp = Input((4,))

        x1 = Dense(100)(inp)

        outputs = []
        for _ in range(n_outputs_):
            # simulate multiple binary classification outputs
            # in reality, these would come from different nodes
            outputs.append(Dense(1, activation="sigmoid")(x1))

        model = Model(inp, outputs)
        losses = "binary_crossentropy"
        model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

        return model


class FunctionAPIMultiOutputRegressor(KerasRegressor):
    """Tests Functional API Regressor with multiple outputs.
    """

    def __call__(self, X, n_outputs_):
        inp = Input((INPUT_DIM,))

        x1 = Dense(100)(inp)

        outputs = [Dense(n_outputs_)(x1)]

        model = Model([inp], outputs)
        losses = "mean_squared_error"
        model.compile(optimizer="adam", loss=losses, metrics=["mse"])

        return model


@keras.utils.generic_utils.register_keras_serializable()
class CustomLoss(keras.losses.MeanSquaredError):
    """Dummy custom loss."""


@keras.utils.generic_utils.register_keras_serializable()
class CustomModelRegistered(Model):
    """Dummy custom Model subclass that is registered to be serializable."""


class CustomModelUnregistered(Model):
    """Dummy custom Model subclass that is not registered to be
    serializable."""


def build_fn_regs_custom_loss(X, n_outputs_, hidden_layer_sizes=None):
    """Build regressor with subclassed loss function."""
    if hidden_layer_sizes is None:
        hidden_layer_sizes = []
    model = Sequential()
    model.add(Dense(X.shape[1], activation="relu", input_shape=X.shape[1:]))
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation="relu"))
    model.add(Dense(n_outputs_))
    model.compile("adam", loss=CustomLoss())
    return model


def build_fn_regs_custom_model_reg(X, n_outputs_, hidden_layer_sizes=None):
    """Build regressor with subclassed Model registered as serializable."""
    if hidden_layer_sizes is None:
        hidden_layer_sizes = []
    x = Input(shape=X.shape[1])
    z = Dense(X.shape[1], activation="relu")(x)
    for size in hidden_layer_sizes:
        z = Dense(size, activation="relu")(z)
    y = Dense(n_outputs_, activation="linear")(z)
    model = CustomModelRegistered(inputs=x, outputs=y)
    model.compile("adam", loss="mean_squared_error")
    return model


def build_fn_regs_custom_model_unreg(X, n_outputs_, hidden_layer_sizes=None):
    """Build regressor with subclassed Model not registered as serializable."""
    if hidden_layer_sizes is None:
        hidden_layer_sizes = []
    x = Input(shape=X.shape[1])
    z = Dense(X.shape[1], activation="relu")(x)
    for size in hidden_layer_sizes:
        z = Dense(size, activation="relu")(z)
    y = Dense(n_outputs_, activation="linear")(z)
    model = CustomModelUnregistered(inputs=x, outputs=y)
    model.compile("adam", loss="mean_squared_error")
    return model


class TestSerializeCustomLayers:
    """Tests serializing custom layers."""

    def test_custom_loss_function(self):
        """Test that a registered subclassed Model can be serialized."""
        estimator = KerasRegressor(build_fn=build_fn_regs_custom_loss)
        check(estimator, load_boston)

    def test_custom_model_registered(self):
        """Test that a registered subclassed loss function can be
        serialized."""
        estimator = KerasRegressor(build_fn=build_fn_regs_custom_model_reg)
        check(estimator, load_boston)

    def test_custom_model_unregistered(self):
        """Test that an unregistered subclassed Model raises an error."""
        estimator = KerasRegressor(build_fn=build_fn_regs_custom_model_unreg)
        with pytest.raises(ValueError):
            check(estimator, load_boston)


class TestScoring:
    """Tests scoring methods.
    """

    def test_scoring_r2(self):
        """Test custom R^2 implementation against scikit-learn's."""
        n_samples = 50

        datasets = []
        y_true = np.arange(n_samples, dtype=float)
        y_pred = y_true + 1
        datasets.append((y_true.reshape(-1, 1), y_pred.reshape(-1, 1)))
        y_true = np.random.random_sample(size=y_true.shape)
        y_pred = np.random.random_sample(size=y_true.shape)
        datasets.append((y_true.reshape(-1, 1), y_pred.reshape(-1, 1)))

        def keras_backend_r2(y_true, y_pred):
            """Wrap Keras operations to numpy."""
            y_true = convert_to_tensor(y_true)
            y_pred = convert_to_tensor(y_pred)
            return KerasRegressor.root_mean_squared_error(
                y_true, y_pred
            ).numpy()

        score_functions = (keras_backend_r2,)
        correct_scorer = sklearn_r2_score

        for (y_true, y_pred) in datasets:
            for f in score_functions:
                np.testing.assert_almost_equal(
                    f(y_true, y_pred),
                    correct_scorer(y_true, y_pred),
                    decimal=5,
                )


class TestMultiInputOutput:
    """Tests involving multiple inputs / outputs.
    """

    def test_multi_input(self):
        """Tests custom multi-input Keras model.
        """
        clf = FunctionalAPIMultiInputClassifier()
        (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(4,),
            num_classes=3,
        )

        clf.fit(x_train, y_train)
        clf.predict(x_test)
        clf.score(x_train, y_train)

    def test_multi_output(self):
        """Compares to scikit-learn RandomForestClassifier classifier.
        """
        clf_keras = FunctionalAPIMultiOutputClassifier()
        clf_sklearn = RandomForestClassifier()

        # generate data
        X = np.random.rand(10, 4)
        y1 = np.random.randint(0, 2, size=(10, 1))
        y2 = np.random.randint(0, 11, size=(10, 1))
        y = np.hstack([y1, y2])

        clf_keras.fit(X, y)
        y_wrapper = clf_keras.predict(X)
        clf_keras.score(X, y)

        clf_sklearn.fit(X, y)
        y_sklearn = clf_sklearn.predict(X)

        assert y_sklearn.shape == y_wrapper.shape

    def test_multi_label_clasification(self):
        """Compares to scikit-learn RandomForestClassifier classifier.
        """
        clf_keras = FunctionAPIMultiLabelClassifier()
        clf_sklearn = RandomForestClassifier()
        # taken from https://scikit-learn.org/stable/modules/multiclass.html
        y = np.array([[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]])
        y = MultiLabelBinarizer().fit_transform(y)

        (x_train, _), (_, _) = testing_utils.get_test_data(
            train_samples=y.shape[0],
            test_samples=0,
            input_shape=(4,),
            num_classes=3,
        )

        clf_keras.fit(x_train, y)
        y_pred_keras = clf_keras.predict(x_train)
        clf_keras.score(x_train, y)

        clf_sklearn.fit(x_train, y)
        y_pred_sklearn = clf_sklearn.predict(x_train)
        clf_sklearn.score(x_train, y)

        assert y_pred_keras.shape == y_pred_sklearn.shape

    def test_multi_output_regression(self):
        """Compares to scikit-learn RandomForestRegressor.
        """
        reg_keras = FunctionAPIMultiOutputRegressor()
        reg_sklearn = RandomForestRegressor()
        # taken from https://scikit-learn.org/stable/modules/multiclass.html
        (X, _), (_, _) = testing_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y = np.random.random_sample(size=(TRAIN_SAMPLES, NUM_CLASSES))

        reg_keras.fit(X, y)
        y_pred_keras = reg_keras.predict(X)
        reg_keras.score(X, y)

        reg_sklearn.fit(X, y)
        y_pred_sklearn = reg_sklearn.predict(X)
        reg_sklearn.score(X, y)

        assert y_pred_keras.shape == y_pred_sklearn.shape

    def test_incompatible_output_dimensions(self):
        """Compares to the scikit-learn RandomForestRegressor classifier.
        """
        # create dataset with 4 outputs
        X = np.random.rand(10, 20)
        y = np.random.randint(low=0, high=3, size=(10, 4))

        # create a model with 2 outputs
        def build_fn_clf():
            """Builds a Sequential based classifier."""
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(20, input_shape=(20,)))
            model.add(keras.layers.Activation("relu"))
            model.add(keras.layers.Dense(np.unique(y).size))
            model.add(keras.layers.Activation("softmax"))
            model.compile(
                optimizer="sgd",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model

        clf = wrappers.KerasClassifier(build_fn=build_fn_clf)

        with pytest.raises(RuntimeError):
            clf.fit(X, y)


class TestInvalidBuildFn:
    """Tests various error cases for BuildFn.
    """

    def test_invalid_build_fn(self):
        clf = wrappers.KerasClassifier(build_fn="invalid")
        with pytest.raises(TypeError):
            clf.fit(np.array([[0]]), np.array([0]))

    def test_no_build_fn(self):
        class NoBuildFn(wrappers.KerasClassifier):
            pass

        clf = NoBuildFn()

        with pytest.raises(ValueError):
            assert_classification_works(clf)

    def test_call_and_build_fn_function(self):
        class Clf(wrappers.KerasClassifier):
            def __call__(self, hidden_dim):
                return build_fn_clf(hidden_dim)

        def dummy_func():
            return None

        clf = Clf(build_fn=dummy_func,)

        with pytest.raises(ValueError):
            assert_classification_works(clf)

    def test_call_and_invalid_build_fn_class(self):
        class Clf(wrappers.KerasClassifier):
            def __call__(self, hidden_dim):
                return build_fn_clf(hidden_dim)

        class DummyBuildClass:
            def __call__(self, hidden_dim):
                return build_fn_clf(hidden_dim)

        clf = Clf(build_fn=DummyBuildClass(),)

        with pytest.raises(ValueError):
            assert_classification_works(clf)


class TestUnfitted:
    """Tests for appropriate error on unfitted models.
    """

    def test_classify_build_fn(self):
        """Tests a classification task for errors."""
        clf = wrappers.KerasClassifier(
            build_fn=build_fn_clf,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

        X = np.random.rand(10, 20)

        with pytest.raises(NotFittedError):
            clf.predict(X)
        with pytest.raises(NotFittedError):
            clf.predict_proba(X)

    def test_regression_build_fn(self):
        """Tests for errors using KerasRegressor."""
        reg = wrappers.KerasRegressor(
            build_fn=build_fn_reg,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

        # create dataset
        X = np.random.rand(10, 20)

        with pytest.raises(NotFittedError):
            reg.predict(X)


class TestBaseEstimatorInputOutputMethods:
    """Test BaseWrapper methods for pre/post processing y.
    """

    def test_post_process_y(self):
        """Quick check for BaseWrapper's _post_process_y method.
        """
        y = np.array([0])
        np.testing.assert_equal(wrappers.BaseWrapper._post_process_y(y)[0], y)
        assert len(wrappers.BaseWrapper._post_process_y(y)[1]) == 0


class TestUnsetParameter:
    """Tests for appropriate error on unfitted models.
    """

    def test_unset_input_parameter(self):
        class ClassBuildFnClf(wrappers.KerasClassifier):
            def __init__(self, input_param):
                # does not set input_param
                super().__init__()

            def __call__(self, hidden_dim):
                return build_fn_clf(hidden_dim)

        with pytest.raises(RuntimeError):
            ClassBuildFnClf(input_param=10)


class TestPrettyPrint:
    """Tests pretty printing of models.
    """

    def test_pprint(self):
        clf = wrappers.KerasClassifier(
            build_fn=build_fn_clf,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )
        print(clf)


class TestWarmStart:
    @pytest.mark.parametrize(
        "config",
        ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
    )
    def test_warm_start(self, config):
        """Test the warm start parameter."""
        warm_start: bool

        loader, model, build_fn, _ = CONFIG[config]
        clf = model(build_fn, epochs=1)
        data = loader()
        X, y = data.data[:100], data.target[:100]

        # Initial fit
        clf = model(build_fn, epochs=1)
        clf.fit(X, y)
        model = clf.model_

        # With warm start, successive calls to fit
        # should NOT create a new model
        clf.fit(X, y, warm_start=True)
        assert model is clf.model_

        # Without warm start, each call to fit
        # should create a new model instance
        clf.fit(X, y, warm_start=False)
        assert model is not clf.model_
        model = clf.model_  # for successive tests

        # The default should be warm_start=False
        clf.fit(X, y)
        assert model is not clf.model_


class TestPartialFit:
    @pytest.mark.parametrize(
        "config",
        ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
    )
    def test_partial_fit(self, config):
        loader, model, build_fn, _ = CONFIG[config]
        clf = model(build_fn, epochs=1)
        data = loader()

        X, y = data.data[:100], data.target[:100]
        clf.partial_fit(X, y)

        assert len(clf.history_["loss"]) == 1
        clf.partial_fit(X, y)
        assert len(clf.history_["loss"]) == 2

        # Make sure new model not created
        model = clf.model_
        clf.partial_fit(X, y)
        assert (
            clf.model_ is model
        ), "Model memory address should remain constant"

    def test_partial_fit_history_len(self, config="CNNClassifier"):
        # history_ records the history from this partial_fit call
        # Make sure for each call to partial_fit a single entry
        # into the history is added
        # As per https://github.com/keras-team/keras/issues/1766,
        # there is no direct measure of epochs
        loader, model, build_fn, _ = CONFIG[config]
        clf = model(build_fn, epochs=1)
        data = loader()

        X, y = data.data[:100], data.target[:100]
        for k in range(10):
            clf = clf.partial_fit(X, y)
            assert len(clf.history_["loss"]) == k + 1
            assert set(clf.history_.keys()) == {"loss", "accuracy"}

    @pytest.mark.parametrize(
        "config", ["CNNClassifier", "CNNClassifierF"],
    )
    def test_pf_pickle_pf(self, config):
        loader, model, build_fn, _ = CONFIG[config]
        clf = model(build_fn, epochs=1)
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
            assert np.allclose(
                clf.history_["loss"][0], clf2.history_["loss"][0]
            )

        weights1 = [w.numpy() for w in clf.model_.weights]
        weights2 = [w.numpy() for w in clf2.model_.weights]
        n_weights = [w1.size for w1 in weights1]
        assert 1000 <= sum(n_weights) <= 2000
        assert 200 <= np.mean(n_weights) <= 300
        assert max(n_weights) >= 1000
        rel_errors = [
            np.linalg.norm(w1 - w2) / np.linalg.norm((w1 + w2) / 2)
            for w1, w2 in zip(weights1, weights2)
        ]
        assert len(rel_errors) == 4
        assert any(x > 0.5 for x in rel_errors)
        assert all(0.01 < x for x in rel_errors)


class TestHistory:
    @pytest.mark.parametrize(
        "config",
        ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"],
    )
    def test_history(self, config):
        loader, model, build_fn, _ = CONFIG[config]
        clf = model(build_fn, epochs=1)
        data = loader()

        X, y = data.data[:100], data.target[:100]
        clf.partial_fit(X, y)

        assert isinstance(clf.history_, dict)
        assert all(isinstance(k, str) for k in clf.history_.keys())
        assert all(isinstance(v, list) for v in clf.history_.values())
