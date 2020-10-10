from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from tensorflow.python.keras.layers import Concatenate, Dense, Input
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.testing_utils import get_test_data

from scikeras.wrappers import BaseWrapper, KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor
from .multi_output_models import MultiOutputClassifier


# Defaults
INPUT_DIM = 5
TRAIN_SAMPLES = 10
TEST_SAMPLES = 5
NUM_CLASSES = 2


class MultiInputTransformer(FunctionTransformer):
    def get_metadata(self):
        return dict()


class FunctionalAPIMultiInputClassifier(KerasClassifier):
    """Tests Functional API Classifier with 2 inputs.
    """

    def _keras_build_fn(
        self, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
    ) -> Model:
        # get params
        n_classes_ = meta["n_classes_"]

        inp1 = Input((1,))
        inp2 = Input((3,))

        x1 = Dense(100)(inp1)
        x2 = Dense(100)(inp2)

        x3 = Concatenate(axis=-1)([x1, x2])

        cat_out = Dense(n_classes_, activation="softmax")(x3)

        model = Model([inp1, inp2], [cat_out])
        losses = ["sparse_categorical_crossentropy"]
        model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

        return model

    @property
    def feature_encoder(self):
        return MultiInputTransformer(func=lambda X: [X[:, 0], X[:, 1:4]],)


class FunctionAPIMultiLabelClassifier(MultiOutputClassifier):
    """Tests Functional API Classifier with multiple binary outputs.
    """

    def _keras_build_fn(
        self, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
    ) -> Model:
        # get params
        n_outputs_ = meta["n_outputs_"]

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

    def _keras_build_fn(
        self, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
    ) -> Model:
        # get params
        n_outputs_ = meta["n_outputs_"]

        inp = Input((INPUT_DIM,))

        x1 = Dense(100)(inp)

        outputs = [Dense(n_outputs_)(x1)]

        model = Model([inp], outputs)
        losses = "mean_squared_error"
        model.compile(optimizer="adam", loss=losses, metrics=["mse"])

        return model


def test_multi_input():
    """Tests custom multi-input Keras model.
    """
    clf = FunctionalAPIMultiInputClassifier()
    (x_train, y_train), (x_test, y_test) = get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(4,),
        num_classes=3,
    )

    clf.fit(x_train, y_train)
    clf.predict(x_test)
    clf.score(x_train, y_train)


def test_multi_output():
    """Compares to scikit-learn RandomForestClassifier classifier.
    """

    def get_model(meta: Dict[str, Any]) -> Model:
        # get params
        n_features_in_ = meta["n_features_in_"]

        inp = Input((n_features_in_,))

        x1 = Dense(100)(inp)

        out = [Dense(1, activation="sigmoid")(x1) for _ in range(meta["n_outputs_"])]

        model = Model([inp], out)
        losses = "binary_crossentropy"
        model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

        return model

    clf_keras = MultiOutputClassifier(model=get_model)
    clf_sklearn = RandomForestClassifier()

    # generate data
    X = np.random.rand(10, 4)
    y1 = np.random.randint(0, 2, size=(10, 1))
    y2 = np.random.randint(0, 2, size=(10, 1))
    y = np.hstack([y1, y2])

    clf_keras.fit(X, y)
    y_wrapper = clf_keras.predict(X)
    clf_keras.score(X, y)

    clf_sklearn.fit(X, y)
    y_sklearn = clf_sklearn.predict(X)

    assert y_sklearn.shape == y_wrapper.shape


def test_multi_label_clasification():
    """Compares to scikit-learn RandomForestClassifier classifier.
    """
    clf_keras = FunctionAPIMultiLabelClassifier()
    clf_sklearn = RandomForestClassifier()
    # taken from https://scikit-learn.org/stable/modules/multiclass.html
    y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
    y = MultiLabelBinarizer().fit_transform(y)

    (x_train, _), (_, _) = get_test_data(
        train_samples=y.shape[0], test_samples=0, input_shape=(4,), num_classes=3,
    )

    clf_keras.fit(x_train, y)
    y_pred_keras = clf_keras.predict(x_train)
    clf_keras.score(x_train, y)

    clf_sklearn.fit(x_train, y)
    y_pred_sklearn = clf_sklearn.predict(x_train)
    clf_sklearn.score(x_train, y)

    assert y_pred_keras.shape == y_pred_sklearn.shape


def test_multi_output_regression():
    """Compares to scikit-learn RandomForestRegressor.
    """
    reg_keras = FunctionAPIMultiOutputRegressor()
    reg_sklearn = RandomForestRegressor()
    # taken from https://scikit-learn.org/stable/modules/multiclass.html
    (X, _), (_, _) = get_test_data(
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


@pytest.mark.parametrize(
    "y, y_type",
    [
        (np.array([1, 2, 3]), "multiclass"),  # ordinal, numeric, sorted
        (np.array([2, 1, 3]), "multiclass"),  # ordinal, numeric, sorted
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "multilabel-indicator",
        ),  # one-hot encoded
        (np.array(["a", "b", "c"]), "multiclass"),  # categorical
    ],
)
def test_KerasClassifier_loss_invariance(y, y_type):
    """Test that KerasClassifier can use both
    categorical_crossentropy and sparse_categorical_crossentropy
    with either one-hot encoded targets or sparse targets.
    """
    X = np.arange(0, y.shape[0]).reshape(-1, 1)
    clf_1 = KerasClassifier(
        model=dynamic_classifier,
        hidden_layer_sizes=(100,),
        loss="categorical_crossentropy",
        random_state=0,
    )
    clf_1.fit(X, y)
    clf_1.partial_fit(X, y)
    y_1 = clf_1.predict(X)
    if y_type != "multilabel-indicator":
        # sparse_categorical_crossentropy is not compatible with
        # one-hot encoded targets, and one-hot encoded targets are not used in sklearn
        # This is a use case that does not natively succeed in Keras or skelarn estimators
        # and thus SciKeras does not intend to auto-convert data to support it
        clf_2 = KerasClassifier(
            model=dynamic_classifier,
            hidden_layer_sizes=(100,),
            loss="sparse_categorical_crossentropy",
            random_state=0,
        )
        clf_2.fit(X, y)
        y_2 = clf_1.predict(X)

        np.testing.assert_equal(y_1, y_2)


@pytest.mark.parametrize(
    "y, y_type",
    [
        (np.array([1, 2, 3]), "multiclass"),  # ordinal, numeric, sorted
        (np.array([2, 1, 3]), "multiclass"),  # ordinal, numeric, sorted
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "multilabel-indicator",
        ),  # one-hot encoded
        (np.array(["a", "b", "c"]), "multiclass"),  # categorical
    ],
)
@pytest.mark.parametrize(
    "loss", ["categorical_crossentropy", "sparse_categorical_crossentropy"]
)
def test_KerasClassifier_transformers_can_be_reused(y, y_type, loss):
    """Test that KerasClassifier can use both
    categorical_crossentropy and sparse_categorical_crossentropy
    with either one-hot encoded targets or sparse targets.
    """
    if y_type == "multilabel-indicator" and loss == "sparse_categorical_crossentropy":
        return  # not compatible, see test_KerasClassifier_loss_invariance
    X1, y1 = np.array([[1, 2, 3]]).T, np.array([1, 2, 3])
    clf = KerasClassifier(
        model=dynamic_classifier, hidden_layer_sizes=(100,), loss=loss, random_state=0,
    )
    clf.fit(X1, y1)
    tfs = clf.target_encoder_
    X2, y2 = X1, np.array([1, 1, 1])  # only 1 out or 3 classes
    clf.partial_fit(X2, y2)
    tfs_new = clf.target_encoder_
    assert tfs_new is tfs  # same transformer was re-used
    assert set(clf.classes_) == set(y1)


def test_incompatible_output_dimensions():
    """Compares to the scikit-learn RandomForestRegressor classifier.
    """
    # create dataset with 4 outputs
    X = np.random.rand(10, 20)
    y = np.random.randint(low=0, high=3, size=(10,))

    # create a model with 2 outputs
    def build_fn_clf(meta: Dict[str, Any], compile_kwargs: Dict[str, Any],) -> Model:
        # get params
        n_features_in_ = meta["n_features_in_"]

        inp = Input((n_features_in_,))

        x1 = Dense(100)(inp)

        binary_out = Dense(1, activation="sigmoid")(x1)
        cat_out = Dense(2, activation="softmax")(x1)

        model = Model([inp], [binary_out, cat_out])
        model.compile(loss=["binary_crossentropy", "categorical_crossentropy"])

        return model

    clf = KerasClassifier(model=build_fn_clf)

    with pytest.raises(RuntimeError):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "dtype", ["float32", "float64", "int64", "int32", "uint8", "uint16", "object"],
)
def test_classifier_handles_dtypes(dtype):
    """Tests that classifiers correctly handle dtype conversions and
    return the same dtype as the inputs.
    """
    n, d = 20, 3
    n_classes = 3
    X = np.random.uniform(size=(n, d)).astype(dtype)
    y = np.random.choice(n_classes, size=n).astype(dtype)
    sample_weight = np.ones(y.shape).astype(dtype)

    class StrictClassifier(KerasClassifier):
        def _fit_keras_model(self, X, y, sample_weight, warm_start, epochs):
            if dtype == "object":
                assert X.dtype == np.dtype(tf.keras.backend.floatx())
            else:
                assert X.dtype == np.dtype(dtype)
            # y is passed through encoders, it is likely not the original dtype
            # sample_weight should always be floatx
            assert sample_weight.dtype == np.dtype(tf.keras.backend.floatx())
            return super()._fit_keras_model(X, y, sample_weight, warm_start, epochs)

    clf = StrictClassifier(model=dynamic_classifier, model__hidden_layer_sizes=(100,))
    clf.fit(X, y, sample_weight=sample_weight)
    assert clf.score(X, y) >= 0
    if y.dtype.kind != "O":
        assert clf.predict(X).dtype == np.dtype(dtype)
    else:
        assert clf.predict(X).dtype == np.float32


@pytest.mark.parametrize(
    "dtype", ["float32", "float64", "int64", "int32", "uint8", "uint16", "object"],
)
def test_regressor_handles_dtypes(dtype):
    """Tests that regressors correctly handle dtype conversions and
    always return float dtypes.
    """
    n, d = 20, 3
    X = np.random.uniform(size=(n, d)).astype(dtype)
    y = np.random.uniform(size=n).astype(dtype)
    sample_weight = np.ones(y.shape).astype(dtype)

    class StrictRegressor(KerasRegressor):
        def _fit_keras_model(self, X, y, sample_weight, warm_start, epochs):
            if dtype == "object":
                assert X.dtype == np.dtype(tf.keras.backend.floatx())
                assert y.dtype == np.dtype(tf.keras.backend.floatx())
            else:
                assert X.dtype == np.dtype(dtype)
                assert y.dtype == np.dtype(dtype)
            # sample_weight should always be floatx
            assert sample_weight.dtype == np.dtype(tf.keras.backend.floatx())
            return super()._fit_keras_model(X, y, sample_weight, warm_start, epochs)

    reg = StrictRegressor(model=dynamic_regressor, model__hidden_layer_sizes=(100,))
    reg.fit(X, y, sample_weight=sample_weight)
    y_hat = reg.predict(X)
    if y.dtype.kind == "f":
        assert y_hat.dtype == np.dtype(dtype)
    else:
        assert y_hat.dtype.kind == "f"


@pytest.mark.parametrize("X_dtype", ["float32", "int64"])
@pytest.mark.parametrize("y_dtype,", ["float32", "float64", "uint8", "int16", "object"])
@pytest.mark.parametrize("run_eagerly", [True, False])
def test_mixed_dtypes(y_dtype, X_dtype, run_eagerly):
    n, d = 20, 3
    n_classes = 3
    X = np.random.uniform(size=(n, d)).astype(X_dtype)
    y = np.random.choice(n_classes, size=n).astype(y_dtype)

    class StrictRegressor(KerasRegressor):
        def _fit_keras_model(self, X, y, sample_weight, warm_start, epochs):
            if X_dtype == "object":
                assert X.dtype == np.dtype(tf.keras.backend.floatx())
            else:
                assert X.dtype == np.dtype(X_dtype)
            if y_dtype == "object":
                assert y.dtype == np.dtype(tf.keras.backend.floatx())
            else:
                assert y.dtype == np.dtype(y_dtype)
            return super()._fit_keras_model(X, y, sample_weight, warm_start, epochs)

    reg = StrictRegressor(
        model=dynamic_regressor,
        run_eagerly=run_eagerly,
        model__hidden_layer_sizes=(100,),
    )
    reg.fit(X, y)
    y_hat = reg.predict(X)
    if y.dtype.kind == "f":
        assert y_hat.dtype == np.dtype(y_dtype)
    else:
        assert y_hat.dtype.kind == "f"
