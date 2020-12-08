from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf

from numpy.core.shape_base import hstack
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from tensorflow.python.keras.layers import Concatenate, Dense, Input
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.testing_utils import get_test_data

from scikeras.wrappers import KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor
from .multi_output_models import MultiOutputClassifier


# Defaults
INPUT_DIM = 5
TRAIN_SAMPLES = 10
TEST_SAMPLES = 5
NUM_CLASSES = 2


class FunctionalAPIMultiLabelClassifier(MultiOutputClassifier):
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


class FunctionalAPIMultiOutputRegressor(KerasRegressor):
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


@pytest.mark.parametrize(
    "tf_fn,error,error_pat",
    [
        (
            lambda X: X,
            ValueError,
            "``X`` has 1 inputs, but the Keras model has 2 inputs",
        ),
        (
            lambda X: [X],
            ValueError,
            "``X`` has 1 inputs, but the Keras model has 2 inputs",
        ),
        (
            lambda X: [X[:, 0], X[:, 1:2], X[:, 2:3]],
            ValueError,
            "``X`` has 3 inputs, but the Keras model has 2 inputs",
        ),
        (lambda X: [X[:, 0], X[:, 1:4]], None, ""),
        (lambda X: [X[:, 0:1], X[:, 1:4]], None, ""),
        (
            lambda X: [X[:, 0], X[:, 1:3]],
            ValueError,
            r"expected shape \(3,\) but got \(2,\)",
        ),
        (lambda X: {"Inp1": X[:, 0], "Inp2": X[:, 1:4]}, None, ""),
        (lambda X: {"Inp1": X[:, 0:1], "Inp2": X[:, 1:4]}, None, ""),
        (
            lambda X: {"Inp1": X[:, 0], "Inp2": X[:, 1:3]},
            ValueError,
            r"expected shape \(3,\) but got \(2,\)",
        ),
    ],
)
def test_multi_input(tf_fn, error, error_pat):
    """Test handling of multiple inputs as lists and dicts.
    """

    class MultiInputClassifier(KerasClassifier):
        def _keras_build_fn(self, meta: Dict[str, Any]) -> Model:
            n_classes_ = meta["n_classes_"]
            inp1 = Input((1,), name="Inp1")
            inp2 = Input((3,), name="Inp2")
            x1 = Dense(100)(inp1)
            x2 = Dense(100)(inp2)
            x3 = Concatenate(axis=-1)([x1, x2])
            cat_out = Dense(n_classes_, activation="softmax")(x3)
            model = Model([inp1, inp2], [cat_out])
            losses = ["sparse_categorical_crossentropy"]
            model.compile(loss=losses)
            return model

        @property
        def feature_encoder(self):
            return FunctionTransformer(func=tf_fn)

    est = MultiInputClassifier()

    X, y = np.random.random((10, 4)), np.random.randint(low=0, high=3, size=(10,))

    if error:
        with pytest.raises(error, match=error_pat):
            est.fit(X, y)
    else:
        est.fit(X, y)
        est.score(X, y)


@pytest.mark.parametrize(
    "tf_fn,error,error_pat",
    [
        (lambda y: [y[:, 0], y[:, 1]], None, "",),
        (lambda y: [y[:, 0:1], y[:, 1]], None, "",),
        (
            lambda y: [y[:, 0], y[:, 1], y[:, 1]],
            ValueError,
            "3 outputs, but this Keras Model has 2 outputs",
        ),
        (lambda y: [y], ValueError, "1 outputs, but this Keras Model has 2 outputs",),
        (lambda y: y, ValueError, "1 outputs, but this Keras Model has 2 outputs",),
        (lambda y: {"Out1": y[:, 0], "Out2": y[:, 1]}, None, "",),
        (lambda y: {"Out1": y[:, 0]}, ValueError, "",),
    ],
)
def test_multi_output_clf(tf_fn, error, error_pat):
    """Test handling of multiple outputs for classifiers as lists and dicts.
    """

    class SciKerasFunctionTransformer(FunctionTransformer):
        def fit(self, y):
            super().fit(y)
            y_tf = super().transform(y)
            if isinstance(y_tf, (list, dict)):
                self.n_outputs_expected_ = len(y_tf)
            else:
                self.n_outputs_expected_ = 1
            return self

        def get_metadata(self) -> Dict[str, Any]:
            return {"n_outputs_expected_": self.n_outputs_expected_}

    class MultiOutputClassifier(KerasClassifier):
        def _keras_build_fn(self) -> Model:
            n_features_in_ = self.n_features_in_
            inp = Input((n_features_in_,))
            x1 = Dense(100)(inp)
            out = [
                Dense(1, activation="sigmoid", name="Out1")(x1),
                Dense(3, activation="softmax", name="Out2")(x1),
            ]
            model = Model([inp], out)
            losses = ["binary_crossentropy", "sparse_categorical_crossentropy"]
            model.compile(loss=losses)
            return model

        @property
        def target_encoder(self):
            def inverse_func(y_tf):
                y1 = np.around(y_tf[0]).astype(np.int64)
                y2 = np.argmax(y_tf[1], axis=1).astype(np.int64)
                return np.column_stack([y1, y2])

            return SciKerasFunctionTransformer(
                func=tf_fn, inverse_func=inverse_func, check_inverse=False
            )

        def scorer(self, y, y_pred, **kwargs):
            return np.average(
                [
                    super().scorer(y[:, 0], y_pred[:, 0]),
                    super().scorer(y[:, 1], y_pred[:, 1]),
                ]
            )

    clf = MultiOutputClassifier()

    # generate data
    X = np.random.rand(10, 4)
    y1 = np.random.randint(0, 2, size=(10,))
    y2 = np.random.randint(0, 3, size=(10,))
    y = np.column_stack([y1, y2])

    if error:
        with pytest.raises(error, match=error_pat):
            clf.fit(X, y)
    else:
        clf.fit(X, y)
        clf.score(X, y)


def test_multi_label_clasification():
    """Compares to scikit-learn RandomForestClassifier classifier.
    """
    clf_keras = FunctionalAPIMultiLabelClassifier()
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
    reg_keras = FunctionalAPIMultiOutputRegressor()
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
        def _fit_keras_model(
            self, X, y, sample_weight, warm_start, epochs, initial_epoch
        ):
            if dtype == "object":
                assert X.dtype == np.dtype(tf.keras.backend.floatx())
            else:
                assert X.dtype == np.dtype(dtype)
            # y is passed through encoders, it is likely not the original dtype
            # sample_weight should always be floatx
            assert sample_weight.dtype == np.dtype(tf.keras.backend.floatx())
            return super()._fit_keras_model(
                X, y, sample_weight, warm_start, epochs, initial_epoch
            )

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
        def _fit_keras_model(
            self, X, y, sample_weight, warm_start, epochs, initial_epoch
        ):
            if dtype == "object":
                assert X.dtype == np.dtype(tf.keras.backend.floatx())
                assert y.dtype == np.dtype(tf.keras.backend.floatx())
            else:
                assert X.dtype == np.dtype(dtype)
                assert y.dtype == np.dtype(dtype)
            # sample_weight should always be floatx
            assert sample_weight.dtype == np.dtype(tf.keras.backend.floatx())
            return super()._fit_keras_model(
                X, y, sample_weight, warm_start, epochs, initial_epoch
            )

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
        def _fit_keras_model(
            self, X, y, sample_weight, warm_start, epochs, initial_epoch
        ):
            if X_dtype == "object":
                assert X.dtype == np.dtype(tf.keras.backend.floatx())
            else:
                assert X.dtype == np.dtype(X_dtype)
            if y_dtype == "object":
                assert y.dtype == np.dtype(tf.keras.backend.floatx())
            else:
                assert y.dtype == np.dtype(y_dtype)
            return super()._fit_keras_model(
                X, y, sample_weight, warm_start, epochs, initial_epoch
            )

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


def test_single_output_multilabel_indicator():
    """Tests a target that a multilabel-indicator
    target can be used without errors.
    """
    X = np.random.random(size=(100, 2))
    y = np.random.randint(0, 1, size=(100, 3))
    y[0, :] = 1  # i.e. not "one hot encoded"

    def build_fn():
        model = Sequential()
        model.add(Dense(10, input_shape=(2,), activation="relu"))
        model.add(Dense(3, activation="sigmoid"))
        return model

    clf = KerasClassifier(model=build_fn, loss="categorical_crossentropy",)
    # check that there are no errors
    clf.fit(X, y)
    clf.predict(X)
    # check the target type
    assert clf.target_type_ == "multilabel-indicator"
    # check classes
    np.testing.assert_equal(clf.classes_, np.arange(3))
