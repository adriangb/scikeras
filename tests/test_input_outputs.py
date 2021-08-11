from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import (
    MultiOutputClassifier as ScikitLearnMultiOutputClassifier,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from tensorflow.python.keras.layers import Concatenate, Dense, Input
from tensorflow.python.keras.models import Model

from scikeras.wrappers import BaseWrapper, KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor
from .multi_output_models import MultiOutputClassifier


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
        return FunctionTransformer(func=lambda X: [X[:, 0], X[:, 1:4]],)


def test_multi_input():
    """Tests custom multi-input Keras model.
    """
    clf = FunctionalAPIMultiInputClassifier()
    X = np.random.uniform(size=(10, 4))
    y = np.arange(0, 10, 1, int)

    clf.fit(X, y)
    clf.predict(X)
    clf.score(X, y)


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

    with pytest.raises(ValueError, match="input of size"):
        clf.fit(X, y)


def create_model(activation, n_units):
    def get_model(meta: Dict[str, Any], hidden_layer_sizes=[200]) -> Model:
        # get params
        n_features_in_ = meta["n_features_in_"]
        inp = Input((n_features_in_,))
        x = inp
        for lsize in hidden_layer_sizes:
            x = Dense(lsize)(x)
        out = [Dense(n, activation=activation)(x) for n in n_units]
        model = Model([inp], out)
        return model

    return get_model


mlp_kwargs = {"hidden_layer_sizes": [200], "max_iter": 15, "random_state": 0}
scikeras_kwargs = {"hidden_layer_sizes": [200], "epochs": 15, "random_state": 0}


@dataclass
class TestParams:
    sklearn_est: BaseEstimator
    scikeras_est: BaseWrapper
    X: np.ndarray
    y: np.ndarray
    X_expected_dtype_keras: np.dtype
    y_expected_dtype_keras: np.dtype
    min_score: float
    scorer: Callable


def single_output_binary_sigmoid():
    y = np.random.randint(low=0, high=2, size=(2000,))
    X = y.reshape(-1, 1)
    sklearn_est = MLPClassifier(**mlp_kwargs)
    scikeras_est = KerasClassifier(
        create_model("sigmoid", [1]), **scikeras_kwargs, loss="binary_crossentropy"
    )
    for dtype in ("float32", "float64", "int64", "int32", "uint8", "uint16", "<U1"):
        y_ = y.astype(dtype)
        yield TestParams(
            sklearn_est=sklearn_est,
            scikeras_est=scikeras_est,
            X=X,
            y=y_,
            X_expected_dtype_keras=X.dtype,
            y_expected_dtype_keras=tf.keras.backend.floatx(),
            min_score=0.95,
            scorer=accuracy_score,
        )


def single_output_binary_softmax():
    y = np.random.randint(low=0, high=2, size=(2000,))
    X = y.reshape(-1, 1)
    y = np.column_stack([y, 1 - y])
    sklearn_est = MLPClassifier(**mlp_kwargs)
    scikeras_est = KerasClassifier(
        create_model("softmax", [2]), **scikeras_kwargs, loss="categorical_crossentropy"
    )
    for dtype in ("float32", "float64", "int64", "int32", "uint8", "uint16"):
        y_ = y.astype(dtype)
        yield TestParams(
            sklearn_est=sklearn_est,
            scikeras_est=scikeras_est,
            X=X,
            y=y_,
            X_expected_dtype_keras=X.dtype,
            y_expected_dtype_keras=tf.keras.backend.floatx(),
            min_score=0.95,
            scorer=accuracy_score,
        )


def single_output_multiclass_sparse():
    y = np.random.randint(low=0, high=3, size=(4000,))
    X = y.reshape(-1, 1)
    sklearn_est = MLPClassifier(**mlp_kwargs)
    scikeras_est = KerasClassifier(
        create_model("softmax", [3]),
        **scikeras_kwargs,
        loss="sparse_categorical_crossentropy",
    )
    for dtype in ("float32", "float64", "int64", "int32", "uint8", "uint16", "<U1"):
        y_ = y.astype(dtype)
        yield TestParams(
            sklearn_est=sklearn_est,
            scikeras_est=scikeras_est,
            X=X,
            y=y_,
            X_expected_dtype_keras=X.dtype,
            y_expected_dtype_keras=tf.keras.backend.floatx(),
            min_score=0.95,
            scorer=accuracy_score,
        )


def single_output_multiclass_one_hot():
    y = np.random.randint(low=0, high=3, size=(2000,))
    X = y.reshape(-1, 1)
    # For compatibility with Keras, accept one-hot-encoded inputs
    # with categorical_crossentropy loss
    y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
    sklearn_est = MLPClassifier(**mlp_kwargs)
    scikeras_est = KerasClassifier(
        create_model("softmax", [3]), **scikeras_kwargs, loss="categorical_crossentropy"
    )
    for dtype in ("float32", "float64", "int64", "int32", "uint8", "uint16"):
        y_ = y.astype(dtype)
        yield TestParams(
            sklearn_est=sklearn_est,
            scikeras_est=scikeras_est,
            X=X,
            y=y_,
            X_expected_dtype_keras=X.dtype,
            y_expected_dtype_keras=tf.keras.backend.floatx(),
            min_score=0.95,
            scorer=accuracy_score,
        )


def multilabel_indicator_single_sigmoid():
    X = np.random.randint(low=0, high=4, size=(2000, 1))
    y = np.zeros((2000, 2))
    y[X[:, 0] == 1, [0]] = 1
    y[X[:, 0] == 2, [1]] = 1
    y[X[:, 0] == 3, [0]] = 1
    y[X[:, 0] == 3, [1]] = 1
    sklearn_est = MLPClassifier(**mlp_kwargs)
    scikeras_est = KerasClassifier(
        create_model("sigmoid", [2]), loss="bce", **scikeras_kwargs
    )
    for dtype in ("float32", "float64", "int64", "int32", "uint8", "uint16"):
        y_ = y.astype(dtype)
        yield TestParams(
            sklearn_est=sklearn_est,
            scikeras_est=scikeras_est,
            X=X,
            y=y_,
            X_expected_dtype_keras=X.dtype,
            y_expected_dtype_keras=dtype,
            min_score=0.4,
            scorer=accuracy_score,
        )


def multilabel_indicator_multiple_sigmoid():
    X = np.random.randint(low=0, high=4, size=(2000, 1))
    y = np.zeros((2000, 2))
    y[X[:, 0] == 1, [0]] = 1
    y[X[:, 0] == 2, [1]] = 1
    y[X[:, 0] == 3, [0]] = 1
    y[X[:, 0] == 3, [1]] = 1
    sklearn_est = MLPClassifier(**mlp_kwargs)
    scikeras_est = MultiOutputClassifier(
        create_model("sigmoid", [1, 1]), loss="bce", split=True, **scikeras_kwargs
    )
    for dtype in ("float32", "float64", "int64", "int32", "uint8", "uint16"):
        y_ = y.astype(dtype)
        yield TestParams(
            sklearn_est=sklearn_est,
            scikeras_est=scikeras_est,
            X=X,
            y=y_,
            X_expected_dtype_keras=X.dtype,
            y_expected_dtype_keras=dtype,
            min_score=0.4,
            scorer=accuracy_score,
        )


def multiclass_multioutput():
    y1 = np.random.randint(low=0, high=3, size=(4000,))
    X = y1.reshape(-1, 1)
    y2 = y1 == 1
    y = np.column_stack([y1, y2])
    sklearn_est = ScikitLearnMultiOutputClassifier(MLPClassifier(**mlp_kwargs))
    scikeras_est = MultiOutputClassifier(
        create_model("softmax", [3, 2]),
        loss="sparse_categorical_crossentropy",
        split=True,
        **scikeras_kwargs,
    )
    for dtype in ("float32", "float64", "int64", "int32", "uint8", "uint16", "<U1"):
        y_ = y.astype(dtype)
        yield TestParams(
            sklearn_est=sklearn_est,
            scikeras_est=scikeras_est,
            X=X,
            y=y_,
            X_expected_dtype_keras=X.dtype,
            y_expected_dtype_keras=dtype,
            min_score=0.55,
            # note that ALL classes must be correct, so 0.55 is a reasonable score (and is ±10% vs MLPClassifier)
            scorer=lambda y, y_pred: np.mean(
                np.all(y == y_pred, axis=1)  # copied from sklearn MultiOutputClassifier
            ),
        )


@pytest.mark.parametrize(
    "test_data",
    chain(
        single_output_binary_sigmoid(),
        single_output_binary_softmax(),
        single_output_multiclass_sparse(),
        single_output_multiclass_one_hot(),
        multilabel_indicator_single_sigmoid(),
        multilabel_indicator_multiple_sigmoid(),
        multiclass_multioutput(),
    ),
)
def test_output_shapes_and_dtypes_against_sklearn_cls(test_data: TestParams):
    """Tests that ensure that SciKeras can cover all common Scikit-Learn
    output situations by comparing to MLPClassifier.
    """
    X, y = test_data.X, test_data.y
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    sklearn_est, scikeras_est = test_data.sklearn_est, test_data.scikeras_est

    scikeras_est.initialize(X_train, y_train)

    keras_model_fit = scikeras_est.model_.fit

    def check_dtypes(x, y, **kwargs):
        if isinstance(y, list):
            assert all(y_.dtype.name == test_data.y_expected_dtype_keras for y_ in y)
        else:
            # array
            assert y.dtype.name == test_data.y_expected_dtype_keras
        assert X.dtype.name == test_data.X_expected_dtype_keras
        return keras_model_fit(x=x, y=y, **kwargs)

    with patch.object(scikeras_est.model_, "fit", new=check_dtypes):
        with pytest.warns(UserWarning, match="Setting the random state for TF"):
            scikeras_est.fit(X_train, y_train)

    y_out_scikeras = scikeras_est.predict(X_test)
    y_out_sklearn = sklearn_est.fit(X_train, y_train).predict(X_test)

    assert y_out_scikeras.shape == y_out_sklearn.shape
    assert y_out_scikeras.dtype == y_out_sklearn.dtype
    scikeras_score = test_data.scorer(y_test, y_out_scikeras)
    assert scikeras_score >= test_data.min_score


def continuous():
    # use ints so that we get measurable scores when castint to uint8
    y = np.random.randint(low=0, high=2, size=(1000,))
    X = y.reshape(-1, 1)
    sklearn_est = MLPRegressor(**mlp_kwargs)
    scikeras_est = KerasRegressor(dynamic_regressor, **scikeras_kwargs)
    for dtype in ("float32", "float64", "int64", "int32", "uint8", "uint16"):
        y_ = y.astype(dtype)
        yield TestParams(
            sklearn_est=sklearn_est,
            scikeras_est=scikeras_est,
            X=X,
            y=y_,
            X_expected_dtype_keras=X.dtype,
            y_expected_dtype_keras=dtype,
            min_score=0.99,
            scorer=r2_score,
        )


def continuous_multioutput():
    # use ints so that we get measurable scores when casting to uint8
    y = np.random.randint(low=0, high=2, size=(1000,))
    X = y.reshape(-1, 1)
    y = np.column_stack([y, y])

    sklearn_est = MLPRegressor(**mlp_kwargs)
    scikeras_est = KerasRegressor(dynamic_regressor, **scikeras_kwargs)
    for dtype in ("float32", "float64", "int64", "int32", "uint8", "uint16"):
        y_ = y.astype(dtype)
        yield TestParams(
            sklearn_est=sklearn_est,
            scikeras_est=scikeras_est,
            X=X,
            y=y_,
            X_expected_dtype_keras=X.dtype,
            y_expected_dtype_keras=dtype,
            min_score=0.99,
            scorer=r2_score,
        )


@pytest.mark.parametrize("test_data", chain(continuous(), continuous_multioutput()))
def test_output_shapes_and_dtypes_against_sklearn_reg(test_data: TestParams):
    """Tests that ensure that SciKeras can cover all common Scikit-Learn
    output situations by comparing to MLPRegressor.
    """
    X, y = test_data.X, test_data.y
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    sklearn_est, scikeras_est = test_data.sklearn_est, test_data.scikeras_est

    scikeras_est.initialize(X_train, y_train)

    keras_model_fit = scikeras_est.model_.fit

    def check_dtypes(x, y, **kwargs):
        if isinstance(y, list):
            assert all(y_.dtype.name == test_data.y_expected_dtype_keras for y_ in y)
        else:
            # array
            assert y.dtype.name == test_data.y_expected_dtype_keras
        assert X.dtype.name == test_data.X_expected_dtype_keras
        return keras_model_fit(x=x, y=y, **kwargs)

    with patch.object(scikeras_est.model_, "fit", new=check_dtypes):
        scikeras_est.fit(X_train, y_train)

    y_out_scikeras = scikeras_est.predict(X_test)
    y_out_sklearn = sklearn_est.fit(X_train, y_train).predict(X_test)

    assert y_out_scikeras.shape == y_out_sklearn.shape
    # Check dtype
    # By default, KerasRegressor (or rather it's default target_encoder)
    # always returns tf.keras.backend.floatx(). This is similar to sklearn, which always
    # returns float64, except that we avoid a pointless conversion from
    # float32 -> float64 that would just be adding noise if TF is using float32
    # internally (which is usually the case).
    assert y_out_scikeras.dtype.name == tf.keras.backend.floatx()
    scikeras_score = test_data.scorer(y_test, y_out_scikeras)
    assert scikeras_score >= test_data.min_score


@pytest.mark.parametrize(
    "est",
    (
        KerasRegressor(dynamic_regressor, model__hidden_layer_sizes=[]),
        KerasClassifier(dynamic_classifier, model__hidden_layer_sizes=[]),
    ),
)
@pytest.mark.parametrize(
    "X_dtype", ("float32", "float64", "int64", "int32", "uint8", "uint16", "object")
)
def test_input_dtype_conversion(X_dtype, est):
    """Tests that using the default transformers in SciKeras,
    `X` is not converted/modified unless it is of dtype object.
    This mimics the behavior of sklearn estimators, which
    try to cast object -> numeric.
    """
    y = np.arange(0, 10, 1, int)
    X = np.random.uniform(size=(y.shape[0], 2)).astype(X_dtype)
    est.fit(X, y)  # generate model_
    fit_orig = est.model_.fit

    def check_dtypes(*args, **kwargs):
        x = kwargs["x"]
        if X_dtype == "object":
            assert x.dtype == tf.keras.backend.floatx()
        else:
            assert x.dtype == X_dtype
        return fit_orig(*args, **kwargs)

    with patch.object(est.model_, "fit", new=check_dtypes):
        est.partial_fit(X, y)


def test_sparse_matrix():
    y = np.random.randint(low=0, high=2, size=(1000,))
    X = coo_matrix(y.reshape(-1, 1))

    est = KerasClassifier(
        dynamic_classifier, model__hidden_layer_sizes=[100], epochs=20,
    )
    est.fit(X, y)
    assert est.score(X, y) > 0.85
