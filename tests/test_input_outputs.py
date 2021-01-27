from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from sklearn.multioutput import (
    MultiOutputClassifier as ScikitLearnMultiOutputClassifier,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import FunctionTransformer
from tensorflow.python.keras.layers import Concatenate, Dense, Input
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.testing_utils import get_test_data

from scikeras.wrappers import KerasClassifier, KerasRegressor

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
    def model(meta: Dict[str, Any]) -> Model:
        # get params
        n_features_in_ = meta["n_features_in_"]
        inp = Input((n_features_in_,))
        x1 = Dense(1)(inp)
        out = [Dense(n, activation=activation)(x1) for n in n_units]
        model = Model([inp], out)
        return model

    return model


y_vals_cls = (
    [0, 1, 0],  # single-output, binary
    [0, 1, 2],  # single output, multiclass
    [
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
    ],  # multilabel-indicator (single multi unit output)
    [
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
    ],  # multilabel-indicator (multiple single unit outputs)
    [
        [1, 0, 2],
        [0, 1, 0],
        [0, 0, 1],
    ],  # multiclass-multioutput (multiple multi-unit outputs)
)
task_names_cls = (
    "binary",
    "multiclass",
    "multilabel-indicator",
    "multilabel-indicator",
    "multiclass-multioutput",
)
y_vals_reg = (
    [0, 1, 2],  # single-output
    [[1, 0, 2], [0, 1, 0], [0, 0, 1]],  # multi-output
)
task_names_reg = (
    "continuous",
    "continuous-multioutput",
)
mlp_kwargs = {"hidden_layer_sizes": [], "max_iter": 1}
est_paris_cls = (
    (
        MLPClassifier(**mlp_kwargs),
        KerasClassifier(dynamic_classifier, hidden_layer_sizes=[]),
    ),
    (
        MLPClassifier(**mlp_kwargs),
        KerasClassifier(dynamic_classifier, hidden_layer_sizes=[]),
    ),
    (
        MLPClassifier(**mlp_kwargs),
        MultiOutputClassifier(
            create_model("sigmoid", [3]), loss="binary_crossentropy", split=False
        ),
    ),
    (
        MLPClassifier(**mlp_kwargs),
        MultiOutputClassifier(
            create_model("sigmoid", [1, 1, 1]), loss="binary_crossentropy"
        ),
    ),
    (
        ScikitLearnMultiOutputClassifier(MLPClassifier()),
        MultiOutputClassifier(
            create_model("softmax", [3, 3, 3]), loss="sparse_categorical_crossentropy"
        ),
    ),
)
est_paris_reg = (
    (MLPRegressor(), KerasRegressor(dynamic_regressor, hidden_layer_sizes=[])),
    (MLPRegressor(), KerasRegressor(dynamic_regressor, hidden_layer_sizes=[])),
)


@pytest.mark.parametrize(
    "y_dtype",
    ("float32", "float64", "int64", "int32", "uint8", "uint16", "object", "str"),
)
@pytest.mark.parametrize(
    "y_val,est_pair,task_name", zip(y_vals_cls, est_paris_cls, task_names_cls)
)
def test_output_shapes_and_dtypes_against_sklearn_cls(
    y_dtype, y_val, task_name, est_pair
):
    """Tests the output shape and dtype for all supported classification tasks
    and target dtypes (except string and object, those are incompatible with 
    multilabel-indicator and are already tested in other tests).

    The outputs' dtype and shape get compared to sklearn's MLPClassifier and are
    expected to match.

    Since `y` gets transformed (by LabelEncoder or OneHotEncoder) within
    KerasClassifier's default transfomer, we also check that this conversion
    goes directly to Keras' internal dtype (that is, we check that we don't
    convert to another intermediary dtype when applying the transformer).
    """
    if task_name == "multilabel-indicator" and y_dtype in ("object", "str"):
        pytest.skip(
            "`multilabel-indicator` tasks are incompatible with object/str target dtypes."
        )
    if y_dtype == "object":
        if task_name == "multiclass-multioutput":
            y_val = [[str(y__) for y__ in y_] for y_ in y_val]
        else:
            y_val = [str(y_) for y_ in y_val]
    y = np.array(y_val, dtype=y_dtype)
    X = np.random.uniform(size=(y.shape[0], 2))
    y_out_sklearn = est_pair[0].fit(X, y).predict(X)
    y_out_scikeras = est_pair[1].fit(X, y).predict(X)
    fit_orig = est_pair[1].model_.fit

    def check_dtypes(*args, **kwargs):
        y = kwargs["y"]
        if isinstance(y, list):
            assert all(y_.dtype.name == tf.keras.backend.floatx() for y_ in y)
        else:
            # array
            assert y.dtype.name == tf.keras.backend.floatx()
        return fit_orig(*args, **kwargs)

    with patch.object(est_pair[1].model_, "fit", new=check_dtypes):
        est_pair[1].partial_fit(X, y)
    # Check shape, should agree with sklearn for all cases
    assert y_out_scikeras.shape == y_out_sklearn.shape
    # Check dtype, should agree with sklearn for all cases except
    # object dtypes: sklearn returns a unicode type for string arrays
    # with object dtype, we return object just like the input.
    # A quirk about sklearn: multilabel-indicator models _always_ return np.int64.
    # We match this in MultiOutputClassifier/MultiLabelTransformer
    if y_dtype == "object":
        assert y_out_scikeras.dtype.name == "object"
    else:
        assert y_out_scikeras.dtype == y_out_sklearn.dtype


@pytest.mark.parametrize(
    "y_dtype", ("float32", "float64", "int64", "int32", "uint8", "uint16")
)
@pytest.mark.parametrize("y_val,est_pair", zip(y_vals_reg, est_paris_reg))
def test_output_shapes_and_dtypes_against_sklearn_reg(y_dtype, y_val, est_pair):
    """Tests the output shape and dtype for all supported regression tasks
    and target dtypes.

    The outputs' dtype and shape get compared to sklearn's MLPRegressor and are
    expected to match except for the output dtype, which in MLPRegressor is always
    float64 but in SciKeras depends on the TF backend dtype (usually float32).

    Since `y` is _not_ transformed by KerasRegressor's default transformer,
    we check that when it is passed to the Keras model it's dtype is unchanged.
    """
    y = np.array(y_val, dtype=y_dtype)
    X = np.random.uniform(size=(y.shape[0], 2))
    y_out_sklearn = est_pair[0].fit(X, y).predict(X)
    y_out_scikeras = est_pair[1].fit(X, y).predict(X)
    fit_orig = est_pair[1].model_.fit

    def check_dtypes(*args, **kwargs):
        y = kwargs["y"]
        assert y.dtype.name == y_dtype
        return fit_orig(*args, **kwargs)

    with patch.object(est_pair[1].model_, "fit", new=check_dtypes):
        est_pair[1].partial_fit(X, y)
    # Check shape, should agree with sklearn for all cases
    assert y_out_scikeras.shape == y_out_sklearn.shape
    # Check dtype
    # By default, KerasRegressor (or rather it's default target_encoder)
    # always returns tf.keras.backend.floatx(). This is similar to sklearn, which always
    # returns float64, except that we avoid a pointless conversion from
    # float32 -> float64 that would just be adding noise if TF is using float32
    # internally.
    assert y_out_scikeras.dtype.name == tf.keras.backend.floatx()


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
