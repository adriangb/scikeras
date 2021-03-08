import numpy as np
import pytest
import tensorflow as tf

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from scikeras.utils import loss_name
from scikeras.wrappers import KerasClassifier, KerasRegressor


N_CLASSES = 4
FEATURES = 8
n_eg = 100
X = np.random.uniform(size=(n_eg, FEATURES)).astype("float32")


def shallow_net(single_output=False, loss=None, compile=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(FEATURES,)))
    if single_output:
        model.add(tf.keras.layers.Dense(1))
    else:
        model.add(tf.keras.layers.Dense(N_CLASSES))

    if compile:
        model.compile(loss=loss)

    return model


@pytest.mark.parametrize(
    "loss",
    [
        "binary_crossentropy",
        "categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "poisson",
        "kl_divergence",
        "hinge",
        "categorical_hinge",
        "squared_hinge",
    ],
)
def test_user_compiled(loss):
    """Test to make sure that user compiled classification models work with all
    classification losses.
    """
    model__single_output = True if "binary" in loss else False
    if loss == "binary_crosentropy":
        y = np.random.randint(0, 2, size=(n_eg,))
    elif loss == "categorical_crossentropy":
        # SciKeras does not auto one-hot encode unless
        # loss="categorical_crossentropy" is explictily passed to the constructor
        y = np.random.randint(0, N_CLASSES, size=(n_eg, 1))
        y = OneHotEncoder(sparse=False).fit_transform(y)
    else:
        y = np.random.randint(0, N_CLASSES, size=(n_eg,))
    est = KerasClassifier(
        shallow_net,
        model__compile=True,
        model__loss=loss,
        model__single_output=model__single_output,
    )
    est.partial_fit(X, y)

    assert est.model_.loss == loss  # not est.model_.loss.__name__ b/c user compiled
    assert est.current_epoch == 1


class NoEncoderClf(KerasClassifier):
    """A classifier overriding default target encoding.
    This simulates a user implementing custom encoding logic in 
    target_encoder to support multiclass-multioutput or
    multilabel-indicator, which by default would raise an error.
    """

    @property
    def target_encoder(self):
        return FunctionTransformer()


@pytest.mark.parametrize("use_case", ["multilabel-indicator", "multiclass-multioutput"])
def test_classifier_unsupported_multi_output_tasks(use_case):
    """Test for an appropriate error for tasks that are not supported
    by `loss="auto"`.
    """
    if use_case == "multiclass-multioutput":
        y1 = np.random.randint(0, 1, size=len(X))
        y2 = np.random.randint(0, 2, size=len(X))
        y = np.column_stack([y1, y2])
    elif use_case == "multilabel-indicator":
        y1 = np.random.randint(0, 1, size=len(X))
        y = np.column_stack([y1, y1])
    est = NoEncoderClf(shallow_net, model__compile=False)
    with pytest.raises(
        NotImplementedError, match='`loss="auto"` is not supported for tasks of type'
    ):
        est.initialize(X, y)


@pytest.mark.parametrize(
    "use_case",
    [
        "binary_classification",
        "binary_classification_w_one_class",
        "classification_w_1d_targets",
        "classification_w_onehot_targets",
    ],
)
def test_classifier_default_loss_only_model_specified(use_case):
    """Test that KerasClassifier will auto-determine a loss function
    when only the model is specified.
    """

    model__single_output = True if "binary" in use_case else False
    if use_case == "binary_classification":
        exp_loss = "binary_crossentropy"
        y = np.random.choice(2, size=len(X)).astype(int)
    elif use_case == "binary_classification_w_one_class":
        exp_loss = "binary_crossentropy"
        y = np.zeros(len(X))
    elif use_case == "classification_w_1d_targets":
        exp_loss = "sparse_categorical_crossentropy"
        y = np.random.choice(N_CLASSES, size=(len(X), 1)).astype(int)
    elif use_case == "classification_w_onehot_targets":
        exp_loss = "categorical_crossentropy"
        y = np.random.choice(N_CLASSES, size=len(X)).astype(int)
        y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

    est = KerasClassifier(model=shallow_net, model__single_output=model__single_output)

    est.fit(X, y=y)
    assert loss_name(est.model_.loss) == exp_loss
    assert est.loss == "auto"


@pytest.mark.parametrize("use_case", ["single_output", "multi_output"])
def test_regressor_default_loss_only_model_specified(use_case):
    """Test that KerasRegressor will auto-determine a loss function
    when only the model is specified.
    """
    y = np.random.uniform(size=len(X))
    if use_case == "multi_output":
        y = np.column_stack([y, y])
    est = KerasRegressor(model=shallow_net, model__single_output=True)
    est.fit(X, y)
    assert est.loss == "auto"
    assert loss_name(est.model_.loss) == "mean_squared_error"


class Multi:
    """Mixin for a simple 2 output model
    """

    def _keras_build_fn(self, compile):
        inp = tf.keras.layers.Input(shape=(FEATURES,))
        out1 = tf.keras.layers.Dense(1)(inp)
        out2 = tf.keras.layers.Dense(1)(inp)
        model = tf.keras.Model(inp, [out1, out2])
        if compile:
            model.compile(loss="mse")
        return model

    @property
    def target_encoder(self):
        return FunctionTransformer(lambda x: [x[:, 0], x[:, 1]])


class RegMulti(Multi, KerasRegressor):
    pass


class ClfMulti(Multi, KerasClassifier):
    pass


@pytest.mark.parametrize("user_compiled", [True, False])
@pytest.mark.parametrize("est_cls", [RegMulti, ClfMulti])
def test_multi_output_support(user_compiled, est_cls):
    """Test that `loss="auto"` does not support SciKeras
    compiling for multi-output models but allows user-compiled models.
    """
    y = np.random.randint(0, 1, size=len(X))
    y = np.column_stack([y, y])
    est = est_cls(model__compile=user_compiled)
    if user_compiled:
        est.fit(X, y)
    else:
        with pytest.raises(
            ValueError,
            match='Only single-output models are supported with `loss="auto"`',
        ):
            est.fit(X, y)
