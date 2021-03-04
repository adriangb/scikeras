import numpy as np
import pytest
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder

from scikeras.utils import loss_name
from scikeras.wrappers import KerasClassifier, KerasRegressor


N_CLASSES = 4
FEATURES = 8
n_eg = 100
X = np.random.uniform(size=(n_eg, FEATURES)).astype("float32")


def shallow_net(single_output=False, loss=None, in_dim=FEATURES, compile=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(in_dim,)))
    model.add(tf.keras.layers.Dense(in_dim, activation="sigmoid"))

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


@pytest.mark.parametrize(
    "use_case,supported",
    [
        ("binary_classification", True),
        ("binary_classification_w_one_class", True),
        ("classification_w_1d_targets", True),
        ("classification_w_onehot_targets", False),
    ],
)
def test_classifier_only_model_specified(use_case, supported):
    """
    Test uses cases where KerasClassifier works with the default loss.
    """

    model__single_output = True if "binary" in use_case else False
    if use_case == "binary_classification":
        y = np.random.choice(2, size=len(X)).astype(int)
    elif use_case == "binary_classification_w_one_class":
        y = np.zeros(len(X))
    elif use_case == "classification_w_1d_targets":
        y = np.random.choice(N_CLASSES, size=(len(X), 1)).astype(int)
    elif use_case == "classification_w_onehot_targets":
        y = np.random.choice(N_CLASSES, size=len(X)).astype(int)
        y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

    est = KerasClassifier(model=shallow_net, model__single_output=model__single_output)

    if supported:
        est.fit(X, y=y)
    else:
        with pytest.raises(
            ValueError, match='`loss="auto"` is not supported for tasks of type'
        ):
            est.fit(X, y=y)


def test_regressor_default_loss():
    y = np.random.uniform(size=len(X))
    est = KerasRegressor(model=shallow_net, model__single_output=True)
    assert est.loss == "auto"
    est.partial_fit(X, y)
    assert loss_name(est.model_.loss) == "mean_squared_error"


@pytest.mark.parametrize(
    "use_case,loss",
    [
        ("binary_classification", "binary_crossentropy"),
        ("binary_classification_w_one_class", "binary_crossentropy"),
        ("classification_w_1d_targets", "sparse_categorical_crossentropy"),
    ],
)
def test_classifier_default_loss(use_case, loss):
    model__single_output = True if "binary" in use_case else False
    if use_case == "binary_classification":
        y = np.random.choice(2, size=len(X)).astype(int)
    elif use_case == "binary_classification_w_one_class":
        y = np.zeros(len(X))
    elif use_case == "classification_w_1d_targets":
        y = np.random.choice(N_CLASSES, size=(len(X), 1)).astype(int)
    est = KerasClassifier(model=shallow_net, model__single_output=model__single_output)
    assert est.loss == "auto"
    est.partial_fit(X, y)
    assert loss_name(est.model_.loss) == loss
