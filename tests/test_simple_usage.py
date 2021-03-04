import numpy as np
import pytest
import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder

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

    SciKeras provides raises an error with a helpful suggestion when the user
    compiles their own model with loss='sparse_categorical_crossentropy'
    but doesn't change the default loss.
    """
    y = np.random.choice(N_CLASSES, size=len(X))
    est = KerasClassifier(shallow_net, model__compile=True, model__loss=loss)

    if loss == "sparse_categorical_crossentropy":
        with pytest.warns(
            UserWarning,
            match="Setting parameter loss='sparse_categorical_crossentropy'",
        ):
            est.partial_fit(X, y)
    else:
        est.partial_fit(X, y)

    assert est.model_.loss == loss  # not est.model_.loss.__name__ b/c user compiled
    assert est.current_epoch == 1


@pytest.mark.parametrize(
    "use_case",
    [
        "binary_classification",
        "binary_classification_w_one_class",
        "classification_w_1d_targets",
        "classification_w_onehot_targets",
    ],
)
def test_classifier_only_model_specified(use_case):
    """
    Test uses cases where KerasClassifier works with the default loss.
    """

    model__single_output = True if "binary" in use_case else False
    if use_case == "binary_classification":
        y = np.random.choice(2, size=len(X)).astype(int)
    elif use_case == "binary_classification_w_one_class":
        y = np.zeros(len(X))
    elif use_case == "classification_w_1d_targets":
        y = np.random.choice(N_CLASSES, size=len(X)).astype(int)
    elif use_case == "classification_w_onehot_targets":
        y = np.random.choice(N_CLASSES, size=len(X)).astype(int)
        y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
    else:
        raise ValueError("use_case={use_case} not recognized")

    est = KerasClassifier(model=shallow_net, model__single_output=model__single_output)
    if "binary" in use_case:
        with pytest.warns(UserWarning, match="Set loss='binary_crossentropy'"):
            est.partial_fit(X, y)
        est.set_params(loss="binary_crossentropy")

    est.partial_fit(X, y=y)
    assert est.current_epoch in {1, 2}


def test_classifier_raises_for_single_output_with_multiple_classes():
    """
    KerasClassifier does not work with one output and multiple classes
    in the target (duh).
    """
    est = KerasClassifier(model=shallow_net, model__single_output=True)
    y = np.random.choice(N_CLASSES, size=len(X))

    loss = "categorical_crossentropy"
    msg = (
        "The model is configured to have a single output with a single output "
        f"neuron, but the loss='{loss}' is expecting n_classes output neurons "
    )
    with pytest.raises(ValueError, match=msg):
        est.partial_fit(X, y)
    assert est.current_epoch == 0


def test_classifier_raises_loss_binary_multi_misspecified():
    est = KerasClassifier(
        model=shallow_net,
        model__single_output=True,
        model__in_dim=1,
        loss="bce",
        epochs=100,
        random_state=42,
    )
    X = np.random.choice(2, size=(20000, 1))
    y = X.copy()
    est.partial_fit(X, y)
    assert est.score(X, y) >= 0.9


def test_regressor_default_loss():
    y = np.random.uniform(size=len(X))
    est = KerasRegressor(model=shallow_net, model__single_output=True)
    assert est.loss == "mse"
    est.partial_fit(X, y)
    assert est.model_.loss.__name__ == "mean_squared_error"


def test_classifier_default_loss():
    y = np.random.choice(N_CLASSES, size=len(X))
    est = KerasClassifier(model=shallow_net, model__single_output=False)
    assert est.loss == "categorical_crossentropy"
    est.partial_fit(X, y)
    assert est.model_.loss.__name__ == "categorical_crossentropy"
