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


def clf(single_output=False, in_dim=FEATURES):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(in_dim,)))
    model.add(tf.keras.layers.Dense(in_dim, activation="sigmoid"))

    if single_output:
        model.add(tf.keras.layers.Dense(1))
    else:
        model.add(tf.keras.layers.Dense(N_CLASSES))

    return model


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

    est = KerasClassifier(model=clf, model__single_output=model__single_output)
    if "binary" in use_case:
        with pytest.raises(ValueError, match="Set loss='binary_crossentropy'"):
            est.partial_fit(X, y)
        est.set_params(loss="binary_crossentropy")

    est.partial_fit(X, y=y)
    assert est.current_epoch == 1


def test_classifier_raises_for_single_output_with_multiple_classes():
    """
    KerasClassifier does not work with one output and multiple classes
    in the target (duh).
    """
    est = KerasClassifier(model=clf, model__single_output=True)
    y = np.random.choice(N_CLASSES, size=len(X))
    msg = (
        "The model is configured to have one output, but the "
        "loss='categorical_crossentropy' is expecting multiple outputs "
    )
    with pytest.raises(ValueError, match=msg):
        est.partial_fit(X, y)
    assert est.current_epoch == 0

def test_classifier_raises_loss_binary_multi_misspecified():
    est = KerasClassifier(model=clf, model__single_output=True, model__in_dim=1, loss="bce", epochs=100, random_state=42)
    X = np.random.choice(2, size=(20000, 1))
    y = X.copy()
    est.partial_fit(X, y)
    assert est.score(X, y) >= 0.9

def test_regressor_default_loss():
    y = np.random.uniform(size=len(X))
    est = KerasRegressor(model=clf, model__single_output=True)
    assert est.loss == "mse"
    est.partial_fit(X, y)
    assert est.model_.loss.__name__ == "mean_squared_error"
