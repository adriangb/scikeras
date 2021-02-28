import numpy as np
import pytest
import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder

from scikeras.wrappers import KerasClassifier


N_CLASSES = 4
FEATURES = 8
n_eg = 100
X = np.random.uniform(size=(n_eg, FEATURES)).astype("float32")


def clf(single_output=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(FEATURES,)))
    model.add(tf.keras.layers.Dense(FEATURES))

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
