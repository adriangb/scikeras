import numpy as np
import pytest
import tensorflow as tf

from sklearn.datasets import make_classification

from scikeras.wrappers import KerasClassifier


N_CLASSES = 4
FEATURES = 8
n_eg = 100
X = np.random.uniform(size=(n_eg, FEATURES)).astype("float32")
y = np.random.choice(N_CLASSES, size=n_eg).astype(int)


def clf(single_output=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(FEATURES,)))
    model.add(tf.keras.layers.Dense(FEATURES))

    if single_output:
        model.add(tf.keras.layers.Dense(1))
    else:
        model.add(tf.keras.layers.Dense(N_CLASSES))

    return model


def test_classifier_only_model_specified():
    """
    This tests uses cases where KerasClassifier works with the default loss.
    It works for the following cases:

    * binary classification
    * one hot classification
    * single class classification

    """
    est = KerasClassifier(model=clf)
    est.partial_fit(X, y=y)
    assert est.current_epoch == 1

    for y2 in [
        np.random.choice(2, size=len(X)).astype(int),
        (np.random.choice(2, size=len(X)).astype(int) * 2 - 1),
        np.ones(len(X)).astype(int),
        np.zeros(len(X)).astype(int),
    ]:
        est = KerasClassifier(model=clf, model__single_output=True)
        est.partial_fit(X, y=y2)
        assert est.current_epoch == 1


def test_classifier_raises_for_single_output_with_multiple_classes():
    """
    KerasClassifier does not work with one output and multiple classes
    in the target (duh).
    """
    est = KerasClassifier(model=clf, model__single_output=True)
    msg = (
        "The model is configured to have one output, but the "
        "loss='categorical_crossentropy' is expecting multiple outputs "
    )
    with pytest.raises(ValueError, match=msg):
        est.partial_fit(X, y)
    assert est.current_epoch == 0
