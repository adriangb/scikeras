import numpy as np
import pytest
import tensorflow as tf
from sklearn.datasets import make_classification

from scikeras.wrappers import KerasClassifier

N_CLASSES = 4
FEATURES = 8
n_eg = 100
X = np.random.uniform(size=(n_eg, FEATURES))
y = np.random.choice(N_CLASSES, size=n_eg)


def clf(single_output=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(FEATURES,)))
    model.add(tf.keras.layers.Dense(FEATURES))

    if single_output:
        model.add(tf.keras.layers.Dense(1))
    else:
        model.add(tf.keras.layers.Dense(N_CLASSES))

    return model


def test_classifier_loss_defaults():
    est = KerasClassifier(model=clf)
    est.partial_fit(X, y=y)
    assert est.current_epoch == 1


def test_classifier_raises_for_single_output():
    est = KerasClassifier(model=clf, model__single_output=True)
    msg = (
        "one output, but the loss='categorical_crossentropy' "
        "can not handle multiple outputs"
    )
    with pytest.raises(ValueError, match=msg):
        est.partial_fit(X, y)
    assert est.current_epoch == 0
