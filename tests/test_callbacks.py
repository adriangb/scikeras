import math
import pickle

import numpy as np

from sklearn.utils.estimator_checks import (
    check_estimators_partial_fit_n_features,  # noqa
)
from sklearn.utils.estimator_checks import check_estimators_pickle
from tensorflow import keras
from tensorflow.keras.callbacks import Callback

from scikeras.wrappers import KerasClassifier

from .mlp_models import dynamic_classifier


class SentinalCallback(Callback):
    """
    Callback class that sets an internal value once it's been acessed.
    """

    def __init__(self, start_at: int = 0):
        self.train_counter = start_at

    def on_train_begin(self, logs=None):
        self.train_counter += 1


def test_callbacks():
    """Test estimators with callbacks.
    """
    estimator = KerasClassifier(
        model=dynamic_classifier,
        callbacks=(SentinalCallback(),),
        optimizer="adam",
        model__hidden_layer_sizes=(100,),
    )
    # Check for picklign and partial fit
    check_estimators_pickle("KerasClassifier", estimator)
    check_estimators_partial_fit_n_features("KerasClassifier", estimator)
    # Check that callback was called
    estimator.fit([[0]], [1])
    assert estimator.callbacks[0].train_counter == 1
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    assert deserialized_estimator.callbacks[0].train_counter == 1
    deserialized_estimator.fit([[0]], [1])
    assert deserialized_estimator.callbacks[0].train_counter == 2


class Schedule:
    """Simple exponential decay lr scheduler.
    Modeled after https://keras.io/api/callbacks/learning_rate_scheduler/
    Used to test nested object construction for callbacks.
    """

    def __init__(self, coef: float = 0.1) -> None:
        self.coef = coef

    def __call__(self, epoch: int, lr: float):
        if epoch > 10:
            lr = lr * math.exp(-self.coef)
        return lr


def test_callback_param_routing():
    """Test that callbacks can be passed as routed parameters (eg. for grid search).

    See https://github.com/adriangb/scikeras/issues/232
    """
    X, y = (
        np.random.uniform(size=(100, 1)),
        np.random.randint(low=0, high=2, size=(100,)),
    )

    def get_clf() -> keras.Model:
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer((1,)))
        model.add(keras.layers.Dense(2, activation="softmax"))
        return model

    # using LRS with a nested instantiation of Schedule
    clf = KerasClassifier(
        get_clf,
        epochs=15,
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.SGD(
            learning_rate=0.1
        ),  # to mirror https://keras.io/api/callbacks/learning_rate_scheduler/
        fit__callbacks=[keras.callbacks.LearningRateScheduler],
        fit__callbacks__0__0=Schedule,  # LRS does not acccept kwargs, only args, hence the 0__0 syntax
        fit__callbacks__0__0__coef=0.2,  # translates to kwarg "coef" to the first arg of the first element of the callbacks kwarg to fit
    )
    clf.fit(X, y)
    np.testing.assert_almost_equal(
        round(clf.optimizer.lr.numpy(), 5), 0.04493
    )  # after applying decay for 4 epochs
