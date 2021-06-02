import math
import pickle

import numpy as np

from sklearn.utils.estimator_checks import (
    check_estimators_partial_fit_n_features,  # noqa
)
from sklearn.utils.estimator_checks import check_estimators_pickle
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.python.ops.gen_math_ops import exp

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

    # we test callback initialization using LRS w/ a nested class requiring its own initialization
    # this test generally mirrors https://keras.io/api/callbacks/learning_rate_scheduler/

    kwargs = dict(
        model=get_clf,
        epochs=15,
        optimizer=keras.optimizers.SGD,
        optimizer__learning_rate=0.1,
        loss="sparse_categorical_crossentropy",
    )

    # using list syntax
    clf1 = KerasClassifier(
        **kwargs,
        fit__callbacks=[keras.callbacks.LearningRateScheduler],
        fit__callbacks__0__0=Schedule,  # LRS does not acccept kwargs, only args, hence the 0__0 syntax
        fit__callbacks__0__0__coef=0.2,  # translates to kwarg "coef" to the first arg of the first element of the callbacks kwarg to fit
    )

    # using dict syntax
    clf2 = KerasClassifier(
        **kwargs,
        fit__callbacks={"lr": keras.callbacks.LearningRateScheduler},
        fit__callbacks__lr__0=Schedule,  # (LRS does not acccept kwargs)
        fit__callbacks__lr__0__coef=0.2,
    )

    # using object syntax
    clf3 = KerasClassifier(
        **kwargs,
        fit__callbacks=[keras.callbacks.LearningRateScheduler(Schedule(coef=0.2))],
    )
    # object syntax should also support a plain `callbacks` parameter
    clf4 = KerasClassifier(
        **kwargs, callbacks=[keras.callbacks.LearningRateScheduler(Schedule(coef=0.2))],
    )

    for clf in (clf1, clf2, clf3, clf4):
        clf.fit(X, y)
        final_lr = clf.model_.optimizer.lr.numpy()
        expected_final_lr = 0.04493  # result of applying decay w/ coef 0.2 for 4 epochs to initial lr of 0.1
        np.testing.assert_allclose(final_lr, expected_final_lr, atol=1e-5)
