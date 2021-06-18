from collections import defaultdict
from typing import Any, DefaultDict, Dict

import pytest

from tensorflow import keras
from tensorflow.keras.callbacks import Callback

from scikeras.wrappers import KerasClassifier


def test_callbacks_prefixes():
    """Test dispatching of callbacks using no prefix, the fit__ prefix or the predict__ prefix.
    """

    class SentinalCallback(Callback):
        def __init__(self, call_logs: DefaultDict[str, int]):
            self.call_logs = call_logs

        def on_test_begin(self, logs=None):
            self.call_logs["on_test_begin"] += 1

        def on_train_begin(self, logs=None):
            self.call_logs["on_train_begin"] += 1

        def on_predict_begin(self, logs=None):
            self.call_logs["on_predict_begin"] += 1

    callbacks_call_logs = defaultdict(int)
    fit_callbacks_call_logs = defaultdict(int)
    predict_callbacks_call_logs = defaultdict(int)

    def get_clf() -> keras.Model:
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer((1,)))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        return model

    clf = KerasClassifier(
        model=get_clf,
        loss="binary_crossentropy",
        callbacks=SentinalCallback(callbacks_call_logs),
        fit__callbacks=SentinalCallback(fit_callbacks_call_logs),
        predict__callbacks=SentinalCallback(predict_callbacks_call_logs),
        validation_split=0.1,
    )

    clf.fit([[0]] * 100, [0] * 100)
    assert callbacks_call_logs == {"on_train_begin": 1, "on_test_begin": 1}
    assert fit_callbacks_call_logs == {"on_train_begin": 1, "on_test_begin": 1}
    assert predict_callbacks_call_logs == {}
    clf.predict([[0]])
    assert callbacks_call_logs == {
        "on_train_begin": 1,
        "on_test_begin": 1,
        "on_predict_begin": 1,
    }
    assert fit_callbacks_call_logs == {"on_train_begin": 1, "on_test_begin": 1}
    assert predict_callbacks_call_logs == {"on_predict_begin": 1}


@pytest.mark.parametrize(
    "callback_kwargs",
    [
        dict(
            callbacks=[keras.callbacks.EarlyStopping],
            callbacks__0__monitor="acc",
            callbacks__0__min_delta=1,
        ),
        dict(
            callbacks={"es": keras.callbacks.EarlyStopping},
            callbacks__es__monitor="acc",
            callbacks__es__min_delta=1,
        ),
        dict(
            callbacks=keras.callbacks.EarlyStopping,
            callbacks__monitor="acc",
            callbacks__min_delta=1,
        ),
        dict(callbacks=[keras.callbacks.EarlyStopping(monitor="acc", min_delta=1)]),
        dict(
            callbacks={"es": keras.callbacks.EarlyStopping(monitor="acc", min_delta=1)}
        ),
        dict(callbacks=keras.callbacks.EarlyStopping(monitor="acc", min_delta=1)),
    ],
    ids=[
        "class list syntax",
        "class dict syntax",
        "single class syntax",
        "object list sytnax",
        "object dict synax",
        "single object sytnax",
    ],
)
def test_callback_param_routing_syntax(callback_kwargs: Dict[str, Any]):
    """Test support for the various parameter routing syntaxes for callbacks.
    """

    def get_clf() -> keras.Model:
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer((1,)))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        return model

    clf = KerasClassifier(
        model=get_clf,
        epochs=5,
        loss="binary_crossentropy",
        metrics="acc",
        **callback_kwargs,
    )
    clf.fit([[1], [1]], [0, 1])
    # should early stop after 1-2 epochs (depending on the TF version) since we set the accuracy delta to 1
    assert clf.current_epoch < 5


def test_callback_compiling_args_or_kwargs():
    """Test compiling callbacks with routed positional (args) or keyword (kwargs) arguments.
    """

    def get_clf() -> keras.Model:
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer((1,)))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        return model

    class ArgsOnlyCallback(keras.callbacks.Callback):
        def __init__(self, *args):
            assert args == ("arg0", "arg1")
            ArgsOnlyCallback.called = True
            super().__init__()

    class KwargsOnlyCallback(keras.callbacks.Callback):
        def __init__(self, **kwargs):
            assert kwargs == {"kwargname": None}
            KwargsOnlyCallback.called = True
            super().__init__()

    class ArgsAndKwargsCallback(keras.callbacks.Callback):
        def __init__(self, *args, **kwargs):
            assert args == ("arg",)
            assert kwargs == {"kwargname": None}
            ArgsAndKwargsCallback.called = True
            super().__init__()

    clf = KerasClassifier(
        model=get_clf,
        epochs=5,
        optimizer=keras.optimizers.SGD,
        optimizer__learning_rate=0.1,
        loss="binary_crossentropy",
        callbacks={
            "args": ArgsOnlyCallback,
            "kwargs": KwargsOnlyCallback,
            "argskwargs": ArgsAndKwargsCallback,
        },
        callbacks__args__1="arg1",  # passed as an arg
        callbacks__args__0="arg0",  # unorder the args on purpose, SciKeras should not care about the order of the keys
        callbacks__kwargs__kwargname=None,  # passed as a kwarg
        callbacks__argskwargs__0="arg",  # passed as an arg
        callbacks__argskwargs__kwargname=None,  # passed as a kwarg
    )
    clf.fit([[1]], [1])

    for cls in (ArgsOnlyCallback, KwargsOnlyCallback, ArgsAndKwargsCallback):
        assert cls.called
