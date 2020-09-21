from inspect import isclass

import numpy as np
import pytest

from numpy.lib.arraysetops import isin
from sklearn.datasets import make_classification
from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module
from tensorflow.keras import optimizers as optimizers_module
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.compile_utils import losses_mod

from scikeras._utils import _class_from_strings
from scikeras.wrappers import KerasClassifier


def get_model(num_hidden=10, meta=None, compile_kwargs=None):
    inp = Input(shape=(meta["n_features_in_"],))
    hidden = Dense(num_hidden, activation="relu")(inp)
    out = [
        Dense(1, activation="sigmoid", name=f"out{i+1}")(hidden)
        for i in range(meta["n_outputs_"])
    ]
    model = Model(inp, out)
    model.compile(**compile_kwargs)
    return model


@pytest.mark.parametrize("optimizer", (optimizers_module.SGD, "SGD", "sgd"))
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_optimizer(optimizer, n_outputs_):
    """Tests compiling of single optimizer with options.
    Since there can only ever be a single optimizer, there is no
    ("name", optimizer, "output") option.
    Only optimizer classes will be compiled with custom options,
    all others (class names, function names) should pass through
    untouched.
    """
    # Single output
    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer=optimizer,
        optimizer__learning_rate=0.15,
        optimizer__momentum=0.5,
        loss="binary_crossentropy",
    )
    est.fit(X, y)
    if isclass(optimizer):
        np.testing.assert_almost_equal(
            est.model_.optimizer.momentum.value().numpy(), 0.5, decimal=2
        )
        np.testing.assert_almost_equal(
            est.model_.optimizer.learning_rate.value().numpy(), 0.15, decimal=2
        )
    else:
        # TF actually instantiates an optimizer object even if passed a string
        # Checking __class__ is not a universally valid check, but it works for the
        # test optimizer (SGD)
        assert (
            est.model_.optimizer.__class__
            == optimizers_module.get(optimizer).__class__
        )


def test_custom_loss_nested_kwargs():

    X, y = make_classification()

    class Foo:
        got = dict()

        def __init__(self, **kwargs):
            self.got = kwargs

    class MyLoss:
        def __init__(self, param1):
            self.param1 = param1
            self.__name__ = str(id(self))

        def __call__(self, y_true, y_pred):
            return losses_module.binary_crossentropy(y_true, y_pred)

    est = KerasClassifier(
        model=get_model,
        loss=MyLoss,
        loss__param1=[Foo, Foo],
        loss__param1__foo_kwarg=1,
        loss__param1__0__foo_kwarg=2,
    )
    est.fit(X, y)
    assert est.model_.loss.param1[0].got["foo_kwarg"] == 2
    assert est.model_.loss.param1[1].got["foo_kwarg"] == 1


@pytest.mark.parametrize(
    "loss",
    (
        losses_module.BinaryCrossentropy,
        "BinaryCrossentropy",
        "binary_crossentropy",
    ),
)
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_loss(loss, n_outputs_):
    """Tests compiling of single loss with options.
    Only loss classes will be compiled with custom options,
    all others (class names, function names) should pass through
    untouched.
    """

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss=loss,
        loss__name="custom_name",
    )
    est.fit(X, y)
    assert str(loss) in str(est.model_.loss) or isinstance(
        est.model_.loss, loss
    )


@pytest.mark.parametrize(
    "loss",
    (
        losses_module.BinaryCrossentropy,
        "BinaryCrossentropy",
        "binary_crossentropy",
    ),
)
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_loss_iterable(loss, n_outputs_):
    """Tests compiling of loss when it is
    given as an iterable of losses
    mapping to outputs.
    """

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    # Test iterable with global routed param
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss=[loss],
        loss__from_logits=True,  # default is False
    )
    est.fit(X, y)
    got = _class_from_strings(loss, "loss")
    if isclass(got):
        got = got()
        assert est.model_.loss[0].from_logits
    else:
        assert est.model_.loss[0].__name__ == got.__name__

    # Test iterable with index-based routed param
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss=[loss],
        loss__from_logits=False,
        loss__0__from_logits=True,  # should override above
    )
    est.fit(X, y)
    got = _class_from_strings(loss, "loss")
    if isclass(got):
        # from_logits should have been passed and set
        assert est.model_.loss[0].from_logits == True
    else:
        # at least check the name
        assert est.model_.loss[0].__name__ == got.__name__


@pytest.mark.parametrize(
    "loss",
    (
        losses_module.BinaryCrossentropy,
        "BinaryCrossentropy",
        "binary_crossentropy",
    ),
)
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_loss_dict(loss, n_outputs_):
    """Tests compiling of loss when it is
    given as an dict of losses
    mapping to outputs.
    """

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    # Test dict with global routed param
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss={"out1": loss},
        loss__from_logits=True,  # default is False
    )
    est.fit(X, y)
    got = _class_from_strings(loss, "loss")
    if isclass(got):
        # from_logits should have been passed and set
        assert est.model_.loss["out1"].from_logits == True
    else:
        # at least check the name
        assert est.model_.loss["out1"].__name__ == got.__name__

    # Test dict with key-based routed param
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss={"out1": loss},
        loss__from_logits=False,
        loss__out1__from_logits=True,  # should override above
    )
    est.fit(X, y)
    got = _class_from_strings(loss, "loss")
    if isclass(got):
        # from_logits should have been passed and set
        assert est.model_.loss["out1"].from_logits == True
    else:
        # at least check the name
        assert est.model_.loss["out1"].__name__ == got.__name__


@pytest.mark.parametrize(
    "metrics", ["binary_accuracy", metrics_module.BinaryAccuracy]
)
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_metrics_single_metric_per_output(metrics, n_outputs_):
    """Metrics without the ("name", metric, "output")
    syntax should ignore all routed and custom options.

    This tests a single metric per output.
    """

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    # loss functions for each output and joined show up as metrics
    metric_idx = 1 + (n_outputs_ if n_outputs_ > 1 else 0)
    prefix = "out1_" if n_outputs_ > 1 else ""

    expected_name = metrics
    if isclass(metrics):
        # Test discovery in VSCode fails if TF prints out _any_
        # warnings during discovery
        # (which of course it does if you try to instantiate anything)
        metrics = metrics()
        expected_name = metrics.name

    # List of metrics
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=[metrics],
        metrics__name="custom_name",  # should be ignored
        metrics__0__name="custom_name",  # should be ignored
    )
    with pytest.warns(
        UserWarning, match="SciKeras does not know how to compile"
    ):
        est.fit(X, y)
    assert est.model_.metrics[metric_idx].name == prefix + expected_name

    # List of lists of metrics
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=[[metrics] for _ in range(n_outputs_)],
        metrics__name="custom_name",  # should be ignored
        metrics__0__name="custom_name",  # should be ignored
        metrics__0__0__name="custom_name",  # should be ignored
    )
    with pytest.warns(
        UserWarning, match="SciKeras does not know how to compile"
    ):
        est.fit(X, y)
    assert prefix + expected_name in est.model_.metrics[metric_idx].name
    assert "custom_name" not in est.model_.metrics[metric_idx].name

    # Dict of metrics
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics={"out1": metrics},
        metrics__name="custom_name",  # should be ignored
        metrics__out1__name="custom_name",  # should be ignored
    )
    with pytest.warns(
        UserWarning, match="SciKeras does not know how to compile"
    ):
        est.fit(X, y)
    assert prefix + expected_name in est.model_.metrics[metric_idx].name
    assert "custom_name" not in est.model_.metrics[metric_idx].name

    # Dict of lists
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics={"out1": metrics},
        metrics__name="custom_name",  # should be ignored
        metrics__out1__name="custom_name",  # should be ignored
        metrics__out1__0__name="custom_name",  # should be ignored
    )
    with pytest.warns(
        UserWarning, match="SciKeras does not know how to compile"
    ):
        est.fit(X, y)
    assert prefix + expected_name in est.model_.metrics[metric_idx].name
    assert "custom_name" not in est.model_.metrics[metric_idx].name


@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_metrics_two_metric_per_output(n_outputs_):
    """Metrics without the ("name", metric, "output")
    syntax should ignore all routed and custom options.

    This tests multiple (two) metrics per output.
    """

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    metric_class = metrics_module.BinaryAccuracy

    # loss functions for each output and joined show up as metrics
    metric_idx = 1 + (n_outputs_ if n_outputs_ > 1 else 0)

    # List of lists of metrics
    if n_outputs_ == 1:
        metrics_ = [metric_class(name="1"), metric_class(name="2")]
    else:
        metrics_ = [
            [metric_class(name="1"), metric_class(name="2")]
            for _ in range(n_outputs_)
        ]

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=metrics_,
        metrics__name="custom_name",  # should be ignored
        metrics__1__name="custom_name",  # should be ignored
        metrics__1__1__name="custom_name",  # should be ignored
    )
    with pytest.warns(
        UserWarning, match="SciKeras does not know how to compile"
    ):
        est.fit(X, y)
    if n_outputs_ == 1:
        assert est.model_.metrics[metric_idx].name == "1"
    else:
        # For multi-output models, Keras pre-appends the output name
        assert est.model_.metrics[metric_idx].name == "out1_1"

    # List of lists of metrics
    if n_outputs_ == 1:
        metrics_ = {"out1": [metric_class(name="1"), metric_class(name="2")]}
    else:
        metrics_ = {
            f"out{i+1}": [metric_class(name="1"), metric_class(name="2")]
            for i in range(n_outputs_)
        }

    # Dict of metrics
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=metrics_,
        metrics__name="custom_name",  # should be ignored
        metrics__out1__name="custom_name",  # should be ignored
    )
    with pytest.warns(
        UserWarning, match="SciKeras does not know how to compile"
    ):
        est.fit(X, y)
    if n_outputs_ == 1:
        assert est.model_.metrics[metric_idx].name == "1"
    else:
        # For multi-output models, Keras pre-appends the output name
        assert est.model_.metrics[metric_idx].name == "out1_1"


@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_metrics_iterable(n_outputs_):

    metrics = metrics_module.BinaryAccuracy

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    # loss functions for each output and joined show up as metrics
    metric_idx = 1 + (n_outputs_ if n_outputs_ > 1 else 0)

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=[metrics],
        metrics__0__name="custom_name",
    )
    est.fit(X, y)
    if n_outputs_ == 1:
        assert est.model_.metrics[metric_idx].name == "custom_name"
    else:
        assert est.model_.metrics[metric_idx].name == "out1_custom_name"

    if n_outputs_ == 1:
        metrics_ = [
            metrics,
        ]
    else:
        metrics_ = [metrics for _ in range(n_outputs_)]
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=metrics_,
        metrics__name="name_all_metrics",  # ends up in index 1 only
        metrics__0__name="custom_name",  # ends up in index 0 only
    )
    est.fit(X, y)
    if n_outputs_ == 1:
        assert est.model_.metrics[metric_idx].name == "custom_name"
    else:
        assert est.model_.metrics[metric_idx].name == "out1_custom_name"
        assert (
            est.model_.metrics[metric_idx + 1].name == "out1_name_all_metrics"
        )
        assert est.model_.metrics[metric_idx + 2].name == "out2_custom_name"
        assert (
            est.model_.metrics[metric_idx + 3].name == "out2_name_all_metrics"
        )


def test_metrics_dict():
    n_outputs_ = 2

    metrics = metrics_module.BinaryAccuracy

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    # loss functions for each output and joined show up as metrics
    metric_idx = 1 + n_outputs_

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics={"out1": metrics},
        metrics__out1__name="custom_name",
    )
    est.fit(X, y)
    assert est.model_.metrics[metric_idx].name == "out1_custom_name"

    if n_outputs_ == 1:
        metrics_ = ({"out1": metrics},)
    else:
        metrics_ = {f"out{i+1}": metrics for i in range(n_outputs_)}
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=metrics_,
        metrics__name="name_all_metrics",  # ends up out2 only
        metrics__out1__name="custom_name",  # ends up in out1 only
    )
    est.fit(X, y)
    assert est.model_.metrics[metric_idx].name == "out1_custom_name"
    assert est.model_.metrics[metric_idx + 1].name == "out2_name_all_metrics"
