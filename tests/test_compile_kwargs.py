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
from tensorflow.python.keras.engine.compile_utils import (
    losses_mod,
    metrics_mod,
)

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


def test_optimizer_invalid_string():
    """Tests that a ValueError is raised when an unknown
    string is passed as an optimizer.
    """

    X, y = make_classification()

    optimizer = "sgf"  # sgf is not a loss

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer=optimizer,
        loss="binary_crossentropy",
    )
    with pytest.raises(ValueError, match=f"Unknown optimizer: {optimizer}"):
        est.fit(X, y)


def test_compiling_of_routed_parameters():
    """Tests that routed parameters
    can themselves be compiled.
    """

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
    "loss", (losses_module.BinaryCrossentropy, "BinaryCrossentropy",),
)
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_loss(loss, n_outputs_):
    """Tests compiling of single loss
    using routed parameters.
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


def test_loss_invalid_string():
    """Tests that a ValueError is raised when an unknown
    string is passed as a loss.
    """

    X, y = make_classification()

    loss = "binary_crossentropr"  # binary_crossentropr is not a loss

    est = KerasClassifier(
        model=get_model, num_hidden=20, optimizer="sgd", loss=loss,
    )
    with pytest.raises(ValueError, match=f"Unknown loss function: {loss}"):
        est.fit(X, y)


def test_loss_uncompilable():
    """Tests that a TypeError is raised when a loss
    that is not compilable is passed routed parameters.
    """

    X, y = make_classification()

    loss = losses_module.binary_crossentropy

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss=loss,
        loss__from_logits=True,
    )
    expected_param = {"from_logits": True}
    with pytest.raises(
        TypeError,
        match=f'TypeError: "{str(loss)}" object of type "{type(loss)}" could not be called with'
        f" parameters {expected_param}",
    ):
        est.fit(X, y)


@pytest.mark.parametrize(
    "loss", (losses_module.BinaryCrossentropy, "BinaryCrossentropy",),
)
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_loss_routed_params_iterable(loss, n_outputs_):
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
    assert est.model_.loss[0].from_logits

    # Test iterable with index-based routed param
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss=[loss],
        loss__from_logits=True,
        loss__0__from_logits=False,  # should override above
    )
    est.fit(X, y)
    assert est.model_.loss[0].from_logits == False


@pytest.mark.parametrize(
    "loss", (losses_module.BinaryCrossentropy, "BinaryCrossentropy",),
)
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_loss_routed_params_dict(loss, n_outputs_):
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
    assert est.model_.loss["out1"].from_logits == True

    # Test dict with key-based routed param
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss={"out1": loss},
        loss__from_logits=True,
        loss__out1__from_logits=False,  # should override above
    )
    est.fit(X, y)
    assert est.model_.loss["out1"].from_logits == False


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
    if metrics != "binary_accuracy":
        # Test discovery in VSCode fails if TF prints out _any_
        # warnings during discovery
        # (which of course it does if you try to instantiate anything)
        expected_name = metrics().name

    # List of metrics
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=[metrics if not isclass(metrics) else metrics()],
    )
    est.fit(X, y)
    assert est.model_.metrics[metric_idx].name == prefix + expected_name

    # List of lists of metrics
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=[
            [metrics if not isclass(metrics) else metrics()]
            for _ in range(n_outputs_)
        ],
    )
    est.fit(X, y)
    assert prefix + expected_name == est.model_.metrics[metric_idx].name

    # Dict of metrics
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics={
            f"out{i+1}": metrics if not isclass(metrics) else metrics()
            for i in range(n_outputs_)
        },
    )
    est.fit(X, y)
    assert prefix + expected_name == est.model_.metrics[metric_idx].name

    # Dict of lists
    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics={
            f"out{i+1}": metrics if not isclass(metrics) else metrics()
            for i in range(n_outputs_)
        },
    )
    est.fit(X, y)
    assert prefix + expected_name == est.model_.metrics[metric_idx].name


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
    )
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
    )
    est.fit(X, y)
    if n_outputs_ == 1:
        assert est.model_.metrics[metric_idx].name == "1"
    else:
        # For multi-output models, Keras pre-appends the output name
        assert est.model_.metrics[metric_idx].name == "out1_1"


@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_metrics_routed_params_iterable(n_outputs_):
    """Tests compiling metrics with routed parameters
    when they are passed as an iterable.
    """

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


def test_metrics_routed_params_dict():
    """Tests compiling metrics with routed parameters
    when they are passed as a dict.
    """
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


def test_metrics_invalid_string():
    """Tests that a ValueError is raised when an unknown
    string is passed as a metric.
    """

    X, y = make_classification()

    metrics = [
        "acccuracy",
    ]  # acccuracy (extra `c`) is not a metric

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=metrics,
    )
    with pytest.raises(
        ValueError, match=f"Unknown metric function: {metrics[0]}"
    ):
        est.fit(X, y)


def test_metrics_uncompilable():
    """Tests that a TypeError is raised when a metric
    that is not compilable is passed routed parameters.
    """

    X, y = make_classification()

    metrics = [
        metrics_module.get("accuracy"),
    ]  # a function

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=metrics,
        metrics__name="custom_name",
    )
    expected_param = {"name": "custom_name"}
    with pytest.raises(
        TypeError,
        match=f'TypeError: "{str(metrics[0])}" object of type "{type(metrics[0])}" could not be called with'
        f" parameters {expected_param}",
    ):
        est.fit(X, y)
