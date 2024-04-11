from __future__ import annotations

import numpy as np
import pytest
from keras import losses as losses_module
from keras import metrics as metrics_module
from keras import optimizers as optimizers_module
from keras.layers import Dense, Input
from keras.models import Model
from keras.src.backend.common.variables import KerasVariable
from sklearn.datasets import make_classification

from scikeras.wrappers import KerasClassifier
from tests.multi_output_models import MultiOutputClassifier
from tests.testing_utils import get_metric_names


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
def test_optimizer(optimizer):
    """Tests compiling of single optimizer with options.
    Since there can only ever be a single optimizer, there is no
    ("name", optimizer, "output") option.
    Only optimizer classes will be compiled with custom options,
    all others (class names, function names) should pass through
    untouched.
    """
    # Single output
    X, y = make_classification()

    est = KerasClassifier(
        model=get_model,
        optimizer=optimizer,
        optimizer__learning_rate=0.15,
        optimizer__momentum=0.5,
        loss="binary_crossentropy",
    )
    est.fit(X, y)
    est_opt = est.model_.optimizer
    if not isinstance(optimizer, str):
        momentum = est_opt.momentum
        if isinstance(momentum, KerasVariable):
            momentum = momentum.numpy()
        assert float(momentum) == pytest.approx(0.5)
        lr = est_opt.learning_rate
        if isinstance(lr, KerasVariable):
            lr = lr.numpy()
        assert lr == pytest.approx(0.15, abs=1e-6)
    else:
        assert est_opt.__class__ == optimizers_module.get(optimizer).__class__


def test_optimizer_invalid_string():
    """Tests that a ValueError is raised when an unknown
    string is passed as an optimizer.
    """

    X, y = make_classification()

    optimizer = "sgf"  # sgf is not a loss

    est = KerasClassifier(
        model=get_model,
        optimizer=optimizer,
        loss="binary_crossentropy",
    )
    with pytest.raises(ValueError, match="Could not interpret optimizer"):
        est.fit(X, y)


def test_compiling_of_routed_parameters():
    """Tests that routed parameters
    can themselves be compiled.
    """

    X, y = make_classification()

    class Foo:
        got = {}

        def __init__(self, foo_kwarg="foo_kwarg_default"):
            self.foo_kwarg = foo_kwarg

    class MyLoss(losses_module.Loss):
        def __init__(self, param1="param1_default", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.param1 = param1

        def __call__(self, y_true, y_pred, sample_weight=None):
            return losses_module.binary_crossentropy(y_true, y_pred)

    est = KerasClassifier(
        model=get_model,
        loss=MyLoss,
        loss__param1=[Foo, Foo],
        loss__param1__foo_kwarg=1,
        loss__param1__0__foo_kwarg=2,
    )
    est.fit(X, y)
    assert est.model_.loss.param1[0].foo_kwarg == 2
    assert est.model_.loss.param1[1].foo_kwarg == 1


@pytest.mark.parametrize(
    "loss",
    (
        losses_module.BinaryCrossentropy,
        "BinaryCrossentropy",
    ),
)
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_loss(loss, n_outputs_):
    """Tests compiling of single loss
    using routed parameters.
    """

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    est = MultiOutputClassifier(model=get_model, loss=loss, loss__name="custom_name")
    est.fit(X, y)
    assert str(loss) in str(est.model_.loss) or isinstance(est.model_.loss, loss)


def test_loss_invalid_string():
    """Tests that a ValueError is raised when an unknown
    string is passed as a loss.
    """

    X, y = make_classification()

    loss = "binary_crossentropr"  # binary_crossentropr is not a loss

    est = KerasClassifier(
        model=get_model,
        num_hidden=20,
        loss=loss,
    )
    with pytest.raises(ValueError, match="Could not interpret loss"):
        est.fit(X, y)


def test_loss_uncompilable():
    """Tests that a TypeError is raised when a loss
    that is not compilable is passed routed parameters.
    """

    X, y = make_classification()

    loss = losses_module.binary_crossentropy

    est = KerasClassifier(
        model=get_model,
        loss=loss,
        loss__from_logits=True,
    )
    with pytest.raises(
        TypeError, match="does not accept parameters because it's not a class"
    ):
        est.fit(X, y)


@pytest.mark.parametrize(
    "loss",
    (
        losses_module.BinaryCrossentropy,
        "BinaryCrossentropy",
    ),
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
    est = MultiOutputClassifier(
        model=get_model,
        loss=[loss] * (y.shape[1] if len(y.shape) == 2 else 1),
        loss__from_logits=True,  # default is False
    )
    est.fit(X, y)
    assert est.model_.loss[0].from_logits

    # Test iterable with index-based routed param
    est = MultiOutputClassifier(
        model=get_model,
        loss=[loss] * (y.shape[1] if len(y.shape) == 2 else 1),
        loss__from_logits=True,
        loss__0__from_logits=False,  # should override above
    )
    est.fit(X, y)
    assert est.model_.loss[0].from_logits is False


@pytest.mark.parametrize(
    "loss",
    (
        losses_module.BinaryCrossentropy,
        "BinaryCrossentropy",
    ),
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
    est = MultiOutputClassifier(
        model=get_model,
        loss={"out1": loss},
        loss__from_logits=True,  # default is False
    )
    est.fit(X, y)
    assert est.model_.loss["out1"].from_logits is True

    # Test dict with key-based routed param
    est = MultiOutputClassifier(
        model=get_model,
        loss={"out1": loss},
        loss__from_logits=True,
        loss__out1__from_logits=False,  # should override above
    )
    est.fit(X, y)
    assert est.model_.loss["out1"].from_logits is False


@pytest.mark.parametrize(
    "metric",
    [
        "binary_accuracy",
        metrics_module.BinaryAccuracy,
        metrics_module.BinaryAccuracy(name="custom_name"),
    ],
)
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_metrics_single_metric_per_output(
    metric: str | metrics_module.Metric | type[metrics_module.Metric], n_outputs_: int
):
    """Test a single metric per output using vanilla
    Keras sytnax and without any routed paramters.
    """

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    metric_value = (
        metric if isinstance(metric, (metrics_module.Metric, str)) else metric()
    )

    if isinstance(metric_value, str):
        expected_name = metric
    else:
        expected_name = metric_value.name

    if n_outputs_ == 1:
        # List of metrics, not supported for multiple outputs where each output is required to get
        # its own metrics if passing metrics as a list
        est = MultiOutputClassifier(
            model=get_model,
            loss="binary_crossentropy",
            metrics=[metric_value],
        )
        est.fit(X, y)
        assert get_metric_names(est) == [expected_name]
    else:
        # List of lists of metrics, only supported if we have multiple outputs
        est = MultiOutputClassifier(
            model=get_model,
            loss="binary_crossentropy",
            metrics=[[metric_value]] * n_outputs_,
        )
        est.fit(X, y)
        assert get_metric_names(est) == [expected_name] * n_outputs_

    # Dict of metrics
    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics={f"out{i+1}": metric_value for i in range(n_outputs_)},
    )
    est.fit(X, y)
    assert get_metric_names(est) == [expected_name] * n_outputs_

    # Dict of lists
    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics={f"out{i+1}": [metric_value] for i in range(n_outputs_)},
    )
    est.fit(X, y)
    assert get_metric_names(est) == [expected_name] * n_outputs_


@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_metrics_two_metric_per_output(n_outputs_: int):
    """Metrics without the ("name", metric, "output")
    syntax should ignore all routed and custom options.

    This tests multiple (two) metrics per output.
    """

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    metric_class = metrics_module.BinaryAccuracy

    metrics_value = [metric_class(name="1"), metric_class(name="2")]

    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics=metrics_value if n_outputs_ == 1 else [metrics_value] * n_outputs_,
    )
    est.fit(X, y)
    assert get_metric_names(est) == ["1", "2"] * n_outputs_

    # Dict of metrics
    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics={f"out{i+1}": metrics_value for i in range(n_outputs_)},
    )
    est.fit(X, y)
    assert get_metric_names(est) == ["1", "2"] * n_outputs_


@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_metrics_routed_params_iterable(n_outputs_: int):
    """Tests compiling metrics with routed parameters when they are passed as an iterable."""

    metrics = metrics_module.BinaryAccuracy

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics=[metrics] * n_outputs_,
        metrics__0__name="custom_name",
    )
    est.fit(X, y)
    expected = (
        ["custom_name", "binary_accuracy"] if n_outputs_ == 2 else ["custom_name"]
    )
    assert get_metric_names(est) == expected

    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics=[metrics] * n_outputs_,
        metrics__name="name_all_metrics",  # ends up in index 1 only
        metrics__0__name="custom_name",  # ends up in index 0 only
    )
    est.fit(X, y)
    expected = (
        ["custom_name", "name_all_metrics"] if n_outputs_ == 2 else ["custom_name"]
    )
    assert get_metric_names(est) == expected, get_metric_names(est)


def test_metrics_routed_params_dict():
    """Tests compiling metrics with routed parameters
    when they are passed as a dict.
    """
    n_outputs_ = 2

    metrics = metrics_module.BinaryAccuracy

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics={"out1": metrics, "out2": metrics},
        metrics__out1__name="custom_name1",
        metrics__out2__name="custom_name2",
    )
    est.fit(X, y)
    assert get_metric_names(est) == ["custom_name1", "custom_name2"]

    if n_outputs_ == 1:
        metrics_ = ({"out1": metrics},)
    else:
        metrics_ = {f"out{i+1}": metrics for i in range(n_outputs_)}
    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics=metrics_,
        metrics__name="name_all_metrics",  # ends up out2 only
        metrics__out1__name="custom_name",  # ends up in out1 only
    )
    est.fit(X, y)
    assert get_metric_names(est) == ["custom_name", "name_all_metrics"]


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
        loss="binary_crossentropy",
        metrics=metrics,
    )
    with pytest.raises(ValueError, match="Could not interpret metric identifier"):
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
        loss="binary_crossentropy",
        metrics=metrics,
        metrics__name="custom_name",
    )
    with pytest.raises(
        TypeError, match="does not accept parameters because it's not a class"
    ):
        est.fit(X, y)
