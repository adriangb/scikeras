import numpy as np
import pytest
from sklearn.datasets import make_classification
from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module
from tensorflow.keras import optimizers as optimizers_module
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from scikeras.wrappers import KerasClassifier
from tests.multi_output_models import MultiOutputClassifier


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
        assert float(est_opt.momentum) == pytest.approx(0.5)
        assert float(est_opt.learning_rate) == pytest.approx(0.15, abs=1e-6)
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
    with pytest.raises(ValueError, match="Unknown optimizer"):
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
    with pytest.raises(ValueError, match="Unknown loss function"):
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


@pytest.mark.parametrize("metrics", ["binary_accuracy", metrics_module.BinaryAccuracy])
@pytest.mark.parametrize("n_outputs_", (1, 2))
def test_metrics_single_metric_per_output(metrics, n_outputs_):
    """Test a single metric per output using vanilla
    Keras sytnax and without any routed paramters.
    """

    X, y = make_classification()
    y = np.column_stack([y for _ in range(n_outputs_)]).squeeze()

    # loss functions for each output and joined show up as metrics
    metric_idx = 1 + (n_outputs_ if n_outputs_ > 1 else 0)
    prefix = "out1_" if n_outputs_ > 1 else ""

    if isinstance(metrics, str):
        expected_name = metrics
    else:
        expected_name = metrics().name

    # List of metrics
    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics=[
            metrics if not isinstance(metrics, metrics_module.Metric) else metrics()
        ],
    )
    est.fit(X, y)
    assert est.model_.metrics[metric_idx].name == prefix + expected_name

    # List of lists of metrics
    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics=[
            [metrics if not isinstance(metrics, metrics_module.Metric) else metrics()]
            for _ in range(n_outputs_)
        ],
    )
    est.fit(X, y)
    assert prefix + expected_name == est.model_.metrics[metric_idx].name

    # Dict of metrics
    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics={
            f"out{i+1}": metrics
            if not isinstance(metrics, metrics_module.Metric)
            else metrics()
            for i in range(n_outputs_)
        },
    )
    est.fit(X, y)
    assert prefix + expected_name == est.model_.metrics[metric_idx].name

    # Dict of lists
    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics={
            f"out{i+1}": metrics
            if not isinstance(metrics, metrics_module.Metric)
            else metrics()
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
            [metric_class(name="1"), metric_class(name="2")] for _ in range(n_outputs_)
        ]

    est = MultiOutputClassifier(
        model=get_model,
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
    est = MultiOutputClassifier(
        model=get_model,
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

    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics=[metrics],
        metrics__0__name="custom_name",
    )
    est.fit(X, y)
    compiled_metrics = est.model_.metrics
    if n_outputs_ == 1:
        assert compiled_metrics[metric_idx].name == "custom_name"
    else:
        assert compiled_metrics[metric_idx].name == "out1_custom_name"

    if n_outputs_ == 1:
        metrics_ = [
            metrics,
        ]
    else:
        metrics_ = [metrics for _ in range(n_outputs_)]
    est = MultiOutputClassifier(
        model=get_model,
        loss="binary_crossentropy",
        metrics=metrics_,
        metrics__name="name_all_metrics",  # ends up in index 1 only
        metrics__0__name="custom_name",  # ends up in index 0 only
    )
    est.fit(X, y)
    compiled_metrics = est.model_.metrics
    if n_outputs_ == 1:
        assert compiled_metrics[metric_idx].name == "custom_name"
    else:
        assert compiled_metrics[metric_idx].name == "out1_custom_name"
        assert compiled_metrics[metric_idx + 1].name == "out1_name_all_metrics"
        assert compiled_metrics[metric_idx + 2].name == "out2_custom_name"
        assert compiled_metrics[metric_idx + 3].name == "out2_name_all_metrics"


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

    est = MultiOutputClassifier(
        model=get_model,
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
    est = MultiOutputClassifier(
        model=get_model,
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
        loss="binary_crossentropy",
        metrics=metrics,
    )
    with pytest.raises(ValueError, match="Unknown metric function"):
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
