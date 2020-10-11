import numpy as np
import pytest

from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module

from scikeras.utils import loss_name, metric_name


class CustomLoss(losses_module.Loss):
    pass


class CustomMetric(metrics_module.AUC):
    pass


@pytest.mark.parametrize(
    "obj",
    [
        "categorical_crossentropy",
        "CategoricalCrossentropy",
        losses_module.categorical_crossentropy,
        losses_module.CategoricalCrossentropy,
        losses_module.CategoricalCrossentropy(),
    ],
)
def test_loss_invariance(obj):
    """Test to make sure loss_name returns same string no matter which object
    is passed (str, function, class, type)"""
    assert loss_name(obj) == "categorical_crossentropy"


@pytest.mark.parametrize("obj", [CustomLoss, CustomLoss()])
def test_custom_loss(obj):
    assert loss_name(obj) == "custom_loss"


@pytest.mark.parametrize(
    "obj",
    [
        "categorical_crossentropy",
        "CategoricalCrossentropy",
        metrics_module.categorical_crossentropy,
        metrics_module.CategoricalCrossentropy,
        metrics_module.CategoricalCrossentropy(),
    ],
)
def test_metric_invariance(obj):
    """Test to make sure same metric returned no matter which object passed"""
    assert metric_name(obj) == "categorical_crossentropy"


@pytest.mark.parametrize("loss", [object(), object, list()])
def test_loss_types(loss):
    with pytest.raises(TypeError, match="`loss` must be a"):
        loss_name(loss)


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="Unknown loss function"):
        loss_name("unknown_loss")


@pytest.mark.parametrize("obj", [object(), object, list()])
def test_metric_types(obj):
    with pytest.raises(TypeError, match="`metric` must be a"):
        metric_name(obj)


def test_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metric function"):
        metric_name("unknown_metric")


@pytest.mark.parametrize("metric", [CustomMetric, CustomMetric()])
def test_custom_metric(metric):
    assert metric_name(metric) == "custom_metric"
