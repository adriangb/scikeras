import numpy as np
import pytest

from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module

from scikeras.utils import loss_name, metric_name, type_of_target


@pytest.mark.parametrize(
    "y, expected",
    [
        (np.array([[1, 0], [0, 1]]), "multiclass-one-hot"),
        (np.array([[1, 0], [1, 1]]), "multilabel-indicator"),
    ],
)
def test_type_of_target(y, expected):
    got = type_of_target(y)
    assert got == expected


class CustomMetric(metrics_module.Metric):
    pass


class CustomLoss(losses_module.Loss):
    pass


@pytest.mark.parametrize(
    "loss,expected,raises",
    [
        ("categorical_crossentropy", "categorical_crossentropy", None),
        ("CategoricalCrossentropy", "CategoricalCrossentropy", None),
        (losses_module.categorical_crossentropy, "categorical_crossentropy", None),
        (losses_module.CategoricalCrossentropy, "CategoricalCrossentropy", None),
        (losses_module.CategoricalCrossentropy(), "CategoricalCrossentropy", None),
        (object(), "", pytest.raises(ValueError, match="Unknown loss")),
        ("unknown_loss", "", pytest.raises(ValueError, match="Unknown loss")),
    ],
)
def test_loss_name(loss, expected, raises):
    if raises:
        with raises:
            got = loss_name(loss)
    else:
        got = loss_name(loss)
        assert got == expected


@pytest.mark.parametrize(
    "metric,expected,raises",
    [
        ("categorical_crossentropy", "categorical_crossentropy", None),
        ("CategoricalCrossentropy", "CategoricalCrossentropy", None),
        (metrics_module.categorical_crossentropy, "categorical_crossentropy", None),
        (metrics_module.CategoricalCrossentropy, "CategoricalCrossentropy", None),
        (metrics_module.CategoricalCrossentropy(), "CategoricalCrossentropy", None),
        (object(), "", pytest.raises(ValueError, match="Unknown metric")),
        ("unknown_metric", "", pytest.raises(ValueError, match="Unknown metric")),
    ],
)
def test_metric_name(metric, expected, raises):
    if raises:
        with raises:
            got = metric_name(metric)
    else:
        got = metric_name(metric)
        assert got == expected
