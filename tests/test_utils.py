import numpy as np
import pytest

from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module

from scikeras.utils import loss_name, metric_name


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
        (CustomLoss, "CustomLoss", None),
        (CustomLoss(), "CustomLoss", None),
    ],
)
def test_loss_name(loss, expected, raises):
    if raises:
        with raises:
            got = loss_name(loss)
    else:
        got = loss_name(loss)
        assert got == expected


class CustomMetric(metrics_module.AUC):
    pass


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
        (CustomMetric, "CustomMetric", None),
        (CustomMetric(), "CustomMetric", None),
    ],
)
def test_metric_name(metric, expected, raises):
    if raises:
        with raises:
            got = metric_name(metric)
    else:
        got = metric_name(metric)
        assert got == expected
