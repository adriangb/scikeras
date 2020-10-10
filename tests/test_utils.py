import numpy as np
import pytest

from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module

from scikeras.utils import loss_name, metric_name


class CustomLoss(losses_module.Loss):
    pass


class NotALoss:
    pass


@pytest.mark.parametrize(
    "loss,expected,raises",
    [
        ("categorical_crossentropy", "categorical_crossentropy", None),
        ("CategoricalCrossentropy", "CategoricalCrossentropy", None),
        (losses_module.categorical_crossentropy, "categorical_crossentropy", None),
        (losses_module.CategoricalCrossentropy, "CategoricalCrossentropy", None),
        (losses_module.CategoricalCrossentropy(), "CategoricalCrossentropy", None),
        (object(), "", pytest.raises(ValueError, match="Unable to determine name")),
        (object, "", pytest.raises(ValueError, match="Unable to determine name")),
        (
            "unknown_loss",
            "",
            pytest.raises(ValueError, match="Unable to determine name"),
        ),
        (CustomLoss, "CustomLoss", None),
        (CustomLoss(), "CustomLoss", None),
        (NotALoss, "", pytest.raises(ValueError, match="Unable to determine name")),
        (NotALoss(), "", pytest.raises(ValueError, match="Unable to determine name")),
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


class NotAMetric:
    pass


@pytest.mark.parametrize(
    "metric,expected,raises",
    [
        ("categorical_crossentropy", "categorical_crossentropy", None),
        ("CategoricalCrossentropy", "CategoricalCrossentropy", None),
        (metrics_module.categorical_crossentropy, "categorical_crossentropy", None),
        (metrics_module.CategoricalCrossentropy, "CategoricalCrossentropy", None),
        (metrics_module.CategoricalCrossentropy(), "CategoricalCrossentropy", None),
        (object(), "", pytest.raises(ValueError, match="Unable to determine name")),
        (object, "", pytest.raises(ValueError, match="Unable to determine name")),
        (
            "unknown_metric",
            "",
            pytest.raises(ValueError, match="Unable to determine name"),
        ),
        (CustomMetric, "CustomMetric", None),
        (CustomMetric(), "CustomMetric", None),
        (NotAMetric, "", pytest.raises(ValueError, match="Unable to determine name")),
        (NotAMetric(), "", pytest.raises(ValueError, match="Unable to determine name")),
    ],
)
def test_metric_name(metric, expected, raises):
    if raises:
        with raises:
            got = metric_name(metric)
    else:
        got = metric_name(metric)
        assert got == expected
