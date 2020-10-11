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
        ("CategoricalCrossentropy", "categorical_crossentropy", None),
        (losses_module.categorical_crossentropy, "categorical_crossentropy", None),
        (losses_module.CategoricalCrossentropy, "categorical_crossentropy", None),
        (losses_module.CategoricalCrossentropy(), "categorical_crossentropy", None),
        (object(), "", pytest.raises(TypeError, match="`loss` must be a")),
        (object, "", pytest.raises(TypeError, match="`loss` must be a")),
        (list(), "", pytest.raises(TypeError, match="`loss` must be a")),
        ("unknown_loss", "", pytest.raises(ValueError, match="Unknown loss function"),),
        (CustomLoss, "custom_loss", None),
        (CustomLoss(), "custom_loss", None),
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
        ("CategoricalCrossentropy", "categorical_crossentropy", None),
        (metrics_module.categorical_crossentropy, "categorical_crossentropy", None),
        (metrics_module.CategoricalCrossentropy, "categorical_crossentropy", None),
        (metrics_module.CategoricalCrossentropy(), "categorical_crossentropy", None),
        (object(), "", pytest.raises(TypeError, match="`metric` must be a")),
        (object, "", pytest.raises(TypeError, match="`metric` must be a")),
        (list(), "", pytest.raises(TypeError, match="`metric` must be a")),
        (
            "unknown_metric",
            "",
            pytest.raises(ValueError, match="Unknown metric function"),
        ),
        (CustomMetric, "custom_metric", None),
        (CustomMetric(), "custom_metric", None),
    ],
)
def test_metric_name(metric, expected, raises):
    if raises:
        with raises:
            got = metric_name(metric)
    else:
        got = metric_name(metric)
        assert got == expected
