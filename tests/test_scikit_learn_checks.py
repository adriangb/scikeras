"""Tests using Scikit-Learn's bundled estimator_checks."""

from distutils.version import LooseVersion
from typing import Any, Dict

import numpy as np
import pytest

from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_no_attributes_set_in_init
from tensorflow.keras import Model, Sequential, layers

from scikeras.wrappers import KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor
from .testing_utils import basic_checks, parametrize_with_checks


@parametrize_with_checks(
    estimators=[
        KerasClassifier(
            build_fn=dynamic_classifier,
            # Set batch size to a large number
            # (larger than X.shape[0] is the goal)
            # if batch_size < X.shape[0], results will very
            # slightly if X is shuffled.
            # This is only required for this tests and is not really
            # applicable to real world datasets
            batch_size=100,
            optimizer="adam",
            model__hidden_layer_sizes=[]
            epochs=1,
        ),
        KerasRegressor(
            build_fn=dynamic_regressor,
            # Set batch size to a large number
            # (larger than X.shape[0] is the goal)
            # if batch_size < X.shape[0], results will very
            # slightly if X is shuffled.
            # This is only required for this tests and is not really
            # applicable to real world datasets
            batch_size=100,
            optimizer="adam",
            loss=KerasRegressor.r_squared,
            model__hidden_layer_sizes=[],
            epochs=1,
        ),
    ],
    ids=["KerasClassifier", "KerasRegressor"],
)
def test_fully_compliant_estimators(estimator, check):
    if sklearn_version <= LooseVersion("0.23.0") and check.func.__name__ in (
        "check_classifiers_predictions",
        "check_classifiers_classes",
        "check_methods_subset_invariance",
        "check_no_attributes_set_in_init",
    ):
        # These tests have bugs that are fixed in 0.23.0
        pytest.skip("This test is broken in sklearn<0.23.0")
    check(estimator)


class SubclassedClassifier(KerasClassifier):
    def __init__(
        self, model__hidden_layer_sizes=(100,), metrics=None, loss=None, **kwargs,
    ):
        super().__init__(**kwargs)
        self.model__hidden_layer_sizes = model__hidden_layer_sizes
        self.metrics = metrics
        self.loss = loss
        self.optimizer = "sgd"

    def _keras_build_fn(
        self, hidden_layer_sizes, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
    ) -> Model:
        return dynamic_classifier(
            hidden_layer_sizes=hidden_layer_sizes,
            meta=meta,
            compile_kwargs=compile_kwargs,
        )


def test_no_attributes_set_init_sublcassed():
    """Tests that subclassed models can be made that
    set all parameters in a single __init__
    """
    estimator = SubclassedClassifier()
    check_no_attributes_set_in_init(estimator.__name__, estimator)
    basic_checks(estimator, load_iris)


def test_no_attributes_set_init_no_args():
    """Tests that models with no build arguments
    set all parameters in a single __init__
    """

    def build_fn():
        model = Sequential()
        model.add(layers.Dense(1, input_dim=1, activation="relu"))
        model.add(layers.Dense(1))
        model.compile(loss="mse")
        return model

    estimator = KerasRegressor(model=build_fn)
    check_no_attributes_set_in_init(estimator.__name__, estimator)
    estimator.fit([[1]], [1])
