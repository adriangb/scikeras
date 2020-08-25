import inspect

from distutils.version import LooseVersion
from typing import Any
from typing import Dict

import numpy as np
import pytest
import tensorflow as tf

from tensorflow.keras import Model

from scikeras.wrappers import BaseWrapper
from scikeras.wrappers import KerasClassifier
from scikeras.wrappers import KerasRegressor

from .mlp_models import dynamic_classifier
from .mlp_models import dynamic_regressor


def test_routing_basic():
    """Tests that parameters are routed to the correct place based on static
    dictionaries.
    """
    n, d = 20, 3
    n_classes = 3
    X = np.random.uniform(size=(n, d)).astype(float)
    y = np.random.choice(n_classes, size=n).astype(int)

    def build_fn(*args, **kwargs):
        assert len(args) == 0, "No *args should be passed to `build_fn`"
        assert tuple(kwargs.keys()) == (
            "hidden_layer_sizes",
            "meta_params",
            "compile_params",
        ), "The number and order of **kwargs passed to `build_fn` should be fixed"
        meta = set(kwargs["meta_params"].keys()) - {"X", "y"}
        expected_meta = KerasClassifier._meta_params - {
            "model_",
            "history_",
            "is_fitted_",
        }
        assert meta == expected_meta
        assert set(kwargs["compile_params"].keys()).issubset(
            KerasClassifier._compile_params
        )
        return dynamic_classifier(*args, **kwargs)

    clf = KerasClassifier(build_fn=build_fn, hidden_layer_sizes=(100,))
    clf.fit(X, y)


def test_routing_override():
    """Tests that routed parameters override default parameters.
    """
    n, d = 20, 3
    n_classes = 3
    X = np.random.uniform(size=(n, d)).astype(float)
    y = np.random.choice(n_classes, size=n).astype(int)

    def build_fn(
        hidden_layer_sizes,
        meta_params: Dict[str, Any],
        compile_params: Dict[str, Any],
    ) -> Model:
        assert hidden_layer_sizes == (200,)
        assert compile_params["optimizer"] == "adam"
        assert "compile__optimizer" not in compile_params
        assert "compile_optimizer" not in compile_params
        return dynamic_classifier(
            hidden_layer_sizes,
            meta_params=meta_params,
            compile_params=compile_params,
        )

    clf = KerasClassifier(
        build_fn=build_fn,
        hidden_layer_sizes=(100,),
        build__hidden_layer_sizes=(200,),  # override build params
        compile__optimizer="adam",  # overwrites `optimizer` param
    )
    clf.fit(X, y)


@pytest.mark.parametrize(
    "wrapper_class,build_fn",
    [
        (KerasClassifier, dynamic_classifier),
        (KerasRegressor, dynamic_regressor),
    ],
)
def test_no_extra_meta_params(wrapper_class, build_fn):
    """Check that wrappers do not create any unexpected meta parameters.
    """
    n, d = 20, 3
    n_classes = 3
    X = np.random.uniform(size=(n, d)).astype(float)
    y = np.random.choice(n_classes, size=n).astype(int)

    clf = wrapper_class(build_fn=build_fn, hidden_layer_sizes=(100,))
    clf.fit(X, y)
    assert set(clf.get_meta_params().keys()) == wrapper_class._meta_params


@pytest.mark.parametrize("dest", ["fit", "compile", "predict"])
def test_routing_sets(dest):
    accepted_params = set(
        inspect.signature(getattr(Model, dest)).parameters.keys()
    ) - {"self", "kwargs"}
    known_params = getattr(BaseWrapper, f"_{dest}_params")
    if LooseVersion(tf.__version__) <= "2.2.0":
        # this parameter is a kwarg in TF 2.2.0
        # it will still work in practice, but breaks this test
        known_params = known_params - {"run_eagerly"}
    assert known_params.issubset(accepted_params)
