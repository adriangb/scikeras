import inspect

import numpy as np
import pytest

from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers as layers_mod

from scikeras.wrappers import BaseWrapper, KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor


keras_classifier_base_meta_set = {
    "X_dtype_",
    "y_dtype_",
    "classes_",
    "target_type_",
    "n_classes_",
    "n_features_in_",
    "X_shape_",
    "n_outputs_expected_",
    "y_ndim_",
    "n_outputs_",
    "feature_encoder_",
    "target_encoder_",
}

keras_regressor_base_meta_set = {
    "X_shape_",
    "n_outputs_expected_",
    "X_dtype_",
    "n_outputs_",
    "y_dtype_",
    "y_ndim_",
    "n_features_in_",
    "target_type_",
    "feature_encoder_",
    "target_encoder_",
}


@pytest.mark.parametrize(
    "wrapper, builder, expected_meta",
    [
        (KerasClassifier, dynamic_classifier, keras_classifier_base_meta_set,),
        (KerasRegressor, dynamic_regressor, keras_regressor_base_meta_set,),
    ],
)
def test_routing_basic(wrapper, builder, expected_meta):
    n, d = 20, 3
    n_classes = 3
    X = np.random.uniform(size=(n, d)).astype(float)
    y = np.random.choice(n_classes, size=n).astype(int)

    foo_val = object()

    def build_fn(hidden_layer_sizes, foo, compile_kwargs, params, meta):
        assert set(params.keys()) == set(est.get_params().keys())
        assert set(meta.keys()) == expected_meta
        assert set(compile_kwargs.keys()).issubset(wrapper._compile_kwargs)
        assert foo is foo_val
        return builder(
            hidden_layer_sizes=hidden_layer_sizes,
            compile_kwargs=compile_kwargs,
            meta=meta,
        )

    est = wrapper(model=build_fn, model__hidden_layer_sizes=(100,), model__foo=foo_val)
    est.fit(X, y)

    est = wrapper(model=build_fn, model__hidden_layer_sizes=(100,), foo=foo_val)
    est.fit(X, y)


@pytest.mark.parametrize(
    "wrapper, builder, expected_meta",
    [
        (KerasClassifier, dynamic_classifier, keras_classifier_base_meta_set,),
        (KerasRegressor, dynamic_regressor, keras_regressor_base_meta_set,),
    ],
)
def test_routing_kwargs(wrapper, builder, expected_meta):
    """Tests that special parameters are passed if
    build_fn accepts kwargs.
    """
    n, d = 20, 3
    n_classes = 3
    X = np.random.uniform(size=(n, d)).astype(float)
    y = np.random.choice(n_classes, size=n).astype(int)

    def build_fn(*args, **kwargs):
        assert len(args) == 0, "No *args should be passed to `build_fn`"
        assert tuple(kwargs.keys()) == (
            "hidden_layer_sizes",
            "meta",
            "compile_kwargs",
            "params",
        ), "The number and order of **kwargs passed to `build_fn` should be fixed"
        assert set(kwargs["meta"].keys()) == expected_meta
        assert set(kwargs["compile_kwargs"].keys()).issubset(wrapper._compile_kwargs)
        kwargs.pop("params")  # dynamic_classifier/regressor don't accept it
        return builder(*args, **kwargs)

    est = wrapper(model=build_fn, model__hidden_layer_sizes=(100,))
    est.fit(X, y)


@pytest.mark.parametrize("dest", ["fit", "compile", "predict"])
def test_routing_sets(dest):
    accepted_params = set(inspect.signature(getattr(Model, dest)).parameters.keys()) - {
        "self",
        "kwargs",
    }
    known_params = getattr(BaseWrapper, f"_{dest}_kwargs")
    assert known_params.issubset(accepted_params)


def test_routed_unrouted_equivalence():
    """Test that `hidden_layer_sizes` and `model__hidden_layer_sizes`
    both work.
    """
    n, d = 20, 3
    n_classes = 3
    X = np.random.uniform(size=(n, d)).astype(float)
    y = np.random.choice(n_classes, size=n).astype(int)

    clf = KerasClassifier(model=dynamic_classifier, model__hidden_layer_sizes=(100,))
    clf.fit(X, y)

    clf = KerasClassifier(model=dynamic_classifier, hidden_layer_sizes=(100,))
    clf.fit(X, y)


def test_parameter_precedence():
    """Routed parameters should override non-routed parameters, and fit keyword arguments should override routed"""

    class TestModel(Sequential):
        def fit(self, *args, **kwargs):
            assert kwargs["class_weight"] == {0: 0.5, 1: 0.5}
            assert kwargs.pop("custom") == "fit_keyword"
            return super().fit(*args, **kwargs)

    def get_model() -> TestModel:
        return TestModel(
            [layers_mod.InputLayer((1,)), layers_mod.Dense(1, activation="sigmoid")]
        )

    X, y = [[1], [2]], [0, 1]

    clf = KerasClassifier(
        get_model,
        loss="binary_crossentropy",
        fit__class_weight={
            0: 0.5,
            1: 0.5,
        },  # test w/ a built in parameter to make sure we can override them
        fit__custom="constructor_routed",
    )

    clf.fit(X, y, custom="fit_keyword")


def test_exclude_parameters_with_further_routing():
    """SciKeras should only route parameters to final destinations that do not contain further routing
    For example, optimizer__xyz__abc should _not_ be passed to the Optimizer as Optimizer(xyz__abc=xyz__abc).
    """

    def get_model() -> Sequential:
        return Sequential(
            [layers_mod.InputLayer((1,)), layers_mod.Dense(1, activation="sigmoid")]
        )

    X, y = [[1], [2]], [0, 1]

    clf = KerasClassifier(
        get_model,
        loss="binary_crossentropy",
        optimizer__this_should_not_pass__abc="error!",
    )

    clf.fit(X, y)
