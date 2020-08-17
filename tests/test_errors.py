"""Misc. tests that only exist to verify errors are raised."""

import numpy as np
import pytest

from sklearn.exceptions import NotFittedError

from scikeras.wrappers import KerasClassifier
from scikeras.wrappers import KerasRegressor

from .mlp_models import dynamic_classifier
from .mlp_models import dynamic_regressor


def test_validate_data():
    """Tests the BaseWrapper._validate_data method.
    """

    estimator = KerasRegressor(
        build_fn=dynamic_regressor, loss=KerasRegressor.r_squared,
    )
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])

    with pytest.raises(RuntimeError, match="Is this estimator fitted?"):
        # First call requires reset=True
        estimator._validate_data(X=X, y=y, reset=False)

    estimator._validate_data(X=X, y=y, reset=True)  # no error

    with pytest.raises(ValueError, match=r"but this \w+ is expecting "):
        # Calling with a different shape for X raises an error
        estimator._validate_data(X=X[:, :1], y=y, reset=False)


def test_not_fitted_error():
    """Tests error when trying to use predict before fit.
    """
    estimator = KerasClassifier(
        build_fn=dynamic_classifier, loss=KerasRegressor.r_squared,
    )
    X = np.random.rand(10, 20)
    with pytest.raises(NotFittedError):
        # This is in BaseWrapper so it covers
        # KerasRegressor as well
        estimator.predict(X)
    with pytest.raises(NotFittedError):
        estimator.predict_proba(X)


class TestInvalidBuildFn:
    """Tests various error cases for BuildFn.
    """

    def test_invalid_build_fn(self):
        clf = KerasClassifier(build_fn="invalid")
        with pytest.raises(TypeError, match="build_fn"):
            clf.fit(np.array([[0]]), np.array([0]))

    def test_no_build_fn(self):
        class NoBuildFn(KerasClassifier):
            pass

        clf = NoBuildFn()

        with pytest.raises(
            ValueError, match="must implement `_keras_build_fn`"
        ):
            clf.fit(np.array([[0]]), np.array([0]))

    def test_call_and_build_fn_function(self):
        class Clf(KerasClassifier):
            def _keras_build_fn(self, hidden_layer_sizes=(100,)):
                return dynamic_classifier(
                    hidden_layer_sizes=hidden_layer_sizes
                )

        def dummy_func():
            return None

        clf = Clf(build_fn=dummy_func,)

        with pytest.raises(
            ValueError, match="cannot implement `_keras_build_fn`"
        ):
            clf.fit(np.array([[0]]), np.array([0]))

    def test_call_and_invalid_build_fn_class(self):
        class Clf(KerasClassifier):
            def _keras_build_fn(self, hidden_layer_sizes):
                return dynamic_classifier(
                    hidden_layer_sizes=hidden_layer_sizes
                )

        class DummyBuildClass:
            def __call__(self, hidden_layer_sizes):
                return dynamic_classifier(
                    hidden_layer_sizes=hidden_layer_sizes
                )

        clf = Clf(build_fn=DummyBuildClass(),)

        with pytest.raises(
            ValueError, match="cannot implement `_keras_build_fn`"
        ):
            clf.fit(np.array([[0]]), np.array([0]))
