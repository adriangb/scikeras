"""Tests for features scheduled for deprecation.
"""
import pytest

from scikeras.wrappers import KerasClassifier

from .mlp_models import dynamic_classifier


def test_build_fn_deprecation():
    """An appropriate warning is raised when using the `build_fn`
    parameter instead of `model`.
    """
    clf = KerasClassifier(build_fn=dynamic_classifier, model__hidden_layer_sizes=(100,))
    with pytest.warns(UserWarning, match="``build_fn`` will be renamed to ``model``"):
        clf.fit([[0], [1]], [0, 1])
