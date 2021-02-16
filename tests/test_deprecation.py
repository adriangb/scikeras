"""Tests for features scheduled for deprecation.
"""
from unittest import mock

import numpy as np
import pytest

from scikeras.wrappers import KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor


def test_build_fn_deprecation():
    """An appropriate warning is raised when using the `build_fn`
    parameter instead of `model`.
    """
    clf = KerasClassifier(build_fn=dynamic_classifier, model__hidden_layer_sizes=(100,))
    with pytest.warns(UserWarning, match="``build_fn`` will be renamed to ``model``"):
        clf.fit([[0], [1]], [0, 1])


@pytest.mark.parametrize(
    "wrapper,builder",
    [
        (KerasClassifier, dynamic_classifier),
        (KerasRegressor, dynamic_regressor),
    ],
)
def test_kwarg_deprecation(wrapper, builder):
    """Test that SciKeras supports the **kwarg interface in fit and predict
    but warns the user about deprecation of this interface.
    """
    original_batch_size = 128
    kwarg_batch_size = 90
    kwarg_epochs = 2  # epochs is a special case for fit since SciKeras also uses it internally
    extra_kwargs = {"workers": 1}  # chosen because it is not a SciKeras hardcoded param
    est = wrapper(
        model=builder,
        model__hidden_layer_sizes=(100,),
        warm_start=True,  # for mocking to work properly
        batch_size=original_batch_size,  # test that this is overridden by kwargs
        fit__batch_size=original_batch_size,  # test that this is overridden by kwargs
        predict__batch_size=original_batch_size,  # test that this is overridden by kwargs
    )
    X, y = np.random.random((100, 10)), np.random.randint(low=0, high=3, size=(100,))
    est.initialize(X, y)
    match_txt = r"``\*\*kwargs`` has been deprecated in SciKeras"
    # check fit
    with pytest.warns(UserWarning, match=match_txt):
        with mock.patch.object(est.model_, "fit", side_effect=est.model_.fit) as mock_fit:
            est.fit(X, y, batch_size=kwarg_batch_size, epochs=kwarg_epochs, **extra_kwargs)
            call_args = mock_fit.call_args_list
            assert len(call_args) == 1
            call_kwargs = call_args[0][1]
            assert "batch_size" in call_kwargs
            assert call_kwargs["batch_size"] == kwarg_batch_size
            assert call_kwargs["epochs"] == kwarg_epochs
            assert len(est.history_["loss"]) == kwarg_epochs
    # check predict
    with pytest.warns(UserWarning, match=match_txt):
        with mock.patch.object(est.model_, "predict", side_effect=est.model_.predict) as mock_predict:
            est.predict(X, batch_size=kwarg_batch_size, **extra_kwargs)
            call_args = mock_predict.call_args_list
            assert len(call_args) == 1
            call_kwargs = call_args[0][1]
            assert "batch_size" in call_kwargs
            assert call_kwargs["batch_size"] == kwarg_batch_size
            if isinstance(est, KerasClassifier):
                est.predict_proba(X, batch_size=kwarg_batch_size, **extra_kwargs)
                call_args = mock_predict.call_args_list
                assert len(call_args) == 2
                call_kwargs = call_args[1][1]
                assert "batch_size" in call_kwargs
                assert call_kwargs["batch_size"] == kwarg_batch_size
    # check that params were restored and extra_kwargs were not stored
    for param_name in ("batch_size", "fit__batch_size", "predict__batch_size"):
        assert getattr(est, param_name) == original_batch_size
    for k in extra_kwargs.keys():
        assert not hasattr(est, k) or hasattr(est, "fit__" + k) or hasattr(est, "predict__" + k)
