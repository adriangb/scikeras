import os

from unittest import mock

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from scikeras.wrappers import KerasClassifier, KerasRegressor

from .mlp_models import dynamic_classifier, dynamic_regressor


class TestRandomState:
    @pytest.mark.parametrize(
        "random_state", [0, 123, np.random.RandomState(0)],
    )
    @pytest.mark.parametrize(
        "estimator",
        [
            KerasRegressor(model=dynamic_regressor, model__hidden_layer_sizes=(100,),),
            KerasClassifier(model=dynamic_classifier, model__hidden_layer_sizes=(100,)),
        ],
    )
    def test_random_states(self, random_state, estimator):
        """Tests that the random_state parameter correctly
        engages deterministric training and prediction.
        """
        X, y = make_classification()

        # With seed
        estimator.set_params(random_state=random_state)
        estimator.fit(X, y)
        y1 = estimator.predict(X)
        estimator.fit(X, y)
        y2 = estimator.predict(X)
        assert np.allclose(y1, y2)

        if isinstance(estimator, KerasRegressor):
            # Without seed, regressors should NOT
            # give the same results
            # Classifiers _may_ give the same classes
            estimator.set_params(random_state=None)
            estimator.fit(X, y)
            y1 = estimator.predict(X)
            estimator.fit(X, y)
            y2 = estimator.predict(X)
            assert not np.allclose(y1, y2)

    @pytest.mark.parametrize(
        "estimator",
        [
            KerasRegressor(model=dynamic_regressor, model__hidden_layer_sizes=(100,),),
            KerasClassifier(model=dynamic_classifier, model__hidden_layer_sizes=(100,)),
        ],
    )
    @pytest.mark.parametrize("pyhash", [None, "0", "1"])
    @pytest.mark.parametrize("gpu", [None, "0", "1"])
    def test_random_states_env_vars(self, estimator, pyhash, gpu):
        """Tests that the random state context management correctly
        handles TF related env variables.
        """
        X, y = make_classification()

        if "random_state" in estimator.get_params():
            estimator.set_params(random_state=None)
        estimator1 = clone(estimator)
        estimator2 = clone(estimator)
        if "random_state" in estimator1.get_params():
            estimator1.set_params(random_state=0)
        if "random_state" in estimator2.get_params():
            estimator2.set_params(random_state=0)
        if gpu is not None:
            os.environ["TF_DETERMINISTIC_OPS"] = gpu
        else:
            if os.environ.get("TF_DETERMINISTIC_OPS"):
                os.environ.pop("TF_DETERMINISTIC_OPS")
        if pyhash is not None:
            os.environ["PYTHONHASHSEED"] = pyhash
        else:
            if os.environ.get("PYTHONHASHSEED"):
                os.environ.pop("PYTHONHASHSEED")
        estimator1.fit(X, y)
        estimator2.fit(X, y)
        if gpu is not None:
            assert os.environ["TF_DETERMINISTIC_OPS"] == gpu
        else:
            assert "TF_DETERMINISTIC_OPS" not in os.environ
        if pyhash is not None:
            assert os.environ["PYTHONHASHSEED"] == pyhash
        else:
            assert "PYTHONHASHSEED" not in os.environ
        y1 = estimator1.predict(X)
        y2 = estimator2.predict(X)
        assert np.allclose(y1, y2)
        if gpu is not None:
            assert os.environ["TF_DETERMINISTIC_OPS"] == gpu
        else:
            assert "TF_DETERMINISTIC_OPS" not in os.environ
        if pyhash is not None:
            assert os.environ["PYTHONHASHSEED"] == pyhash
        else:
            assert "PYTHONHASHSEED" not in os.environ


def test_sample_weights_fit():
    """Checks that the `sample_weight` parameter when passed to `fit`
    has the intended effect.
    """
    # build estimator
    estimator = KerasClassifier(
        model=dynamic_classifier,
        model__hidden_layer_sizes=(100,),
        epochs=10,
        random_state=0,
    )
    estimator1 = clone(estimator)
    estimator2 = clone(estimator)

    # we create 20 points
    X = np.array([1] * 10000).reshape(-1, 1)
    y = [1] * 5000 + [-1] * 5000

    # heavily weight towards y=1 points
    sw_first_class = [0.8] * 5000 + [0.2] * 5000
    # train estimator 1 with weights
    with pytest.warns(UserWarning, match="Setting the random state"):
        estimator1.fit(X, y, sample_weight=sw_first_class)
    # train estimator 2 without weights
    with pytest.warns(UserWarning, match="Setting the random state"):
        estimator2.fit(X, y)
    # estimator1 should tilt towards y=1
    # estimator2 should predict about equally
    average_diff_pred_prob_1 = np.average(np.diff(estimator1.predict_proba(X), axis=1))
    average_diff_pred_prob_2 = np.average(np.diff(estimator2.predict_proba(X), axis=1))
    assert average_diff_pred_prob_2 < average_diff_pred_prob_1

    # equal weighting
    sw_equal = [0.5] * 5000 + [0.5] * 5000
    # train estimator 1 with weights
    estimator1.fit(X, y, sample_weight=sw_equal)
    # train estimator 2 without weights
    estimator2.fit(X, y)
    # both estimators should have about the same predictions
    np.testing.assert_allclose(
        actual=estimator1.predict_proba(X), desired=estimator2.predict_proba(X)
    )


def test_sample_weights_score():
    """Checks that the `sample_weight` parameter when passed to
    `score` has the intended effect.
    """
    # build estimator
    estimator = KerasRegressor(
        model=dynamic_regressor,
        model__hidden_layer_sizes=(100,),
        epochs=10,
        random_state=0,
    )
    estimator1 = clone(estimator)
    estimator2 = clone(estimator)

    # we create 20 points
    X = np.array([1] * 10000).reshape(-1, 1)
    y = [1] * 5000 + [-1] * 5000

    # train
    estimator1.fit(X, y)
    estimator2.fit(X, y)

    # heavily weight towards y=1 points
    bad_sw = [0.999] * 5000 + [0.001] * 5000

    # score with weights, estimator2 should
    # score higher since the weights "unbalance"
    score1 = estimator1.score(X, y, sample_weight=bad_sw)
    score2 = estimator2.score(X, y)
    assert score2 > score1


def test_build_fn_default_params():
    """Tests that default arguments arguments of
    `build_fn` are registered as hyperparameters.
    """
    est = KerasClassifier(model=dynamic_classifier, model__hidden_layer_sizes=(100,))
    params = est.get_params()
    # (100, ) is the default for dynamic_classifier
    assert params["model__hidden_layer_sizes"] == (100,)

    est = KerasClassifier(model=dynamic_classifier, model__hidden_layer_sizes=(200,))
    params = est.get_params()
    assert params["model__hidden_layer_sizes"] == (200,)


class TestMetricsParam:
    @pytest.mark.parametrize("metric", ("accuracy", "sparse_categorical_accuracy"))
    def test_metrics(self, metric):
        """Test the metrics param.

        Specifically test ``accuracy``, which Keras automatically
        matches to the loss function and hence should be passed through
        as a string and not as a retrieved function.
        """
        est = KerasClassifier(
            model=dynamic_classifier, model__hidden_layer_sizes=(100,), metrics=[metric]
        )
        X, y = make_classification()
        est.fit(X, y)
        assert len(est.history_[metric]) == 1


def test_class_weight_param():
    """Backport of sklearn.utils.estimator_checks.check_class_weight_classifiers
    for sklearn <= 0.23.0.

    Tests that fit and partial_fit correctly handle the class_weight parameter.
    """
    clf = KerasClassifier(
        model=dynamic_classifier, model__hidden_layer_sizes=(100,), random_state=0,
    )
    problems = (2, 3)
    for n_centers in problems:
        # create a very noisy dataset
        X, y = make_blobs(centers=n_centers, random_state=0, cluster_std=20)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        n_centers = len(np.unique(y_train))

        if n_centers == 2:
            class_weight = {0: 1000, 1: 0.0001}
            fit_epochs = 4
            partial_fit_epochs = 3
        else:
            class_weight = {0: 1000, 1: 0.0001, 2: 0.0001}
            fit_epochs = 8
            partial_fit_epochs = 6

        clf.set_params(class_weight=class_weight)

        # run fit epochs followed by several partial_fit iterations
        # these numbers are purely empirical, just like they are in the
        # original sklearn test
        clf.set_params(fit__epochs=fit_epochs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        assert np.mean(y_pred == 0) > 0.8
        for _ in range(partial_fit_epochs):
            clf.partial_fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        assert np.mean(y_pred == 0) > 0.95


@pytest.mark.parametrize(
    "wrapper,builder",
    [(KerasClassifier, dynamic_classifier), (KerasRegressor, dynamic_regressor)],
)
def test_kwargs(wrapper, builder):
    """Test that SciKeras supports the **kwarg interface in fit and predict."""
    original_batch_size = 128
    kwarg_batch_size = 90
    kwarg_epochs = (
        2  # epochs is a special case for fit since SciKeras also uses it internally
    )
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
    # check fit
    match = "estimator parameters as keyword arguments"
    with mock.patch.object(est.model_, "fit", side_effect=est.model_.fit) as mock_fit:
        est.fit(
            X, y, batch_size=kwarg_batch_size, epochs=kwarg_epochs, **extra_kwargs
        )
        call_args = mock_fit.call_args_list
        assert len(call_args) == 1
        call_kwargs = call_args[0][1]
        assert "batch_size" in call_kwargs
        assert call_kwargs["batch_size"] == kwarg_batch_size
        assert call_kwargs["epochs"] == kwarg_epochs
        assert len(est.history_["loss"]) == kwarg_epochs
    # check predict
    with mock.patch.object(
        est.model_, "predict", side_effect=est.model_.predict
    ) as mock_predict:
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
        assert (
            not hasattr(est, k)
            or hasattr(est, "fit__" + k)
            or hasattr(est, "predict__" + k)
        )


@pytest.mark.parametrize("length", (10, 100))
@pytest.mark.parametrize("prefix", ("", "fit__"))
@pytest.mark.parametrize("base", ("validation_batch_size", "batch_size"))
def test_batch_size_all_fit(length, prefix, base):

    kw = prefix + base

    y = np.random.random((length,))
    X = y.reshape((-1, 1))
    est = KerasRegressor(dynamic_regressor, hidden_layer_sizes=[], **{kw: -1})

    est.initialize(X, y)

    fit_orig = est.model_.fit

    def check_batch_size(**kwargs):
        assert kwargs[base] == X.shape[0]
        return fit_orig(**kwargs)

    with mock.patch.object(est.model_, "fit", new=check_batch_size):
        est.fit(X, y)


@pytest.mark.parametrize("length", (10, 100))
@pytest.mark.parametrize("prefix", ("", "predict__"))
@pytest.mark.parametrize("base", ("batch_size",))
def test_batch_size_all_predict(length, prefix, base):

    kw = prefix + base

    y = np.random.random((length,))
    X = y.reshape((-1, 1))
    est = KerasRegressor(dynamic_regressor, hidden_layer_sizes=[], **{kw: -1})

    est.fit(X, y)

    pred_orig = est.model_.predict

    def check_batch_size(**kwargs):
        assert kwargs[base] == X.shape[0]
        return pred_orig(**kwargs)

    with mock.patch.object(est.model_, "predict", new=check_batch_size):
        est.predict(X)


@pytest.mark.parametrize("length", (10, 100))
@pytest.mark.parametrize("prefix", ("", "fit__"))
@pytest.mark.parametrize("base", ("validation_batch_size", "batch_size"))
def test_batch_size_all_fit(length, prefix, base):

    kw = prefix + base

    y = np.random.random((length,))
    X = y.reshape((-1, 1))
    est = KerasRegressor(dynamic_regressor, hidden_layer_sizes=[], **{kw: -1})

    est.initialize(X, y)

    fit_orig = est.model_.fit

    def check_batch_size(**kwargs):
        assert kwargs[base] == X.shape[0]
        return fit_orig(**kwargs)

    with mock.patch.object(est.model_, "fit", new=check_batch_size):
        est.fit(X, y)


@pytest.mark.parametrize("prefix", ("", "fit__"))
@pytest.mark.parametrize("base", ("validation_batch_size", "batch_size"))
def test_batch_size_all_fit_non_array(prefix, base):

    kw = prefix + base

    class CustomReg(KerasRegressor):
        @property
        def feature_encoder(self):
            return FunctionTransformer(lambda x: [x])

    y = np.random.random((100,))
    X = y.reshape((-1, 1))
    est = CustomReg(dynamic_regressor, hidden_layer_sizes=[], **{kw: -1})

    with pytest.raises(ValueError, match="requires that `X` implement `shape`"):
        est.fit(X, y)


def test_batch_size_all_predict_non_array():
    class CustomReg(KerasRegressor):
        @property
        def feature_encoder(self):
            return FunctionTransformer(lambda x: [x])

    y = np.random.random((100,))
    X = y.reshape((-1, 1))
    est = CustomReg(dynamic_regressor, hidden_layer_sizes=[], predict__batch_size=-1)

    est.fit(X, y)

    with pytest.raises(ValueError, match="requires that `X` implement `shape`"):
        est.predict(X)
