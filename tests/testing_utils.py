import pickle

from functools import partial

import numpy as np
import pytest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.estimator_checks import (
    parametrize_with_checks as _parametrize_with_checks,
)


def basic_checks(estimator, loader):
    """Run basic checks (fit, score, pickle) on estimator."""
    data = loader()
    # limit to 100 data points to speed up testing
    X, y = data.data[:100], data.target[:100]
    estimator.fit(X, y)
    estimator.predict(X)
    score = estimator.score(X, y)
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    deserialized_estimator.predict(X)
    score_new = deserialized_estimator.score(X, y)
    np.testing.assert_almost_equal(score, score_new)


def _get_check_estimator_ids(obj, estimator_ids=None):
    """Backport from Scikit-Learn = 0.23.0, not available in 0.22.0"""
    if callable(obj):
        if not isinstance(obj, partial):
            return obj.__name__

        if not obj.keywords:
            return obj.func.__name__

        kwstring = ",".join(
            ["{}={}".format(k, v) for k, v in obj.keywords.items()]
        )
        return "{}({})".format(obj.func.__name__, kwstring)
    if hasattr(obj, "get_params"):
        # An estimator
        if estimator_ids is None:
            # Get id/string from parameters
            with config_context(print_changed_only=True):
                return re.sub(r"\s", "", str(obj))
        else:
            # User user supplied ID
            return estimator_ids[obj]


def parametrize_with_checks(estimators, ids=None):
    """Wraps scikit-learn's fixture to allow setting IDs.

    This is done for 2 reasons:
    1)  The fixture generator calls clone() on the estimators,
        which makes a copy of build_fn and so a different memory
        address gets printed each time, which makes pytest
        xdist fail.
    2)  Avoid cluttering the test log with all of the parameters of
        KerasClassifier and KerasRegressor (we are only testing 1
        config for each, it is easy to track).
    """
    checks_generator = _parametrize_with_checks(estimators).args[1]

    if ids is not None:
        estimator_ids = {
            estimator: _id for _id, estimator in zip(ids, estimators)
        }
        ids = partial(_get_check_estimator_ids, estimator_ids=estimator_ids)

    return pytest.mark.parametrize(
        "estimator, check", checks_generator, ids=ids
    )
