import pickle

from itertools import chain

import numpy as np
import pytest

from sklearn.utils.estimator_checks import _mark_xfail_checks
from sklearn.utils.estimator_checks import _set_check_estimator_ids
from sklearn.utils.estimator_checks import check_estimator


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


def parametrize_with_checks(estimators, ids=None):
    """Wraps scikit-learns fixture to allow setting IDs.

    This is done for 2 reasons:
    1)  The fixture generator calls clone() on the estimators,
        which makes a copy of build_fn and so a different memory
        address gets printed each time, which makes pytest
        xdist fail.
    2)  Avoid cluttering the test log with all of the parameters of
        KerasClassifier and KerasRegressor (we are only testing 1
        config for each, it is easy to track).
    """
    checks_generator = chain.from_iterable(
        check_estimator(estimator, generate_only=True)
        for estimator in estimators
    )

    checks_with_marks = (
        _mark_xfail_checks(estimator, check, pytest)
        for estimator, check in checks_generator
    )

    if ids is None:
        ids = _set_check_estimator_ids
    else:
        estimator_ids = {
            estimator: _id for _id, estimator in zip(ids, estimators)
        }

        def get_id(obj):
            if callable(obj):
                return _set_check_estimator_ids(obj)
            if hasattr(obj, "get_params"):
                # An estimator
                # return custom id
                return estimator_ids[obj]

        ids = get_id

    return pytest.mark.parametrize(
        "estimator, check", checks_with_marks, ids=ids
    )


# from distutils.version import LooseVersion


# from sklearn import __version__ as sklearn_version
# from scikeras.wrappers import KerasClassifier
# from scikeras.wrappers import KerasRegressor
# import pytest

# from tests.utils import parametrize_with_checks
# from tests.mlp_models import dynamic_classifier, dynamic_regressor


# parametrize_with_checks(
#     estimators=[KerasClassifier(
#         build_fn=dynamic_classifier,
#         # Set batch size to a large number (larger than X.shape[0] is the goal)
#         # if batch_size < X.shape[0], results will very
#         # slightly if X is shuffled.
#         # This is only required for this tests and is not really
#         # applicable to real world datasets
#         batch_size=1000,
#         optimizer="adam",
#     )],
#     ids=["KerasClassifier"],
# )
