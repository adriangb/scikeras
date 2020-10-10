import pickle

from sklearn.utils.estimator_checks import (
    check_estimators_partial_fit_n_features,  # noqa
)
from sklearn.utils.estimator_checks import check_estimators_pickle
from tensorflow.keras.callbacks import Callback

from scikeras.wrappers import KerasClassifier

from .mlp_models import dynamic_classifier


class SentinalCallback(Callback):
    """
    Callback class that sets an internal value once it's been acessed.
    """

    called = 0

    def on_train_begin(self, logs=None):
        """Increments counter."""
        self.called += 1


def test_callbacks():
    """Test estimators with callbacks.
    """
    estimator = KerasClassifier(
        model=dynamic_classifier,
        callbacks=(SentinalCallback(),),
        optimizer="adam",
        model__hidden_layer_sizes=(100,),
    )
    # Check for picklign and partial fit
    check_estimators_pickle("KerasClassifier", estimator)
    check_estimators_partial_fit_n_features("KerasClassifier", estimator)
    # Check that callback was called
    estimator.fit([[0]], [1])  # quick fit
    assert estimator.callbacks[0].called != SentinalCallback.called
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    assert deserialized_estimator.callbacks[0].called == estimator.callbacks[0].called
    estimator.fit([[0]], [1])  # quick fit
