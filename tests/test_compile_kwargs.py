import numpy as np
import pytest

from sklearn.datasets import make_classification
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from scikeras.wrappers import KerasClassifier, metrics_module


def get_model(num_hidden=10, meta=None, compile_kwargs=None):
    inp = Input(shape=(meta["n_features_in_"],))
    hidden = Dense(num_hidden, activation="relu")(inp)
    out = [Dense(1, activation="sigmoid", name="my_output")(hidden)]
    model = Model(inp, out)
    model.compile(**compile_kwargs)
    return model


@pytest.mark.parametrize("optimizer", (optimizers.SGD, "SGD"))
def test_optimizer(optimizer):
    """Tests that if build_fn returns an un-compiled model,
    the __init__ parameters will be used to compile it.
    """
    est = KerasClassifier(
        model=get_model,
        model__num_hidden=20,
        optimizer=optimizer,
        optimizer__learning_rate=0.15,
        optimizer__momentum=0.5,
        loss="binary_crossentropy",
    )
    X, y = make_classification()
    est.fit(X, y)
    np.testing.assert_almost_equal(
        est.model_.optimizer.momentum.value().numpy(), 0.5, decimal=2
    )
    np.testing.assert_almost_equal(
        est.model_.optimizer.learning_rate.value().numpy(), 0.15, decimal=2
    )


@pytest.mark.parametrize(
    "loss",
    (losses.BinaryCrossentropy, "BinaryCrossentropy", "binary_crossentropy"),
)
def test_loss(loss):
    """Tests that if build_fn returns an un-compiled model,
    the __init__ parameters will be used to compile it.
    """
    est = KerasClassifier(
        model=get_model,
        model__num_hidden=20,
        optimizer="sgd",
        loss=loss,
        loss__name="custom_name",
    )
    X, y = make_classification()
    est.fit(X, y)
    if not callable(est.model_.loss):
        assert est.model_.loss.name == "custom_name"


def test_loss_iterable():
    """Tests that if build_fn returns an un-compiled model,
    the __init__ parameters will be used to compile it.
    """
    # Test iterable
    est = KerasClassifier(
        model=get_model,
        model__num_hidden=20,
        optimizer="sgd",
        loss=[("custom_name", losses.BinaryCrossentropy, "my_output"),],
    )
    X, y = make_classification()
    est.fit(X, y)
    assert est.model_.loss["my_output"].name == "custom_name"

    # Test iterable with global routed param
    est = KerasClassifier(
        model=get_model,
        model__num_hidden=20,
        optimizer=optimizers.SGD(),
        loss=[("custom_name", losses.BinaryCrossentropy, "my_output"),],
        loss__from_logits=True,  # default is False
    )
    X, y = make_classification()
    est.fit(X, y)
    assert est.model_.loss["my_output"].from_logits == True

    # Test iterable with specific routed param
    est = KerasClassifier(
        model=get_model,
        model__num_hidden=20,
        optimizer=optimizers.SGD(),
        loss=[("custom_name", losses.BinaryCrossentropy, "my_output"),],
        loss__from_logits=False,
        loss__custom_name__from_logits=True,  # should override above
    )
    X, y = make_classification()
    est.fit(X, y)
    assert est.model_.loss["my_output"].from_logits == True


@pytest.mark.xfail  # TODO: fix metrics
@pytest.mark.parametrize(
    "metrics", ("acc", "Accuracy", metrics_module.Accuracy)
)
def test_metrics(metrics):
    """Tests that if build_fn returns an un-compiled model,
    the __init__ parameters will be used to compile it.
    """
    est = KerasClassifier(
        model=get_model,
        model__num_hidden=20,
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=[metrics,],
        metrics__name="custom_name",
    )
    X, y = make_classification()
    est.fit(X, y)
    # metric 0 is always the loss function
    if not callable(est.model_.metrics[1]):
        assert est.model_.metrics[1].name == "custom_name"
        assert est.model_.metrics_names[1] == "custom_name"
