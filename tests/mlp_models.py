from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

from scikeras.wrappers import KerasRegressor


def dynamic_classifier(
    n_features_in_,
    cls_type_,
    n_classes_,
    metrics=None,
    keras_expected_n_ouputs_=1,
    loss=None,
    optimizer="sgd",
    hidden_layer_sizes=(100,),
):
    """Creates a basic MLP classifier dynamically choosing binary/multiclass
    classification loss and ouput activations.
    """

    inp = Input(shape=(n_features_in_,))

    hidden = inp
    for layer_size in hidden_layer_sizes:
        hidden = Dense(layer_size, activation="relu")(hidden)

    if cls_type_ == "binary":
        loss = loss or "binary_crossentropy"
        out = [Dense(1, activation="sigmoid")(hidden)]
    elif cls_type_ == "multilabel-indicator":
        loss = loss or "binary_crossentropy"
        out = [
            Dense(1, activation="sigmoid")(hidden)
            for _ in range(keras_expected_n_ouputs_)
        ]
    elif cls_type_ == "multiclass-multioutput":
        loss = loss or "binary_crossentropy"
        out = [Dense(n, activation="softmax")(hidden) for n in n_classes_]
    else:
        # multiclass
        loss = loss or "categorical_crossentropy"
        out = [Dense(n_classes_, activation="softmax")(hidden)]

    model = Model(inp, out)

    model.compile(
        loss=loss, optimizer=optimizer, metrics=metrics,
    )

    return model


def dynamic_regressor(
    n_features_in_,
    n_outputs_,
    loss=KerasRegressor.root_mean_squared_error,
    optimizer="adam",
    metrics=None,
    hidden_layer_sizes=(100,),
):
    """Creates a basic MLP regressor dynamically.
    """
    if loss is None:
        # Default Model loss, not appropriate for a classifier
        loss = KerasRegressor.root_mean_squared_error

    inp = Input(shape=(n_features_in_,))

    hidden = inp
    for layer_size in hidden_layer_sizes:
        hidden = Dense(layer_size, activation="relu")(hidden)

    out = Dense(n_outputs_)(hidden)

    model = Model(inp, out)

    model.compile(
        optimizer=optimizer,
        loss=loss,  # KerasRegressor.root_mean_squared_error
        metrics=metrics,
    )
    return model
