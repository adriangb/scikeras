from typing import Any, Dict

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from scikeras.wrappers import KerasRegressor


def dynamic_classifier(
    hidden_layer_sizes, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Creates a basic MLP classifier dynamically choosing binary/multiclass
    classification loss and ouput activations.
    """
    # get parameters
    n_features_in_ = meta["n_features_in_"]
    target_type_ = meta["target_type_"]
    n_classes_ = meta["n_classes_"]
    model_n_outputs_ = meta["model_n_outputs_"]
    metrics = compile_kwargs["metrics"]
    loss = compile_kwargs["loss"]
    optimizer = compile_kwargs["optimizer"]

    inp = Input(shape=(n_features_in_,))

    hidden = inp
    for layer_size in hidden_layer_sizes:
        hidden = Dense(layer_size, activation="relu")(hidden)

    if target_type_ == "binary":
        loss = loss or "binary_crossentropy"
        out = [Dense(1, activation="sigmoid")(hidden)]
    elif target_type_ == "multilabel-indicator":
        loss = loss or "binary_crossentropy"
        out = [
            Dense(1, activation="sigmoid")(hidden)
            for _ in range(model_n_outputs_)
        ]
    elif target_type_ == "multiclass-multioutput":
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
    hidden_layer_sizes, meta: Dict[str, Any], compile_kwargs: Dict[str, Any],
) -> Model:
    """Creates a basic MLP regressor dynamically.
    """
    # get parameters
    n_features_in_ = meta["n_features_in_"]
    n_outputs_ = meta["n_outputs_"]
    metrics = compile_kwargs["metrics"]
    loss = compile_kwargs["loss"]
    optimizer = compile_kwargs["optimizer"]

    if loss is None:
        # Default Model loss, not appropriate for a classifier
        loss = KerasRegressor.r_squared

    inp = Input(shape=(n_features_in_,))

    hidden = inp
    for layer_size in hidden_layer_sizes:
        hidden = Dense(layer_size, activation="relu")(hidden)

    out = Dense(n_outputs_)(hidden)

    model = Model(inp, out)

    model.compile(
        optimizer=optimizer,
        loss=loss,  # KerasRegressor.r_squared
        metrics=metrics,
    )
    return model
