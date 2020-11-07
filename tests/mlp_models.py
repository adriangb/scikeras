from typing import Any, Dict, Optional

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from scikeras.wrappers import KerasRegressor


def dynamic_classifier(
    hidden_layer_sizes=(10, ), meta: Optional[Dict[str, Any]]=None, compile_kwargs: Optional[Dict[str, Any]]=None,
) -> Model:
    """Creates a basic MLP classifier dynamically choosing binary/multiclass
    classification loss and ouput activations.
    """
    # get parameters
    n_features_in_ = meta["n_features_in_"]
    target_type_ = meta["target_type_"]
    n_classes_ = meta["n_classes_"]
    n_outputs_expected_ = meta["n_outputs_expected_"]

    inp = Input(shape=(n_features_in_,))

    hidden = inp
    for layer_size in hidden_layer_sizes:
        hidden = Dense(layer_size, activation="relu")(hidden)

    if target_type_ == "binary":
        compile_kwargs["loss"] = compile_kwargs["loss"] or "binary_crossentropy"
        out = [Dense(1, activation="sigmoid")(hidden)]
    elif target_type_ == "multilabel-indicator":
        compile_kwargs["loss"] = compile_kwargs["loss"] or "binary_crossentropy"
        if isinstance(n_classes_, list):
            out = [
                Dense(1, activation="sigmoid")(hidden)
                for _ in range(n_outputs_expected_)
            ]
        else:
            out = Dense(n_classes_, activation="softmax")(hidden)
    elif target_type_ == "multiclass-multioutput":
        compile_kwargs["loss"] = compile_kwargs["loss"] or "binary_crossentropy"
        out = [Dense(n, activation="softmax")(hidden) for n in n_classes_]
    else:
        # multiclass
        compile_kwargs["loss"] = (
            compile_kwargs["loss"] or "sparse_categorical_crossentropy"
        )
        out = [Dense(n_classes_, activation="softmax")(hidden)]

    model = Model(inp, out)

    model.compile(**compile_kwargs)

    return model


def dynamic_regressor(
    hidden_layer_sizes=(10, ), meta: Optional[Dict[str, Any]]=None, compile_kwargs: Optional[Dict[str, Any]]=None,
) -> Model:
    """Creates a basic MLP regressor dynamically.
    """
    # get parameters
    n_features_in_ = meta["n_features_in_"]
    n_outputs_ = meta["n_outputs_"]

    compile_kwargs["loss"] = compile_kwargs["loss"] or KerasRegressor.r_squared

    inp = Input(shape=(n_features_in_,))

    hidden = inp
    for layer_size in hidden_layer_sizes:
        hidden = Dense(layer_size, activation="relu")(hidden)

    out = Dense(n_outputs_)(hidden)

    model = Model(inp, out)

    model.compile(**compile_kwargs)
    return model
