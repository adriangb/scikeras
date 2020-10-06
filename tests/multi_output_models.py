from typing import Any, Dict

import numpy as np

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from tensorflow.python.keras.losses import is_categorical_crossentropy

from scikeras.utils.transformers import Ensure2DTransformer
from scikeras.wrappers import BaseWrapper, KerasClassifier


class MultiOutputClassifier(KerasClassifier):
    """Extend KerasClassifier with the ability to process
    "multilabel-indicator" and "multiclass-multioutput"
    by mapping them to multiple Keras outputs.
    """

    def _get_meta(self, X=None, y=None) -> Dict[str, Any]:
        meta = super()._get_meta(X=X, y=y)
        if y is not None and meta["target_type_"] in (
            "multilabel-indicator",
            "multiclass-multioutput",
        ):
            meta["model_n_outputs_"] = meta["n_outputs_"] = y.shape[1]
        return meta

    def preprocess_y(self, y):
        if self.target_type_ not in ("multilabel-indicator", "multiclass-multioutput"):
            return super().preprocess_y(y)

        y, meta = BaseWrapper.preprocess_y(self, y)
        y = np.split(y, y.shape[1], axis=1)
        loss = self.loss
        target_encoder_ = getattr(self, "target_encoder_", None)
        if self.target_type_ == "multilabel-indicator":
            # y = array([1, 1, 1, 0], [0, 0, 1, 1])
            # each col will be processed as multiple binary classifications
            if target_encoder_ is None:
                target_encoder_ = [FunctionTransformer().fit(y_) for y_ in y]
            meta["classes_"] = [np.array([0, 1])] * len(y)
        else:  # multiclass-multioutput
            # y = array([1, 0, 5], [2, 1, 3])
            # each col be processed as a separate multiclass problem
            if target_encoder_ is None:
                if is_categorical_crossentropy(loss):
                    encoder = make_pipeline(
                        Ensure2DTransformer(), OneHotEncoder(sparse=False),
                    )
                else:
                    encoder = make_pipeline(
                        Ensure2DTransformer(), OrdinalEncoder(dtype=np.float32)
                    )
                target_encoder_ = [clone(encoder).fit(y_) for y_ in y]
            meta["classes_"] = [
                encoder[1].categories_[0] for encoder in target_encoder_
            ]
        meta["target_encoder_"] = target_encoder_
        y = [encoder.transform(y_) for encoder, y_ in zip(target_encoder_, y)]
        meta["n_classes_"] = [c.size for c in meta["classes_"]]
        # Ensure consistency
        if (
            hasattr(self, "model_n_outputs_")
            and not self.model_n_outputs_ == meta["model_n_outputs_"]
        ):
            raise ValueError(
                f"`y` was detected to map to {meta['model_n_outputs_']} model outputs,"
                f" but this {self.__name__} expected {self.model_n_outputs_}"
                " model outputs."
            )
        if hasattr(self, "n_outputs_") and not self.n_outputs_ == meta["n_outputs_"]:
            raise ValueError(
                f"`y` was detected to map to {meta['n_outputs_']} model outputs,"
                f" but this {self.__name__} expected {self.n_outputs_}"
                " model outputs."
            )
        return y, meta

    def postprocess_y(self, y, return_proba=False):
        if self.target_type_ in ("multilabel-indicator", "multiclass-multioutput"):

            target_type_ = self.target_type_

            class_predictions = []

            for i in range(self.n_outputs_):
                if target_type_ == "multilabel-indicator":
                    class_predictions.append(np.argmax(y[i], axis=1))
                else:  # multiclass-multioutput
                    # array([0.8, 0.1, 0.1], [.1, .8, .1]) ->
                    # array(['apple', 'orange'])
                    idx = np.argmax(y[i], axis=-1)
                    if not is_categorical_crossentropy(self.loss):
                        y_ = idx.reshape(-1, 1)
                    else:
                        y_ = np.zeros(y[i].shape, dtype=int)
                        y_[np.arange(y[i].shape[0]), idx] = 1
                    class_predictions.append(
                        self.target_encoder_[i].inverse_transform(y_)
                    )

            if return_proba:
                return np.squeeze(np.column_stack(y))
            else:
                return np.squeeze(np.column_stack(class_predictions)).astype(
                    self.y_dtype_, copy=False
                )
        else:
            return super().postprocess_y(y, return_proba)
