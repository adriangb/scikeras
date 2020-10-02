import numpy as np

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from tensorflow.python.keras.losses import is_categorical_crossentropy

from scikeras.utils.transformers import Ensure2DTransformer
from scikeras.wrappers import BaseWrapper, KerasClassifier


class MultiOuputClassifier(KerasClassifier):
    """Extend KerasClassifier with the ability to process
    "multilabel-indicator" and "multiclass-multioutput"
    by mapping them to multiple Keras outputs.
    """

    def preprocess_y(self, y, reset):
        if self.target_type_ in ("multilabel-indicator", "multiclass-multioutput"):
            y = BaseWrapper.preprocess_y(self, y, reset)
            loss = self.loss

            if self.target_type_ == "multilabel-indicator":
                # y = array([1, 1, 1, 0], [0, 0, 1, 1])
                # split into lists for multi-output Keras
                # will be processed as multiple binary classifications
                classes_ = [np.array([0, 1])] * y.shape[1]
                encoders_ = [None] * y.shape[1]
                y = np.split(y, y.shape[1], axis=1)
                model_n_outputs_ = len(y)
            else:  # multiclass-multioutput
                # y = array([1, 0, 5], [2, 1, 3])
                # split into lists for multi-output Keras
                # each will be processed as a separate multiclass problem
                y = np.split(y, y.shape[1], axis=1)
                model_n_outputs_ = len(y)
                # encode
                if reset:
                    if is_categorical_crossentropy(loss):
                        # one-hot encode
                        encoder = make_pipeline(
                            Ensure2DTransformer(), OneHotEncoder(sparse=False),
                        )
                    else:
                        encoder = make_pipeline(
                            Ensure2DTransformer(), OrdinalEncoder(dtype=np.float32)
                        )
                    encoders_ = [clone(encoder) for _ in range(len(y))]
                else:
                    encoders_ = self.encoders_
                y = [encoder.fit_transform(y_) for encoder, y_ in zip(encoders_, y)]
                classes_ = [encoder[1].categories_[0] for encoder in encoders_]

            if len(classes_) == 1:
                n_classes_ = classes_[0].size
                classes_ = classes_[0]
                n_outputs_ = 1
            else:
                n_classes_ = [class_.shape[0] for class_ in classes_]
                n_outputs_ = len(n_classes_)

            if reset:
                self.classes_ = (
                    self.user_classes_ if self.user_classes_ is not None else classes_
                )
                self.encoders_ = encoders_
                self.n_outputs_ = n_outputs_
                self.model_n_outputs_ = model_n_outputs_
                self.n_classes_ = n_classes_
            else:
                check = (
                    (self.classes_, classes_)
                    if isinstance(self.classes_, list)
                    else ([self.classes_], [classes_])
                )
                for col, (old, new) in enumerate(zip(*check)):
                    is_subset = len(set(old) - set(new)) == 0
                    if not is_subset:
                        raise ValueError(
                            f"Col {col} of `y` was detected to have {new} classes,"
                            f" but this {self.__name__} expected only {old} classes."
                        )
                if not self.model_n_outputs_ == model_n_outputs_:
                    raise ValueError(
                        f"`y` was detected to map to {model_n_outputs_} model outputs,"
                        f" but this {self.__name__} expected {self.model_n_outputs_}"
                        " model outputs."
                    )
                if not self.n_outputs_ == n_outputs_:
                    raise ValueError(
                        f"`y` was detected to map to {n_outputs_} model outputs,"
                        f" but this {self.__name__} expected {self.n_outputs_}"
                        " model outputs."
                    )

            return y
        else:
            return super().preprocess_y(y, reset)

    def postprocess_y(self, y, return_proba=False):
        if self.target_type_ in ("multilabel-indicator", "multiclass-multioutput"):
            if not isinstance(y, list):
                # convert single-target y to a list for easier processing
                y = [y]

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
                    class_predictions.append(self.encoders_[i].inverse_transform(y_))

            if return_proba:
                return np.squeeze(np.column_stack(y))
            else:
                return np.squeeze(np.column_stack(class_predictions)).astype(
                    self.y_dtype_, copy=False
                )
        else:
            return super().postprocess_y(y, return_proba)
