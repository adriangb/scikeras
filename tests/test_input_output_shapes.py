import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.testing_utils import get_test_data

from scikeras.wrappers import KerasClassifier
from scikeras.wrappers import KerasRegressor


# Defaults
INPUT_DIM = 5
HIDDEN_DIM = 5
TRAIN_SAMPLES = 10
TEST_SAMPLES = 5
NUM_CLASSES = 2
BATCH_SIZE = 5
EPOCHS = 1


class FunctionalAPIMultiInputClassifier(KerasClassifier):
    """Tests Functional API Classifier with 2 inputs.
    """

    def _keras_build_fn(self, X, n_classes_):
        inp1 = Input((1,))
        inp2 = Input((3,))

        x1 = Dense(100)(inp1)
        x2 = Dense(100)(inp2)

        x3 = Concatenate(axis=-1)([x1, x2])

        cat_out = Dense(n_classes_, activation="softmax")(x3)

        model = Model([inp1, inp2], [cat_out])
        losses = ["categorical_crossentropy"]
        model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

        return model

    @staticmethod
    def preprocess_X(X):
        """To support multiple inputs, a custom method must be defined.
        """
        return [X[:, 0], X[:, 1:4]], dict()


class FunctionalAPIMultiOutputClassifier(KerasClassifier):
    """Tests Functional API Classifier with 2 outputs of different type.
    """

    def _keras_build_fn(self, X, n_classes_):
        inp = Input((4,))

        x1 = Dense(100)(inp)

        binary_out = Dense(1, activation="sigmoid")(x1)
        cat_out = Dense(n_classes_[1], activation="softmax")(x1)

        model = Model([inp], [binary_out, cat_out])
        losses = ["binary_crossentropy", "categorical_crossentropy"]
        model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

        return model

    def score(self, X, y):
        """Taken from sklearn.multiouput.MultiOutputClassifier
        """
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))


class FunctionAPIMultiLabelClassifier(KerasClassifier):
    """Tests Functional API Classifier with multiple binary outputs.
    """

    def _keras_build_fn(self, X, n_outputs_):
        inp = Input((4,))

        x1 = Dense(100)(inp)

        outputs = []
        for _ in range(n_outputs_):
            # simulate multiple binary classification outputs
            # in reality, these would come from different nodes
            outputs.append(Dense(1, activation="sigmoid")(x1))

        model = Model(inp, outputs)
        losses = "binary_crossentropy"
        model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

        return model


class FunctionAPIMultiOutputRegressor(KerasRegressor):
    """Tests Functional API Regressor with multiple outputs.
    """

    def _keras_build_fn(self, X, n_outputs_):
        inp = Input((INPUT_DIM,))

        x1 = Dense(100)(inp)

        outputs = [Dense(n_outputs_)(x1)]

        model = Model([inp], outputs)
        losses = "mean_squared_error"
        model.compile(optimizer="adam", loss=losses, metrics=["mse"])

        return model


def test_multi_input():
    """Tests custom multi-input Keras model.
    """
    clf = FunctionalAPIMultiInputClassifier()
    (x_train, y_train), (x_test, y_test) = get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(4,),
        num_classes=3,
    )

    clf.fit(x_train, y_train)
    clf.predict(x_test)
    clf.score(x_train, y_train)


def test_multi_output():
    """Compares to scikit-learn RandomForestClassifier classifier.
    """
    clf_keras = FunctionalAPIMultiOutputClassifier()
    clf_sklearn = RandomForestClassifier()

    # generate data
    X = np.random.rand(10, 4)
    y1 = np.random.randint(0, 2, size=(10, 1))
    y2 = np.random.randint(0, 11, size=(10, 1))
    y = np.hstack([y1, y2])

    clf_keras.fit(X, y)
    y_wrapper = clf_keras.predict(X)
    clf_keras.score(X, y)

    clf_sklearn.fit(X, y)
    y_sklearn = clf_sklearn.predict(X)

    assert y_sklearn.shape == y_wrapper.shape


def test_multi_label_clasification():
    """Compares to scikit-learn RandomForestClassifier classifier.
    """
    clf_keras = FunctionAPIMultiLabelClassifier()
    clf_sklearn = RandomForestClassifier()
    # taken from https://scikit-learn.org/stable/modules/multiclass.html
    y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
    y = MultiLabelBinarizer().fit_transform(y)

    (x_train, _), (_, _) = get_test_data(
        train_samples=y.shape[0],
        test_samples=0,
        input_shape=(4,),
        num_classes=3,
    )

    clf_keras.fit(x_train, y)
    y_pred_keras = clf_keras.predict(x_train)
    clf_keras.score(x_train, y)

    clf_sklearn.fit(x_train, y)
    y_pred_sklearn = clf_sklearn.predict(x_train)
    clf_sklearn.score(x_train, y)

    assert y_pred_keras.shape == y_pred_sklearn.shape


def test_multi_output_regression():
    """Compares to scikit-learn RandomForestRegressor.
    """
    reg_keras = FunctionAPIMultiOutputRegressor()
    reg_sklearn = RandomForestRegressor()
    # taken from https://scikit-learn.org/stable/modules/multiclass.html
    (X, _), (_, _) = get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES,
    )
    y = np.random.random_sample(size=(TRAIN_SAMPLES, NUM_CLASSES))

    reg_keras.fit(X, y)
    y_pred_keras = reg_keras.predict(X)
    reg_keras.score(X, y)

    reg_sklearn.fit(X, y)
    y_pred_sklearn = reg_sklearn.predict(X)
    reg_sklearn.score(X, y)

    assert y_pred_keras.shape == y_pred_sklearn.shape


def test_incompatible_output_dimensions():
    """Compares to the scikit-learn RandomForestRegressor classifier.
    """
    # create dataset with 4 outputs
    X = np.random.rand(10, 20)
    y = np.random.randint(low=0, high=3, size=(10, 4))

    # create a model with 2 outputs
    def build_fn_clf():
        """Builds a Sequential based classifier."""
        model = Sequential()
        model.add(Dense(20, input_shape=(20,), activation="relu"))
        model.add(Dense(np.unique(y).size, activation="relu"))
        model.compile(
            optimizer="sgd",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    clf = KerasClassifier(build_fn=build_fn_clf)

    with pytest.raises(RuntimeError):
        clf.fit(X, y)
