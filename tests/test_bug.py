import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from scikeras.wrappers import KerasClassifier

def get_keras():
    (X_train, y_train), _ = mnist.load_data()
    X_train, y_train = X_train[:100], y_train[:100]
    X_train = X_train.reshape(X_train.shape[0], 784).astype("float32") / 255
    return X_train, y_train

def _keras_build_fn(opt="sgd"):
    model = Sequential([Dense(512, input_shape=(784,)), Activation("relu"),
                        Dense(10), Activation("softmax")])
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

def test_keras():
    X, y = get_keras()
    assert X.ndim == 2 and X.shape[-1] == 784 and y.ndim == 1
    model = KerasClassifier(build_fn=_keras_build_fn)
    model.fit(X, y)

if __name__ == "__main__":
    test_keras()