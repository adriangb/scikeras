"""Test that BaseWrapper for uses other than KerasClassifier and KerasRegressor.
"""
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

from scikeras.wrappers import BaseWrapper


class AutoEncoderTransformer(BaseWrapper, TransformerMixin):
    """A mixin that enables the transform and fit_transform methods.
    """

    def transform(self, X):
        return self.predict(X)


class TestAutoencoder:
    def test_simple_autoencoder_mnist(self):
        """Tests an autoencoder following .
        """
        # Data
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        # Model
        encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
        # This is our input image
        input_img = keras.Input(shape=(784,))
        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(encoding_dim, activation="relu")(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(784, activation="sigmoid")(encoded)
        # This model maps an input to its reconstruction
        autoencoder_model = keras.Model(input_img, decoded)
        # This model maps an input to its encoded representation
        encoder_model = keras.Model(input_img, encoded)
        # This is our encoded (32-dimensional) input
        encoded_input = keras.Input(shape=(encoding_dim,))
        # Retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder_model.layers[-1]
        # Create the decoder model
        decoder_model = keras.Model(encoded_input, decoder_layer(encoded_input))

        # Wrap model
        compile_params = {"optimizer": "adam", "loss": "binary_crossentropy"}
        fit_params = {"epochs": 20, "batch_size": 256, "shuffle": True}

        autoencoder = BaseWrapper(
            model=autoencoder_model, **compile_params, **fit_params
        )
        encoder = AutoEncoderTransformer(
            model=encoder_model, **compile_params, **fit_params
        )
        decoder = AutoEncoderTransformer(
            model=decoder_model, **compile_params, **fit_params
        )

        # Training
        autoencoder.fit(x_train, x_train)
        roundtrip_imgs = decoder.fit_transform(encoder.fit_transform(x_test))
        mse = mean_squared_error(roundtrip_imgs, x_test)
        assert mse <= 0.05  # 0.05 comes from experimentation
