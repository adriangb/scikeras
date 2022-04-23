"""Test that BaseWrapper for uses other than KerasClassifier and KerasRegressor.
"""
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

from scikeras.wrappers import BaseWrapper


class AutoEncoderTransformer(BaseWrapper, TransformerMixin):
    """Enables the ``transform`` and ``fit_transform`` methods."""

    def fit(self, X):
        self.initialize(X)
        return self

    def transform(self, X):
        return self.predict(X)


class TestAutoencoder:
    def test_simple_autoencoder_mnist(self):
        """Tests an autoencoder following."""
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
        fit_params = {
            "batch_size": 256,
            "shuffle": True,
            "random_state": 0,
        }

        autoencoder = BaseWrapper(
            model=autoencoder_model, **compile_params, **fit_params
        )
        encoder = AutoEncoderTransformer(
            model=encoder_model, **compile_params, **fit_params
        )
        decoder = AutoEncoderTransformer(
            model=decoder_model, **compile_params, **fit_params
        )

        # Initialize autoencoder
        autoencoder.initialize(x_train, x_train)

        # Test shape of output images
        encoded_images = encoder.fit_transform(x_train)
        assert encoded_images.shape == (x_train.shape[0], 32)
        roundtrip_imgs = decoder.fit_transform(encoded_images)
        assert roundtrip_imgs.shape == x_train.shape

        # Get current MSE of reconstruction
        roundtrip_imgs = decoder.transform(encoder.transform(x_test))
        mse_untrained = mean_squared_error(roundtrip_imgs, x_test)

        # Train for 1 epoch
        autoencoder.partial_fit(x_train, x_train)

        # Test that training is working by checking that the
        # MSE decreases by at least 5x
        roundtrip_imgs = decoder.transform(encoder.transform(x_test))
        mse_trained = mean_squared_error(roundtrip_imgs, x_test)
        assert mse_trained < mse_untrained * 0.2
