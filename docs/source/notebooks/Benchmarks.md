---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #raw -->
<a href="https://colab.research.google.com/github/adriangb/scikeras/blob/docs-deploy/refs/heads/master/notebooks/Benchmarks.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>
<!-- #endraw -->

# SciKeras Benchmarks

SciKeras wraps Keras Models, but does not alter their performance since all of the heavy lifting still happens within Keras/Tensorflow. In this notebook, we compare the performance and accuracy of a pure-Keras Model to the same model wrapped in SciKeras.

## Table of contents

* [1. Setup](#1.-Setup)
* [2. Dataset](#2.-Dataset)
* [3. Define Keras Model](#3.-Define-Keras-Model)
* [4. Keras benchmarks](#4.-Keras-benchmarks)
* [5. SciKeras benchmark](#5.-SciKeras-benchmark)

## 1. Setup

```python
try:
    import scikeras
except ImportError:
    !python -m pip install scikeras
```

Silence TensorFlow logging to keep output succinct.

```python
import warnings
from tensorflow import get_logger
get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="Setting the random state for TF")
```

```python
import numpy as np
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow import keras
```

## 2. Dataset

We will be using the MNIST dataset available within Keras.

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# Reduce dataset size for faster benchmarks
x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:500], y_test[:500]
```

## 3. Define Keras Model

Next we will define our Keras model (adapted from [keras.io](https://keras.io/examples/vision/mnist_convnet/)):

```python
num_classes = 10
input_shape = (28, 28, 1)


def get_model():
    model = keras.Sequential(
        [
            keras.Input(input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam"
    )
    return model
```

## 4. Keras benchmarks

```python
fit_kwargs = {"batch_size": 128, "validation_split": 0.1, "verbose": 0, "epochs": 5}
```

```python
from sklearn.metrics import accuracy_score
from scikeras.utils.random_state import tensorflow_random_state
```

```python
from time import time

with tensorflow_random_state(seed=0):  # we force a TF random state to be able to compare accuracy
    model = get_model()
    start = time()
    model.fit(x_train, y_train, **fit_kwargs)
    print(f"Training time: {time()-start:.2f}")
    y_pred = np.argmax(model.predict(x_test), axis=1)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

## 5. SciKeras benchmark

```python
clf = KerasClassifier(
    model=get_model,
    random_state=0,
    **fit_kwargs
)
```

```python
start = time()
clf.fit(x_train, y_train)
print(f"Training time: {time()-start:.2f}")
y_pred = clf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

As you can see, the overhead for SciKeras is <1 sec, and the accuracy is identical.
