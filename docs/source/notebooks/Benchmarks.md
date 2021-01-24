---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[![Run in Colab](https://www.tensorflow.org/images/colab_logo_32px.png)](https://colab.research.google.com/github/adriangb/scikeras/blob/master/docs/source/notebooks/Benchmarks.ipynb) Run in Colab

# SciKeras Benchmarks

SciKeras wraps Keras Models, but does not alter their performance since all of the heavy lifting still happens within Keras/Tensorflow. In this notebook, we compare the performance and accuracy of a pure-Keras Model to the same model wrapped in SciKeras.

## Table of contents

- [SciKeras Benchmarks](#scikeras-benchmarks)
  - [Table of contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Define Keras Model](#define-keras-model)
  - [Keras benchmarks](#keras-benchmarks)
  - [SciKeras benchmark](#scikeras-benchmark)

Install SciKeras

```python
try:
    import scikeras
except ImportError:
    !python -m pip install scikeras
```

Silence TensorFlow warnings to keep output succint.

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

<a id='1'></a>
## Dataset

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

<a id='2'></a>
## Define Keras Model

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

<a id='3'></a>
## Keras benchmarks

```python
fit_kwargs = {"batch_size": 128, "validation_split": 0.1, "verbose": 0, "epochs": 5}
```

```python
from sklearn.metrics import accuracy_score
from scikeras._utils import TFRandomState
```

```python
from time import time

with TFRandomState(seed=0):  # we force a TF random state to be able to compare accuracy
    model = get_model()
    start = time()
    model.fit(x_train, y_train, **fit_kwargs)
    print(f"Training time: {time()-start:.2f}")
    y_pred = np.argmax(model.predict(x_test), axis=1)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

<a id='4'></a>
## SciKeras benchmark

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
