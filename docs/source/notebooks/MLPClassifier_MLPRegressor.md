---
jupyter:
  jupytext:
    formats: ipynb,md
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

<!-- #raw -->
<a href="https://colab.research.google.com/github/adriangb/scikeras/blob/docs-deploy/refs/heads/master/notebooks/MLPClassifier_MLPRegressor.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>
<!-- #endraw -->

# MLPClassifier and MLPRegressor in SciKeras

SciKeras is a bridge between Keras and Scikit-Learn. As such, one of SciKeras' design goals is to be able to create a Scikit-Learn style estimator backed by Keras.

This notebook implements an estimator that is analogous to `sklearn.neural_network.MLPClassifier` using Keras. This estimator should (for the most part) work as a drop-in replacement for `MLPClassifier`!

## Table of contents

* [1. Setup](#1.-Setup)
* [2. Defining the Keras Model](#2.-Defining-the-Keras-Model)
  * [2.1 Inputs](#2.1-Inputs)
  * [2.2 Hidden layers](#2.2-Hidden-layers)
  * [2.3 Output layers](#2.3-Output-layers)
  * [2.4 Losses and optimizer](#2.4-Losses-and-optimizer)
  * [2.5 Wrapping with SciKeras](#2.5-Wrapping-with-SciKeras)
* [3. Testing our classifier](#3.-Testing-our-classifier)
* [4. Self contained MLPClassifier](#4.-Self-contained-MLPClassifier)
  * [4.1 Subclassing](#4.1-Subclassing)
* [5. MLPRegressor](#5.-MLPRegressor)

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

## 2. Defining the Keras Model

First, we outline our model building function, using a `Sequential` Model:

```python
def get_clf_model():
    model = keras.Sequential()
    return model
```

### 2.1 Inputs

We need to define an input layer for Keras. SciKeras allows you to dynamically determine the input size based on the features (`X`). To do this, you need to add the `meta` parameter to `get_clf_model`'s parameters. `meta` will be a dictionary with all of the `meta` attributes that `KerasClassifier` generates during the `fit` call, including `n_features_in_`, which we will use to dynamically size the input layer.

```python
from typing import Dict, Iterable, Any


def get_clf_model(meta: Dict[str, Any]):
    model = keras.Sequential()
    inp = keras.layers.Input(shape=(meta["n_features_in_"]))
    model.add(inp)
    return model
```

### 2.2 Hidden Layers

Multilayer perceptrons are generally composed of an input layer, an output layer and 0 or more hidden layers. The size of the hidden layers is specified via the `hidden_layer_sizes` parameter in MLClassifier, where the the ith element represents the number of neurons in the ith hidden layer. Let's add that parameter:

```python
def get_clf_model(hidden_layer_sizes: Iterable[int], meta: Dict[str, Any]):
    model = keras.Sequential()
    inp = keras.layers.Input(shape=(meta["n_features_in_"]))
    model.add(inp)
    for hidden_layer_size in hidden_layer_sizes:
        layer = keras.layers.Dense(hidden_layer_size, activation="relu")
        model.add(layer)
    return model
```

### 2.3 Output layers

The output layer needs to reflect the type of classification task being performed. Here, we will handle 2 cases:

- binary classification: single output unit with sigmoid activation
- multiclass classification: one output unit for each class, with softmax activation
The main complication arises from determining which one to use. Like with the input features, SciKeras provides useful information on the target within the `meta` parameter. Specifically, we will use the `n_classes_` and `target_type_` attributes to determine the number of output units and activation function.

```python
def get_clf_model(hidden_layer_sizes: Iterable[int], meta: Dict[str, Any]):
    model = keras.Sequential()
    inp = keras.layers.Input(shape=(meta["n_features_in_"]))
    model.add(inp)
    for hidden_layer_size in hidden_layer_sizes:
        layer = keras.layers.Dense(hidden_layer_size, activation="relu")
        model.add(layer)
    if meta["target_type_"] == "binary":
        n_output_units = 1
        output_activation = "sigmoid"
    elif meta["target_type_"] == "multiclass":
        n_output_units = meta["n_classes_"]
        output_activation = "softmax"
    else:
        raise NotImplementedError(f"Unsupported task type: {meta['target_type_']}")
    out = keras.layers.Dense(n_output_units, activation=output_activation)
    model.add(out)
    return model
```

For now, we raise a `NotImplementedError` for other target types. For an example handling multi-output target types, see the [Multi Output notebook](https://colab.research.google.com/github/adriangb/scikeras/blob/master/notebooks/MultiInput.ipynb).

### 2.4 Losses and optimizer

Like the output layer, the loss must match the type of classification task. Generally, it is easier and safet to allow SciKeras to compile your model for you by passing the loss to `KerasClassifier` directly (`KerasClassifier(loss="binary_crossentropy")`). However, in order to implement custom logic around the choice of loss function, we compile the model ourselves within `get_clf_model`; SciKeras will not re-compile the model.

```python
def get_clf_model(hidden_layer_sizes: Iterable[int], meta: Dict[str, Any]):
    model = keras.Sequential()
    inp = keras.layers.Input(shape=(meta["n_features_in_"]))
    model.add(inp)
    for hidden_layer_size in hidden_layer_sizes:
        layer = keras.layers.Dense(hidden_layer_size, activation="relu")
        model.add(layer)
    if meta["target_type_"] == "binary":
        n_output_units = 1
        output_activation = "sigmoid"
        loss = "binary_crossentropy"
    elif meta["target_type_"] == "multiclass":
        n_output_units = meta["n_classes_"]
        output_activation = "softmax"
        loss = "sparse_categorical_crossentropy"
    else:
        raise NotImplementedError(f"Unsupported task type: {meta['target_type_']}")
    out = keras.layers.Dense(n_output_units, activation=output_activation)
    model.add(out)
    model.compile(loss=loss)
    return model
```

At this point, we have a valid, compiled model. However if we want to be able to tune the optimizer, we should accept `compile_kwargs` as a parameter in `get_clf_model`. `compile_kwargs` will be a dictionary containing valid `kwargs` for `Model.compile`, so we can unpack it directly like `model.compile(**compile_kwargs)`. In this case however, we will only be taking the `optimizer` kwarg.

```python
def get_clf_model(hidden_layer_sizes: Iterable[int], meta: Dict[str, Any], compile_kwargs: Dict[str, Any]):
    model = keras.Sequential()
    inp = keras.layers.Input(shape=(meta["n_features_in_"]))
    model.add(inp)
    for hidden_layer_size in hidden_layer_sizes:
        layer = keras.layers.Dense(hidden_layer_size, activation="relu")
        model.add(layer)
    if meta["target_type_"] == "binary":
        n_output_units = 1
        output_activation = "sigmoid"
        loss = "binary_crossentropy"
    elif meta["target_type_"] == "multiclass":
        n_output_units = meta["n_classes_"]
        output_activation = "softmax"
        loss = "sparse_categorical_crossentropy"
    else:
        raise NotImplementedError(f"Unsupported task type: {meta['target_type_']}")
    out = keras.layers.Dense(n_output_units, activation=output_activation)
    model.add(out)
    model.compile(loss=loss, optimizer=compile_kwargs["optimizer"])
    return model
```

### 2.5 Wrapping with SciKeras

Our last step in defining our model is to wrap it with SciKeras. A couple of things to note are:
- Every user-defined parameter in `model`/`get_clf_model` (in our case just `hidden_layer_sizes`) must be defined as a keyword argument to `KerasClassifier` with a default value.
- Keras defaults to `"rmsprop"` for `optimizer`. We set it to `"adam"` to mimic MLPClassifier.
- We set the learning rate for the optimizer to `0.001`, again to mimic MLPClassifier. We set this parameter using [parameter routing](https://www.adriangb.com/scikeras/stable/advanced.html#routed-parameters).
- Other parameters, such as `activation`, can be added similar to `hidden_layer_sizes`, but we omit them here for simplicity.

```python
clf = KerasClassifier(
    model=get_clf_model,
    hidden_layer_sizes=(100, ),
    optimizer="adam",
    optimizer__learning_rate=0.001,
    epochs=50,
    verbose=0,
)
```

## 3. Testing our classifier

Before continouing, we will run a small test to make sure we get somewhat reasonable results.

```python
from sklearn.datasets import make_classification


X, y = make_classification()

# check that fit works
clf.fit(X, y)
# check score
print(clf.score(X, y))
```

We get a score above 0.7, which is reasonable and indicates that our classifier is generally working.

## 4. Self contained MLPClassifier

You will have noticed that up until now, we define our Keras model in a function and pass that function to `KerasClassifier` via the `model` argument.

This is convenient, but it does not give us a self-contained class that we could package within a module for users to instantiate. To do that, we need to subclass `KerasClassifier`.

### 4.1 Subclassing

By subclassing KerasClassifier, you can embed your Keras model into directly into your estimator class. We start by inheriting from KerasClassifier and defining an `__init__` method with all of our parameters.

```python
class MLPClassifier(KerasClassifier):

    def __init__(
        self,
        hidden_layer_sizes=(100, ),
        optimizer="adam",
        optimizer__learning_rate=0.001,
        epochs=200,
        verbose=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
```

Next, we will embed our model into `_keras_build_fn`, which takes the place of `get_clf_model`. Note that since this is now an part of the model, we no longer need to accept the any parameters in the function signature. We still accept `compile_kwargs` because we use it to get the optimizer initialized with all of it's parameters.

```python
class MLPClassifier(KerasClassifier):

    def __init__(
        self,
        hidden_layer_sizes=(100, ),
        optimizer="adam",
        optimizer__learning_rate=0.001,
        epochs=200,
        verbose=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.n_features_in_))
        model.add(inp)
        for hidden_layer_size in self.hidden_layer_sizes:
            layer = keras.layers.Dense(hidden_layer_size, activation="relu")
            model.add(layer)
        if self.target_type_ == "binary":
            n_output_units = 1
            output_activation = "sigmoid"
            loss = "binary_crossentropy"
        elif self.target_type_ == "multiclass":
            n_output_units = self.n_classes_
            output_activation = "softmax"
            loss = "sparse_categorical_crossentropy"
        else:
            raise NotImplementedError(f"Unsupported task type: {self.target_type_}")
        out = keras.layers.Dense(n_output_units, activation=output_activation)
        model.add(out)
        model.compile(loss=loss, optimizer=compile_kwargs["optimizer"])
        return model
```

Let's check that our subclassed model works:

```python
clf = MLPClassifier(epochs=20)  # for notebook execution time

# check score
print(clf.fit(X, y).score(X, y))
```

## 5. MLPRegressor

The process for MLPRegressor is similar, we only change the loss function and output layers.

```python
class MLPRegressor(KerasRegressor):

    def __init__(
        self,
        hidden_layer_sizes=(100, ),
        optimizer="adam",
        optimizer__learning_rate=0.001,
        epochs=200,
        verbose=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.n_features_in_))
        model.add(inp)
        for hidden_layer_size in self.hidden_layer_sizes:
            layer = keras.layers.Dense(hidden_layer_size, activation="relu")
            model.add(layer)
        out = keras.layers.Dense(1)
        model.add(out)
        model.compile(loss="mse", optimizer=compile_kwargs["optimizer"])
        return model
```


```python
from sklearn.datasets import make_regression


reg = MLPRegressor(epochs=20)  # for notebook execution time

# Define a simple linear relationship
y = np.arange(100)
X = (y/2).reshape(-1, 1)

# check score
reg.fit(X, y)
print(reg.score(X, y))
```
