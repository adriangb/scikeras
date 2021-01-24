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
<a href="https://colab.research.google.com/github/adriangb/scikeras/blob/docs-deploy/refs/master/notebooks/DataTransformers.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>
<!-- #endraw -->

# Data Transformers

Keras support many types of input and output data formats, including:

* Multiple inputs
* Multiple outputs
* Higher-dimensional tensors

In this notebook, we explore how to reconcile this functionality with the sklearn ecosystem via SciKeras data transformer interface.

## Table of contents

* [1. Setup](#1.-Setup)
* [2. Data transformer interface](#2.-Data-transformer-interface)
  * [2.1 get_metadata method](#2.1-get_metadata-method)
* [3. Multiple outputs](#3.-Multiple-outputs)
  * [3.1 Define Keras Model](#3.1-Define-Keras-Model)
  * [3.2 Define output data transformer](#3.2-Define-output-data-transformer)
  * [3.3 Test classifier](#3.3-Test-classifier)
* [4. Multiple inputs](#4-multiple-inputs)
  * [4.1 Define Keras Model](#4.1-Define-Keras-Model)
  * [4.2 Define data transformer](#4.2-Define-data-transformer)
  * [4.3 Test regressor](#4.3-Test-regressor)
* [5. Multidimensional inputs with MNIST dataset](#5.-Multidimensional-inputs-with-MNIST-dataset)
  * [5.1 Define Keras Model](#5.1-Define-Keras-Model)
  * [5.2 Test](#5.2-Test)

## 1. Setup

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

## 2. Data transformer interface

SciKeras enables advanced Keras use cases by providing an interface to convert sklearn compliant data to whatever format your Keras model requires within SciKeras, right before passing said data to the Keras model.

This interface is implemented in the form of two sklearn transformers, one for the features (`X`) and one for the target (`y`).  SciKeras loads these transformers via the `target_encoder` and `feature_encoder` methods.

By default, SciKeras implements `target_encoder` for both KerasClassifier and KerasRegressor to facilitate common types of tasks in sklearn. The default implementations are `scikeras.utils.transformers.ClassifierLabelEncoder` and `scikeras.utils.transformers.RegressorTargetEncoder` for KerasClassifier and KerasRegressor respectively. Information on the types of tasks that these default transformers are able to perform can be found in the [SciKeras docs](https://scikeras.readthedocs.io/en/latest/advanced.html#data-transformers).

Below is an outline of the inner workings of the data transfomer interfaces to help understand when they are called:

```python
if False:  # avoid executing pseudocode
    from scikeras.utils.transformers import (
        ClassifierLabelEncoder,
        RegressorTargetEncoder,
    )


    class BaseWrapper:
        def fit(self, X, y):
            self.target_encoder_ = self.target_encoder
            self.feature_encoder_ = self.feature_encoder
            y = self.target_encoder_.fit_transform(y)
            X = self.feature_encoder_.fit_transform(X)
            self.model_.fit(X, y)
            return self
        
        def predict(self, X):
            X = self.feature_encoder_.transform(X)
            y_pred = self.model_.predict(X)
            return self.target_encoder_.inverse_transform(y_pred)

    class KerasClassifier(BaseWrapper):

        @property
        def target_encoder(self):
            return ClassifierLabelEncoder(loss=self.loss)
        
        def predict_proba(self, X):
            X = self.feature_encoder_.transform(X)
            y_pred = self.model_.predict(X)
            return self.target_encoder_.inverse_transform(y_pred, return_proba=True)


    class KerasRegressor(BaseWrapper):

        @property
        def target_encoder(self):
            return RegressorTargetEncoder()
```

To substitute your own data transformation routine, you must subclass the wrappers and override one of the encoder defining functions. You will have access to all attributes of the wrappers, and you can pass these to your transformer, like we do above with `loss`.

```python
from sklearn.base import BaseEstimator, TransformerMixin
```

```python
if False:  # avoid executing pseudocode

    class MultiOutputTransformer(BaseEstimator, TransformerMixin):
        ...


    class MultiOutputClassifier(KerasClassifier):

        @property
        def target_encoder(self):
            return MultiOutputTransformer(...)
```

### 2.1 get_metadata method

SciKeras recognized an optional `get_metadata` on the transformers. `get_metadata` is expected to return a dicionary of with key strings and arbitrary values. SciKeras will set add these items to the wrappers namespace and make them available to your model building function via the `meta` keyword argument:

```python
if False:  # avoid executing pseudocode

    class MultiOutputTransformer(BaseEstimator, TransformerMixin):
        def get_metadata(self):
            return {"my_param_": "foobarbaz"}


    class MultiOutputClassifier(KerasClassifier):

        @property
        def target_encoder(self):
            return MultiOutputTransformer(...)


    def get_model(meta):
        print(f"Got: {meta['my_param_']}")


    clf = MultiOutputClassifier(model=get_model)
    clf.fit(X, y)  # Got: foobarbaz
    print(clf.my_param_)  # foobarbaz
```

## 3. Multiple outputs

Keras makes it straight forward to define models with multiple outputs, that is a Model with multiple sets of fully-connected heads at the end of the network. This functionality is only available in the Functional Model and subclassed Model definition modes, and is not available when using Sequential.

In practice, the main thing about Keras models with multiple outputs that you need to know as a SciKeras user is that Keras expects `X` or `y` to be a list of arrays/tensors, with one array/tensor for each input/output.

Note that "multiple outputs" in Keras has a slightly different meaning than "multiple outputs" in sklearn. Many tasks that would be considered "multiple output" tasks in sklearn can be mapped to a single "output" in Keras with multiple units. This notebook specifically focuses on the cases that require multiple distinct Keras outputs.

### 3.1 Define Keras Model

Here we define a simple perceptron that has two outputs, corresponding to one binary classification taks and one multiclass classification task. For example, one output might be "image has car" (binary) and the other might be "color of car in image" (multiclass).

```python
def get_clf_model(meta):
    inp = keras.layers.Input(shape=(meta["n_features_in_"]))
    x1 = keras.layers.Dense(100, activation="relu")(inp)
    out_bin = keras.layers.Dense(1, activation="sigmoid")(x1)
    out_cat = keras.layers.Dense(meta["n_classes_"][1], activation="softmax")(x1)
    model = keras.Model(inputs=inp, outputs=[out_bin, out_cat])
    model.compile(
        loss=["binary_crossentropy", "sparse_categorical_crossentropy"]
    )
    return model
```

Let's test that this model works with the kind of inputs and outputs we expect.

```python
X = np.random.random(size=(100, 10))
y_bin = np.random.randint(0, 2, size=(100,))
y_cat = np.random.randint(0, 5, size=(100, ))
y = [y_bin, y_cat]

# build mock meta
meta = {
    "n_features_in_": 10,
    "n_classes_": [2, 5]  # note that we made this a list, one for each output
}

model = get_clf_model(meta=meta)

model.fit(X, y, verbose=0)
y_pred = model.predict(X)
```

```python
print(y_pred[0][:2, :])
```

```python
print(y_pred[1][:2, :])
```

As you can see, our `predict` output is also a list of arrays, except it contains probabilities instead of the class predictions.

Our data transormer's job will be to convert from a single numpy array (which is what the sklearn ecosystem works with) to the list of arrays and then back. Additionally, for classifiers, we will want to be able to convert probabilities to class predictions.

We will structure our data on the sklearn side by column-stacking our list
of arrays. This works well in this case since we have the same number of datapoints in each array.

### 3.2 Define output data transformer

Let's go ahead and protoype this data transformer:

```python
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class MultiOutputTransformer(BaseEstimator, TransformerMixin):

    def fit(self, y):
        y_bin, y_cat = y[:, 0], y[:, 1]
        # Create internal encoders to ensure labels are 0, 1, 2...
        self.bin_encoder_ = LabelEncoder()
        self.cat_encoder_ = LabelEncoder()
        # Fit them to the input data
        self.bin_encoder_.fit(y_bin)
        self.cat_encoder_.fit(y_cat)
        # Save the number of classes
        self.n_classes_ = [
            self.bin_encoder_.classes_.size,
            self.cat_encoder_.classes_.size,
        ]
        # Save number of expected outputs in the Keras model
        # SciKeras will automatically use this to do error-checking
        self.n_outputs_expected_ = 2
        return self

    def transform(self, y: np.ndarray) -> List[np.ndarray]:
        y_bin, y_cat = y[:, 0], y[:, 1]
        # Apply transformers to input array
        y_bin = self.bin_encoder_.transform(y_bin)
        y_cat = self.cat_encoder_.transform(y_cat)
        # Split the data into a list
        return [y_bin, y_cat]

    def inverse_transform(self, y: List[np.ndarray], return_proba: bool = False) -> np.ndarray:
        y_pred_proba = y  # rename for clarity, what Keras gives us are probs
        if return_proba:
            return np.column_stack(y_pred_proba, axis=1)
        # Get class predictions from probabilities
        y_pred_bin = (y_pred_proba[0] > 0.5).astype(int).reshape(-1, )
        y_pred_cat = np.argmax(y_pred_proba[1], axis=1)
        # Pass back through LabelEncoder
        y_pred_bin = self.bin_encoder_.inverse_transform(y_pred_bin)
        y_pred_cat = self.cat_encoder_.inverse_transform(y_pred_cat)
        return np.column_stack([y_pred_bin, y_pred_cat])
    
    def get_metadata(self):
        return {
            "n_classes_": self.n_classes_,
            "n_outputs_expected_": self.n_outputs_expected_,
        }
```

Note that in addition to the usual `transform` and `inverse_transform` methods, we implement the `get_metadata` method to return the `n_classes_` attribute.

Lets test our transformer with the same dataset we previoulsy used to test our model:

```python
tf = MultiOutputTransformer()

y_sklearn = np.column_stack(y)

y_keras = tf.fit_transform(y_sklearn)
print("`y`, as will be passed to Keras:")
print([y_keras[0][:4], y_keras[1][:4]])
```

```python
y_pred_sklearn = tf.inverse_transform(y_pred)
print("`y_pred`, as will be returned to sklearn:")
y_pred_sklearn[:5]
```

```python
print(f"metadata = {tf.get_metadata()}")
```

Since this looks good, we move on to integrating our transformer into our classifier.

```python
from sklearn.metrics import accuracy_score


class MultiOutputClassifier(KerasClassifier):

    @property
    def target_encoder(self):
        return MultiOutputTransformer()
    
    @staticmethod
    def scorer(y_true, y_pred, **kwargs):
        y_bin, y_cat = y_true[:, 0], y_true[:, 1]
        y_pred_bin, y_pred_cat = y_pred[:, 0], y_pred[:, 1]
        # Keras by default uses the mean of losses of each outputs, so here we do the same
        return np.mean([accuracy_score(y_bin, y_pred_bin), accuracy_score(y_cat, y_pred_cat)])
```

### 3.3 Test classifier

```python
from sklearn.preprocessing import StandardScaler

# Use labels as features, just to make sure we can learn correctly
X = y_sklearn
X = StandardScaler().fit_transform(X)
```

```python
clf = MultiOutputClassifier(model=get_clf_model, verbose=0, random_state=0)

clf.fit(X, y_sklearn).score(X, y_sklearn)
```

## 4. Multiple inputs

The process for multiple inputs is similar, but instead of overriding the transformer in `target_encoder` we override `feature_encoder`.

```python
if False:
    from sklearn.base import BaseEstimator, TransformerMixin


    class MultiOutputTransformer(BaseEstimator, TransformerMixin):
        ...


    class MultiOutputClassifier(KerasClassifier):

        @property
        def feature_encoder(self):
            return MultiInputTransformer(...)
```

### 4.1 Define Keras Model

Let's define a Keras **regression** Model with 2 inputs:

```python
def get_reg_model():

    inp1 = keras.layers.Input(shape=(1, ))
    inp2 = keras.layers.Input(shape=(1, ))

    x1 = keras.layers.Dense(100, activation="relu")(inp1)
    x2 = keras.layers.Dense(50, activation="relu")(inp2)

    concat = keras.layers.Concatenate(axis=-1)([x1, x2])

    out = keras.layers.Dense(1)(concat)

    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    model.compile(loss="mse")

    return model
```

And test it with a small mock dataset:

```python
X = np.random.random(size=(100, 2))
y = np.sum(X, axis=1)
X = np.split(X, 2, axis=1)

model = get_reg_model()

model.fit(X, y, verbose=0)
y_pred = model.predict(X).squeeze()
```

```python
from sklearn.metrics import r2_score

r2_score(y, y_pred)
```

Having verified that our model builds without errors and accepts the inputs types we expect, we move onto integrating a transformer into our SciKeras model.

### 4.2 Define data transformer

Just like for overriding `target_encoder`, we just need to define a sklearn transformer and drop it into our SciKeras wrapper. Since we hardcoded the input
shapes into our model and do not rely on any transformer-generated metadata, we can simply use `sklearn.preprocessing.FunctionTransformer`:

```python
from sklearn.preprocessing import FunctionTransformer


class MultiInputRegressor(KerasRegressor):

    @property
    def feature_encoder(self):
        return FunctionTransformer(
            func=lambda X: [X[:, 0], X[:, 1]],
        )
```

Note that we did **not** implement `inverse_transform` (that is, we did not pass an `inverse_func` argument to `FunctionTransformer`) because features are never converted back to their original form.

### 4.3 Test regressor

```python
reg = MultiInputRegressor(model=get_reg_model, verbose=0, random_state=0)

X_sklearn = np.column_stack(X)

reg.fit(X_sklearn, y).score(X_sklearn, y)
```

## 5. Multidimensional inputs with MNIST dataset

In this example, we look at how we can use SciKeras to process the MNIST dataset. The dataset is composed of 60,000 images of digits, each of which is a 2D 28x28 image.

The dataset and Keras Model architecture used come from a [Keras example](https://keras.io/examples/vision/mnist_convnet/). It may be beneficial to understand the Keras model by reviewing that example first.

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train.shape
```

The outputs (labels) are numbers 0-9:

```python
print(y_train.shape)
print(np.unique(y_train))
```

First, we will "flatten" the data into an array of shape `(n_samples, 28*28)` (i.e. a 2D array). This will allow us to use sklearn ecosystem utilities, for example, `sklearn.preprocessing.MinMaxScaler`.

```python
from sklearn.preprocessing import MinMaxScaler

n_samples_train = x_train.shape[0]
n_samples_test = x_test.shape[0]

x_train = x_train.reshape((n_samples_train, -1))
x_test = x_test.reshape((n_samples_test, -1))
x_train = MinMaxScaler().fit_transform(x_train)
x_test = MinMaxScaler().fit_transform(x_test)
```

```python
print(x_train.shape[1:])  # 784 = 28*28
```

```python
print(np.min(x_train), np.max(x_train))  # scaled 0-1
```

Of course, in this case, we could have just as easily used numpy functions to scale our data, but we use `MinMaxScaler` to demonstrate use of the sklearn ecosystem.

### 5.1 Define Keras Model

Next we will define our Keras model (adapted from [keras.io](https://keras.io/examples/vision/mnist_convnet/)):

```python
num_classes = 10
input_shape = (28, 28, 1)


def get_model(meta):
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
        loss="sparse_categorical_crossentropy"
    )
    return model
```

Now let's define a transformer that we will use to reshape our input from the sklearn shape (`(n_samples, 784)`) to the Keras shape (which we will be `(n_samples, 28, 28, 1)`).

```python
class MultiDimensionalClassifier(KerasClassifier):

    @property
    def feature_encoder(self):
        return FunctionTransformer(
            func=lambda X: X.reshape(X.shape[0], *input_shape),
        )
```

```python
clf = MultiDimensionalClassifier(
    model=get_model,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    random_state=0,
)
```

### 5.2 Test

Train and score the model (this takes some time)

```python
clf.fit(x_train, y_train)
```

```python
score = clf.score(x_test, y_test)
print(f"Test score (accuracy): {score:.2f}")
```
