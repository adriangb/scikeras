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
<a href="https://colab.research.google.com/github/adriangb/scikeras/blob/docs-deploy/refs/heads/master/notebooks/DataTransformers.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>
<!-- #endraw -->

# Data Transformers

Keras support many types of input and output data formats, including:

* Multiple inputs
* Multiple outputs
* Higher-dimensional tensors

This notebook walks through an example of the different data transformations and how SciKeras bridges Keras and Scikit-learn.
It may be helpful to have a general understanding of the dataflow before tackling these examples, which is available in
the [data transformer docs](https://www.adriangb.com/scikeras/refs/heads/master/advanced.html#data-transformers).

## Table of contents

* [1. Setup](#1.-Setup)
* [2. Multiple outputs](#2.-Multiple-outputs)
  * [2.1 Define Keras Model](#2.1-Define-Keras-Model)
  * [2.2 Define output data transformer](#2.2-Define-output-data-transformer)
  * [2.3 Test classifier](#2.3-Test-classifier)
* [3. Multiple inputs](#3-multiple-inputs)
  * [3.1 Define Keras Model](#3.1-Define-Keras-Model)
  * [3.2 Define data transformer](#3.2-Define-data-transformer)
  * [3.3 Test regressor](#3.3-Test-regressor)
* [4. Multidimensional inputs with MNIST dataset](#4.-Multidimensional-inputs-with-MNIST-dataset)
  * [4.1 Define Keras Model](#4.1-Define-Keras-Model)
  * [4.2 Test](#4.2-Test)
* [5. Ragged datasets with tf.data.Dataset](#5.-Ragged-datasets-with-tf.data.Dataset)
* [6. Multi-output class_weight](#6.-Multi-output-class_weight)

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

## 2. Multiple outputs

Keras makes it straight forward to define models with multiple outputs, that is a Model with multiple sets of fully-connected heads at the end of the network. This functionality is only available in the Functional Model and subclassed Model definition modes, and is not available when using Sequential.

In practice, the main thing about Keras models with multiple outputs that you need to know as a SciKeras user is that Keras expects `X` or `y` to be a list of arrays/tensors, with one array/tensor for each input/output.

Note that "multiple outputs" in Keras has a slightly different meaning than "multiple outputs" in sklearn. Many tasks that would be considered "multiple output" tasks in sklearn can be mapped to a single "output" in Keras with multiple units. This notebook specifically focuses on the cases that require multiple distinct Keras outputs.

### 2.1 Define Keras Model

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

### 2.2 Define output data transformer

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

Lets test our transformer with the same dataset we previously used to test our model:

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

### 2.3 Test classifier

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

## 3. Multiple inputs

The process for multiple inputs is similar, but instead of overriding the transformer in `target_encoder` we override `feature_encoder`.


```python .noeval
class MultiInputTransformer(BaseEstimator, TransformerMixin):
    ...

class MultiInputClassifier(KerasClassifier):
    @property
    def feature_encoder(self):
        return MultiInputTransformer(...)
```

### 3.1 Define Keras Model

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

### 3.2 Define data transformer

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

### 3.3 Test regressor

```python
reg = MultiInputRegressor(model=get_reg_model, verbose=0, random_state=0)

X_sklearn = np.column_stack(X)

reg.fit(X_sklearn, y).score(X_sklearn, y)
```

## 4. Multidimensional inputs with MNIST dataset

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

# reduce dataset size for faster training
n_samples = 1000
x_train, y_train, x_test, y_test = x_train[:n_samples], y_train[:n_samples], x_test[:n_samples], y_test[:n_samples]
```

```python
print(x_train.shape[1:])  # 784 = 28*28
```

```python
print(np.min(x_train), np.max(x_train))  # scaled 0-1
```

Of course, in this case, we could have just as easily used numpy functions to scale our data, but we use `MinMaxScaler` to demonstrate use of the sklearn ecosystem.

### 4.1 Define Keras Model

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

### 4.2 Test

Train and score the model (this takes some time)

```python
clf.fit(x_train, y_train)
```

```python
score = clf.score(x_test, y_test)
print(f"Test score (accuracy): {score:.2f}")
```

## 5. Ragged datasets with tf.data.Dataset

SciKeras provides a third dependency injection point that operates on the entire dataset: X, y & sample_weight. This `dataset_transformer` is applied after `target_transformer` and `feature_transformer`. One use case for this dependency injection point is to transform data from tabular/array-like to the `tf.data.Dataset` format, which only requires iteration. We can use this to create a `tf.data.Dataset` of ragged tensors.

Note that `dataset_transformer` should accept a single **3 element tuple** as its argument and return value; more details on this are in the [docs](https://www.adriangb.com/scikeras/refs/heads/master/advanced.html#data-transformers).

Let's start by defining our data. We'll have an extra "feature" that marks the observation index, but we'll remove it when we deconstruct our data in the transformer.

```python
feature_1 = np.random.uniform(size=(10, ))
feature_2 = np.random.uniform(size=(10, ))
obs = [0, 0, 0, 1, 1, 2, 3, 3, 4, 4]

X = np.column_stack([feature_1, feature_2, obs]).astype("float32")

y = np.array(["class1"] * 5 + ["class2"] * 5, dtype=str)
```

Next, we define our `dataset_transformer`. We will do this by defining a custom forward transformation outside of the Keras model. Note that we do not define an inverse transformation since that is never used.
Also note that `dataset_transformer` will _always_ be called with `X` (i.e. the first element of the tuple will always be populated), but will be called with `y=None` when used for `predict`. Thus,
you should check if `y` and `sample_weigh` are None before doing any operations on them.

```python
from typing import Tuple, Optional

import tensorflow as tf


def ragged_transformer(data: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]) -> Tuple[tf.RaggedTensor, None, None]:
    X, y, sample_weights = data
    if y is not None:
        y = y.reshape(-1, 1 if len(y.shape) == 1 else y.shape[1])
        y = y[tf.RaggedTensor.from_value_rowids(y, X[:, -1]).row_starts().numpy()]
    if sample_weights is not None:
        sample_weights = sample_weights.reshape(-1, 1 if len(sample_weights.shape) == 1 else sample_weights.shape[1])
        sample_weights = sample_weights[tf.RaggedTensor.from_value_rowids(sample_weights, X[:, -1]).row_starts().numpy()]
    X = tf.RaggedTensor.from_value_rowids(X[:, :-1], X[:, -1])
    return (X, y, sample_weights)
```

In this case, we chose to keep `y` and `sample_weights` as numpy arrays, which will allow us to re-use ClassWeightDataTransformer,
the default `dataset_transformer` for `KerasClassifier`.

Lets quickly test our transformer:

```python
data = ragged_transformer((X, y, None))
data
```

```python
data = ragged_transformer((X, None, None))
data
```

Our shapes look good, and we can handle the `y=None` case.

Because Keras will not accept a RaggedTensor directly, we will need to wrap our entire dataset into a tensorflow `Dataset`. We can do this by adding one more transformation step:

Next, we can add our transormers to our model. We use an sklearn `Pipeline` (generated via `make_pipeline`) to keep ClassWeightDataTransformer operational while implementing our custom transformation.

```python
def dataset_transformer(data: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]) -> Tuple[tf.data.Dataset, None, None]:
    return (tf.data.Dataset.from_tensor_slices(data), None, None)
```

```python
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline


class RaggedClassifier(KerasClassifier):

    @property
    def dataset_transformer(self):
        t1 = FunctionTransformer(ragged_transformer)
        t2 = super().dataset_transformer  # ClassWeightDataTransformer
        t3 = FunctionTransformer(dataset_transformer)
        return make_pipeline(t1, t2, t3)
```

Now we can define a Model. We need some way to handle/flatten our ragged arrays within our model. For this example, we use a custom mean layer, but you could use an Embedding layer, LSTM, etc.

```python
from tensorflow import reduce_mean, reshape
from tensorflow.keras import Sequential, layers


class CustomMean(layers.Layer):

    def __init__(self, axis=None):
        super(CustomMean, self).__init__()
        self._supports_ragged_inputs = True
        self.axis = axis

    def call(self, inputs, **kwargs):
        input_shape = inputs.get_shape()
        return reshape(reduce_mean(inputs, axis=self.axis), (1, *input_shape[1:]))


def get_model(meta):
    inp_shape = meta["X_shape_"][1]-1
    model = Sequential([               
        layers.Input(shape=(inp_shape,), ragged=True),
        CustomMean(axis=0),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

And attach our model to our classifier wrapper:

```python
clf = RaggedClassifier(get_model, loss="bce")
```

Finally, let's train and predict:

```python
clf.fit(X, y)
y_pred = clf.predict(X)
y_pred
```

If we define our custom layers, transformers and wrappers in their own module, we can easily create a self-contained classifier that is able to handle ragged datasets and has a clean Scikit-Learn compatible API:

```python
class RaggedClassifier(KerasClassifier):

    @property
    def dataset_transformer(self):
        t1 = FunctionTransformer(ragged_transformer)
        t2 = super().dataset_transformer  # ClassWeightDataTransformer
        t3 = FunctionTransformer(dataset_transformer)
        return make_pipeline(t1, t2, t3)
    
    def _keras_build_fn(self):
        inp_shape = self.X_shape_[1] - 1
        model = Sequential([               
            layers.Input(shape=(inp_shape,), ragged=True),
            CustomMean(axis=0),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
```

```python
clf = RaggedClassifier(loss="bce")
clf.fit(X, y)
y_pred = clf.predict(X)
y_pred
```

## 6. Multi-output class_weight

In this example, we will use `dataset_transformer` to support multi-output class weights. We will re-use our `MultiOutputTransformer` from our previous example to split the output, then we will create `sample_weights` from `class_weight`

```python
from collections import defaultdict
from typing import Union

from sklearn.utils.class_weight import compute_sample_weight


class DatasetTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, output_names, class_weight=None):
        self.class_weight = class_weight
        self.output_names = output_names

    def fit(self, data: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]) -> "DatasetTransformer":
        return self

    def transform(self, data: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]) -> Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:
        if self.class_weight is None:
            return data
        class_weight = self.class_weight
        if isinstance(class_weight, str):  # handle "balanced"
            class_weight_ = class_weight
            class_weight = defaultdict(lambda: class_weight_)
        X, y, sample_weights = data
        assert sample_weights is None, "Cannot use class_weight & sample_weights together"
        if y is not None:
            # y should be a list of arrays, as split up by MultiOutputTransformer
            sample_weights = {
                output_name: compute_sample_weight(class_weight[output_num], output_data)
                for output_num, (output_name, output_data) in enumerate(zip(self.output_names, y))
            }
            # Note: class_weight is expected to be indexable by output_number in sklearn
            # see https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
            # It is trivial to change the expected format to match Keras' ({output_name: weights, ...})
            # see https://github.com/keras-team/keras/issues/4735#issuecomment-267473722
        return X, y, sample_weights

```

```python
def get_model(meta, compile_kwargs):
    inp = keras.layers.Input(shape=(meta["n_features_in_"]))
    x1 = keras.layers.Dense(100, activation="relu")(inp)
    out_bin = keras.layers.Dense(1, activation="sigmoid")(x1)
    out_cat = keras.layers.Dense(meta["n_classes_"][1], activation="softmax")(x1)
    model = keras.Model(inputs=inp, outputs=[out_bin, out_cat])
    model.compile(
        loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
        optimizer=compile_kwargs["optimizer"]
    )
    return model


class CustomClassifier(KerasClassifier):

    @property
    def target_encoder(self):
        return MultiOutputTransformer()
    
    @property
    def dataset_transformer(self):
        return DatasetTransformer(
            output_names=self.model_.output_names,
            class_weight=self.class_weight
        )
```

Next, we define the data. We'll use `sklearn.datasets.make_blobs` to generate a relatively noisy dataset:

```python
from sklearn.datasets import make_blobs


X, y = make_blobs(centers=3, random_state=0, cluster_std=20)
# make a binary target for "is the value of the first class?"
y_bin = y == y[0]
y = np.column_stack([y_bin, y])
```

Test the model without specifying class weighting:

```python
clf = CustomClassifier(get_model, epochs=100, verbose=0, random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X)
(_, counts_bin) = np.unique(y_pred[:, 0], return_counts=True)
print(counts_bin)
(_, counts_cat) = np.unique(y_pred[:, 1], return_counts=True)
print(counts_cat)
```

As you can see, without `class_weight="balanced"`, our classifier only predicts mainly a single class for the first output. Now with `class_weight="balanced"`:

```python
clf = CustomClassifier(get_model, class_weight="balanced", epochs=100, verbose=0, random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X)
(_, counts_bin) = np.unique(y_pred[:, 0], return_counts=True)
print(counts_bin)
(_, counts_cat) = np.unique(y_pred[:, 1], return_counts=True)
print(counts_cat)
```

Now, we get (mostly) balanced classes. But what if we want to specify our classes manually? You will notice that in when we defined `DatasetTransformer`, we gave it the ability to handle
a list of class weights. For demonstration purposes, we will highly bias towards the second class in each output:

```python
clf = CustomClassifier(get_model, class_weight=[{0: 0.1, 1: 1}, {0: 0.1, 1: 1, 2: 0.1}], epochs=100, verbose=0, random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X)
(_, counts_bin) = np.unique(y_pred[:, 0], return_counts=True)
print(counts_bin)
(_, counts_cat) = np.unique(y_pred[:, 1], return_counts=True)
print(counts_cat)
```

Or mixing the two methods, because our first output is unbalanced but our second is (presumably) balanced:

```python
clf = CustomClassifier(get_model, class_weight=["balanced", None], epochs=100, verbose=0, random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X)
(_, counts_bin) = np.unique(y_pred[:, 0], return_counts=True)
print(counts_bin)
(_, counts_cat) = np.unique(y_pred[:, 1], return_counts=True)
print(counts_cat)
```
