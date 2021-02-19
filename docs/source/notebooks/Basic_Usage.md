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
<a href="https://colab.research.google.com/github/adriangb/scikeras/blob/docs-deploy/refs/heads/master/notebooks/Basic_Usage.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>
<!-- #endraw -->

# Basic usage

`SciKeras` is designed to maximize interoperability between `sklearn` and `Keras/TensorFlow`. The aim is to keep 99% of the flexibility of `Keras` while being able to leverage most features of `sklearn`. Below, we show the basic usage of `SciKeras` and how it can be combined with `sklearn`.

This notebook shows you how to use the basic functionality of `SciKeras`.

## Table of contents

* [1. Setup](#1.-Setup)
* [2. Training a classifier and making predictions](#2.-Training-a-classifier-and-making-predictions)
  * [2.1 A toy binary classification task](#2.1-A-toy-binary-classification-task)
  * [2.2 Definition of the Keras classification Model](#2.2-Definition-of-the-Keras-classification-Model)
  * [2.3 Defining and training the neural net classifier](#2.3-Defining-and-training-the-neural-net-classifier)
  * [2.4 Making predictions, classification](#2.4-Making-predictions-classification)
* [3 Training a regressor](#3.-Training-a-regressor)
  * [3.1 A toy regression task](#3.1-A-toy-regression-task)
  * [3.2 Definition of the Keras regression Model](#3.2-Definition-of-the-Keras-regression-Model)
  * [3.3 Defining and training the neural net regressor](#3.3-Defining-and-training-the-neural-net-regressor)
  * [3.4 Making predictions, regression](#3.4-Making-predictions-regression)
* [4. Saving and loading a model](#4.-Saving-and-loading-a-model)
  * [4.1 Saving the whole model](#4.1-Saving-the-whole-model)
  * [4.2 Saving using Keras' saving methods](#4.2-Saving-using-Keras-saving-methods)
* [5. Usage with an sklearn Pipeline](#5.-Usage-with-an-sklearn-Pipeline)
* [6. Callbacks](#6.-Callbacks)
* [7. Usage with sklearn GridSearchCV](#7.-Usage-with-sklearn-GridSearchCV)
  * [7.1 Special prefixes](#7.1-Special-prefixes)
  * [7.2 Performing a grid search](#7.2-Performing-a-grid-search)

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

## 2. Training a classifier and making predictions

### 2.1 A toy binary classification task

We load a toy classification task from `sklearn`.

```python
import numpy as np
from sklearn.datasets import make_classification


X, y = make_classification(1000, 20, n_informative=10, random_state=0)

X.shape, y.shape, y.mean()
```

### 2.2 Definition of the Keras classification Model

We define a vanilla neural network with.

Because we are dealing with 2 classes, the output layer can be constructed in
two different ways:

1. Single unit with a `"sigmoid"` nonlinearity. The loss must be `"binary_crossentropy"`.
2. Two units (one for each class) and a `"softmax"` nonlinearity. The loss must be `"sparse_categorical_crossentropy"`.

In this example, we choose the first option, which is what you would usually
do for binary classification. The second option is usually reserved for when
you have >2 classes.

```python
from tensorflow import keras


def get_clf(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    return model
```

### 2.3 Defining and training the neural net classifier

We use `KerasClassifier` because we're dealing with a classifcation task. The first argument should be a callable returning a `Keras.Model`, in this case, `get_clf`. As additional arguments, we pass the number of loss function (required) and the optimizer, but the later is optional. We must also pass all of the arguments to `get_clf` as keyword arguments to `KerasClassifier` if they don't have a default value in `get_clf`. Note that if you do not pass an argument to `KerasClassifier`, it will not be avilable for hyperparameter tuning. Finally, we also pass `random_state=0` for reproducible results.

```python
from scikeras.wrappers import KerasClassifier


clf = KerasClassifier(
    model=get_clf,
    loss="binary_crossentropy",
    hidden_layer_sizes=(100,),
    dropout=0.5,
)
```

As in `sklearn`, we call `fit` passing the input data `X` and the targets `y`.

```python
clf.fit(X, y);
```

Also, as in `sklearn`, you may call `predict` or `predict_proba` on the fitted model.

### 2.4 Making predictions, classification

```python
y_pred = clf.predict(X[:5])
y_pred
```

```python
y_proba = clf.predict_proba(X[:5])
y_proba
```

## 3 Training a regressor

### 3.1 A toy regression task

```python
from sklearn.datasets import make_regression


X_regr, y_regr = make_regression(1000, 20, n_informative=10, random_state=0)

X_regr.shape, y_regr.shape, y_regr.min(), y_regr.max()
```

### 3.2 Definition of the Keras regression Model

Again, define a vanilla neural network. The main difference is that the output layer always has a single unit and does not apply any nonlinearity.

```python
def get_reg(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))
    return model
```

### 3.3 Defining and training the neural net regressor

Training a regressor has nearly the same data flow as training a classifier. The differences include using `KerasRegressor` instead of `KerasClassifier` and adding `KerasRegressor.r_squared` as a metric. Most of the Scikit-learn regressors use the coefficient of determination or R^2 as a metric function, which measures correlation between the true labels and predicted labels.

```python
from scikeras.wrappers import KerasRegressor


reg = KerasRegressor(
    model=get_reg,
    loss="mse",
    metrics=[KerasRegressor.r_squared],
    hidden_layer_sizes=(100,),
    dropout=0.5,
)
```

```python
reg.fit(X_regr, y_regr);
```

### 3.4 Making predictions, regression

You may call `predict` or `predict_proba` on the fitted model. For regressions, both methods return the same value.

```python
y_pred = reg.predict(X_regr[:5])
y_pred
```

## 4. Saving and loading a model

Save and load either the whole model by using pickle, or use Keras' specialized save methods on the `KerasClassifier.model_` or `KerasRegressor.model_` attribute that is created after fitting. You will want to use Keras' model saving utilities if any of the following apply:

1. You wish to save only the weights or only the training configuration of your model.
2. You wish to share your model with collaborators. Pickle is a relatively unsafe protocol and it is not recommended to share or load pickle objects publically.
3. You care about performance, especially if doing in-memory serialization.

For more information, see Keras' [saving documentation](https://www.tensorflow.org/guide/keras/save_and_serialize).

### 4.1 Saving the whole model

```python
import pickle


bytes_model = pickle.dumps(reg)
new_reg = pickle.loads(bytes_model)
new_reg.predict(X_regr[:5])  # model is still trained
```

### 4.2 Saving using Keras' saving methods

This efficiently and safely saves the model to disk, including trained weights.
You should use this method if you plan on sharing your saved models.

```python
# Save to disk
pred_old = reg.predict(X_regr)
reg.model_.save("/tmp/my_model")  # saves just the Keras model
```

```python
# Load the model back into memory
new_reg_model = keras.models.load_model("/tmp/my_model")
# Now we need to instantiate a new SciKeras object
# since we only saved the Keras model
reg_new = KerasRegressor(new_reg_model)
# use initialize to avoid re-fitting
reg_new.initialize(X_regr, y_regr)
pred_new = reg_new.predict(X_regr)
np.testing.assert_allclose(pred_old, pred_new)
```

## 5. Usage with an sklearn Pipeline

It is possible to put the `KerasClassifier` inside an `sklearn Pipeline`, as you would with any `sklearn` classifier.


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


pipe = Pipeline([
    ('scale', StandardScaler()),
    ('clf', clf),
])


y_proba = pipe.fit(X, y).predict(X)
```

To save the whole pipeline, including the Keras model, use `pickle`.

## 6. Callbacks

Adding a new callback to the model is straightforward. Below we define a threashold callback
to avoid training past a certain accuracy. This a rudimentary for of
[early stopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).

```python
class MaxValLoss(keras.callbacks.Callback):

    def __init__(self, monitor: str, threashold: float):
        self.monitor = monitor
        self.threashold = threashold

    def on_epoch_end(self, epoch, logs=None):
        if logs[self.monitor] > self.threashold:
            print("Threashold reached; stopping training") 
            self.model.stop_training = True
```

Define a test dataset:

```python
from sklearn.datasets import make_moons


X, y = make_moons(n_samples=100, noise=0.2, random_state=0)
```

And try fitting it with and without the callback:

```python
kwargs = dict(
    model=get_clf,
    loss="binary_crossentropy",
    dropout=0.5,
    hidden_layer_sizes=(100,),
    metrics=["binary_accuracy"],
    fit__validation_split=0.2,
    epochs=20,
    verbose=False,
    random_state=0
)

# First test without the callback
clf = KerasClassifier(**kwargs)
clf.fit(X, y)
print(f"Trained {len(clf.history_['loss'])} epochs")
print(f"Final accuracy: {clf.history_['val_binary_accuracy'][-1]}")  # get last value of last fit/partial_fit call
```

And with:

```python
# Test with the callback

cb = MaxValLoss(monitor="val_binary_accuracy", threashold=0.75)

clf = KerasClassifier(
    **kwargs,
    callbacks=[cb]
)
clf.fit(X, y)
print(f"Trained {len(clf.history_['loss'])} epochs")
print(f"Final accuracy: {clf.history_['val_binary_accuracy'][-1]}")  # get last value of last fit/partial_fit call
```

For information on how to write custom callbacks, have a look at the
[Advanced Usage](https://nbviewer.jupyter.org/github/adriangb/scikeras/blob/master/notebooks/Advanced_Usage.ipynb) notebook.

## 7. Usage with sklearn GridSearchCV

### 7.1 Special prefixes

SciKeras allows to direct access to all parameters passed to the wrapper constructors, including deeply nested routed parameters. This allows tunning of
paramters like `hidden_layer_sizes` as well as `optimizer__learning_rate`.

This is exactly the same logic that allows to access estimator parameters in `sklearn Pipeline`s and `FeatureUnion`s.

This feature is useful in several ways. For one, it allows to set those parameters in the model definition. Furthermore, it allows you to set parameters in an `sklearn GridSearchCV` as shown below.

To differentiate paramters like `callbacks` which are accepted by both `tf.keras.Model.fit` and `tf.keras.Model.predict` you can add a `fit__` or `predict__` routing suffix respectively. Similar, the `model__` prefix may be used to specify that a paramter is destined only for `get_clf`/`get_reg` (or whatever callable you pass as your `model` argument).

For more information on parameter routing with special prefixes, see the [Advanced Usage Docs](https://www.adriangb.com/scikeras/stable/advanced.html#routed-parameters)

### 7.2 Performing a grid search

Below we show how to perform a grid search over the learning rate (`optimizer__lr`), the model's number of hidden layers (`model__hidden_layer_sizes`), the model's dropout rate (`model__dropout`).

```python
from sklearn.model_selection import GridSearchCV


clf = KerasClassifier(
    model=get_clf,
    loss="binary_crossentropy",
    optimizer="adam",
    optimizer__lr=0.1,
    model__hidden_layer_sizes=(100,),
    model__dropout=0.5,
    verbose=False,
)
```

*Note*: We set the verbosity level to zero (`verbose=False`) to prevent too much print output from being shown.

```python
params = {
    'optimizer__lr': [0.05, 0.1],
    'model__hidden_layer_sizes': [(100, ), (50, 50, )],
    'model__dropout': [0, 0.5],
}

gs = GridSearchCV(clf, params, scoring='accuracy', n_jobs=-1, verbose=True)

gs.fit(X, y)

print(gs.best_score_, gs.best_params_)
```

Of course, we could further nest the `KerasClassifier` within an `sklearn.pipeline.Pipeline`,
in which case we just prefix the parameter by the name of the net (e.g. `clf__model__hidden_layer_sizes`).
