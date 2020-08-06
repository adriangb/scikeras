# Scikit-Learn Wrapper for Keras


[![build status](https://secure.travis-ci.org/adriangb/scikeras.png?branch=master)](https://travis-ci.org/github/adriangb/scikeras) [![Coverage Status](https://codecov.io/gh/adriangb/scikeras/branch/master/graph/badge.svg)](https://codecov.io/gh/adriangb/scikeras)


The goal of this project is to provide wrappers for Keras models so that they can be used as part of a `Scikit-Learn` workflow. These wrappers seeek to emulate the base classes found in `sklearn.base`. 

This project was originally part of Keras itself, but to simplify maintenence and implementation it is now hosted in this repository.

Learn more about the [`Scikit-Learn` API](https://scikit-learn.org/stable/modules/classes.html).

Learn more about [Keras](https://www.tensorflow.org/guide/keras), TensorFlow's Python API.

Python versions supported: the maximum overlap between `Scikit-Learn` and `TensorFlow`. Currently, this means Python >=3.6 and <=3.8.

## Installation
This package is available on PyPi:
```
pip install scikeras
```

The only dependencies are `scikit-learn>=0.22` and `TensorFlow>=2.1.0`.

## Wrapper Classes

### `BaseWrapper`
Base implementation that wraps Keras models for use with `Scikit-Learn` workflows. Inherit from this wrapper to build other types of estimators, for example a [Transformer](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html). Refer to `KerasClassifier` and `KerasRegressor` for inspiration.


### `KerasClassifier`
Implements the `Scikit-Learn` classifier interface, akin to `sklearn.base.ClassifierMixin`. By default, scoring is done using `sklearn.metrics.accuracy_score`.

### `KerasRegressor`
Implements the `Scikit-Learn` classifier interface, akin to `sklearn.base.RegressorMixin`. By default, scoring is done using `sklearn.metrics.r2_score`. Note that `Keras` does *not* have R2 as a built in loss function. A basic implementation of a `Keras` compatible R2 loss funciton is provided in `KerasRegressor.root_mean_squared_error`.

## Basic Usage

To use the wrappers, you must specify how to build the `Keras` model. The wrappers support both [`Sequential`](https://www.tensorflow.org/guide/keras/overview) and [`Functional API`](https://www.tensorflow.org/guide/keras/functional) models. There are 2 basic options to specify how the model is built:
1. Prebuilt model: pass an existing `Keras` model object to the wrapper, which will be copied to avoid modifying the existing model. You must pass the prebuilt model via the `build_fn` parameter when initializing the wrapper.
2. Dynamically built model: pass a function that returns a model object. The model will not be built until `fit` is called. More details below.

### Prebuilt models
Example usage:
```python3
from scikeras.wrappers import KerasRegressor
from some_module import keras_model_object

estimator = KerasRegressor(build_fn=keras_model_object)

estimator.fit(X, y)
estimator.score(X, y)
```


### Dynamically built models
There are 2 ways to specify a model building function for dynamically built models:
1. Pass a callable function or an instance of a class implementing `__call__` as the `build_fn` parameter.
2. Subclass the wrapper and implement `__call__` in your class.
   
The logic for selecting which method to use is in `BaseWrapper._check_build_fn`. For either method, the ultimate function used to build the model is stored by reference in `BaseWrapper.__call__`. From now on this will be refered to as `model building function`.

The signature of the model building function will be used to dynamically determine which parameters should be passed. Parameters are chosen from the arguments of `fit` and from the public parameters of the wrapper instance (ex: `n_classes_` or `n_outputs_`). For example, to create a Multi Layer Perceptron model that is able to dynamically set the input and output sizes as well as hidden layer sizes, you would add `X` and `n_outputs_` to your model building function's signature:

```python3
from scikeras.wrappers import KerasRegressor


def model_building_function(X, n_outputs_, hidden_layer_sizes):
    """Dynamically build regressor."""
    model = Sequential()
    model.add(Dense(X.shape[1], activation="relu", input_shape=X.shape[1:]))
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation="relu"))
    model.add(Dense(n_outputs_))
    model.compile("adam", loss="mean_squared_error")
    return model

estimator = KerasRegressor(build_fn=model_building_function, hidden_layer_sizes=[200, 100])

estimator.fit(X, y)
estimator.score(X, y)
```

Note that in this example, hidden_layer_sizes was specified as an keyword argument to `Keras Regressor`. The value of this keyword argument will be stored as `estimator.hidden_layer_sizes` and will be available to the `Scikit-Learn` API for use with `sklearn.model_selection.GridSearchCV` and other hyperparameter tuning methods. Because `hidden_layer_sizes` is a public attribute of `estimator` as well as an argument to `model_build_function`, it's value will be passed to `model_build_function` when `fit` is called.

Also note that the input itself, `X` was passed, allowing the input shape/size to be dynamically determined when `fit` is called. Finally, `n_outputs_` is generated by `KerasRegressor` and `KerasClassifier` when `fit` is called and is stored as a public attribute in `estimator.n_outputs_`. Because this is a public attribute of `estimator`, `model_build_function` can request it simply by having a parameter with that name.

The model parameters generated while fitting that are used by various parts of the `Scikit-Learn` API are:
* `n_outputs_`: number of outputs. For regression, this is always `y.shape[1]`.
* `n_classes_`: The number of classes (for single output problems), or a list containing the number of classes for each output (for multi-output problems).
* `classes_`: The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).

### Subclassing wrappers
It may be convenient to subclass a wrapper to hardcode keyword arguments and defaults. In general, this is more compatible with the `Scikit-Learn` API. If the class also implements the model building function as `__call__`, this becaome a self-contained estimator that is fully compatible with the `Scikit-Learn` API. A brief example:

```python3
from scikeras.wrappers import KerasRegressor


class MLPRegressor(KerasRegressor):

    def __init__(self, hidden_layer_sizes=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        super().__init__()   # this is very important!

    def __call__(self, X, n_outputs_, hidden_layer_sizes):
        """Dynamically build regressor."""
        if hidden_layer_sizes is None:
            hidden_layer_sizes = (100, )
        model = Sequential()
        model.add(Dense(X.shape[1], activation="relu", input_shape=X.shape[1:]))
        for size in hidden_layer_sizes:
            model.add(Dense(size, activation="relu"))
        model.add(Dense(n_outputs_))
        model.compile("adam", loss=KerasRegressor.root_mean_squared_error)
        return model
```

A couple of notes:
1. It is very important to call `super().__init__()` to properly register kwargs and the model building function.
2. You must assign all parameters to a public attribute of the same name and should *not* change it's value. To change the value from a default you can either (1) change the value in the model building function (as above) or save the parameter under another name (ex: `_hidden_layer_sizes`, remember to also use this name in the model building function).
3. You should set a default for all tunable arguments (in this case, `hidden_layer_sizes=None`) as this is expected by the `Scikit-Learn` API.
4. In the example above, no kwargs are accepted, and none are passed on. You may choose to accept and pass on keyword arguments. Once the `__init__` method resolution reaches `BaseWrapper`, any keyword arguments that have not been consumed by child classes will be saved as instance attributes and will be accessible to the `Scikit-Learn` API. For example:

```python3
from scikeras.wrappers import KerasRegressor


class MLPRegressor(KerasRegressor):

    def __init__(self, hidden_layer_sizes=None, **kwargs):
        self.hidden_layer_sizes = hidden_layer_sizes
        super().__init__(**kwargs)   # this is very important!

    def __call__(self, X, n_outputs_, hidden_layer_sizes):
        ...

estimator = MLPRegressor(hidden_layer_sizes=[200], a_kwarg="saveme")

estimator.a_kwarg == "saveme"  # True
```

This interface allows for multiple layers of inheritence and consumption of arguments:

```python3
from scikeras.wrappers import KerasRegressor


class ChildMLPRegressor(MLPRegressor):

    def __init__(self, child_argument=None, **kwargs):
        self.child_argument = child_argument
        super().__init__(**kwargs)   # this is very important!

    def __call__(self, X, n_outputs_, hidden_layer_sizes):
        ...

estimator = ChildMLPRegressor(child_argument="hello", a_kwarg="saveme")

estimator.child_argument == "hello"  # True
estimator.a_kwarg == "saveme"  # True
estimator.hidden_layer_sizes == None  # True, the default in MLPRegressor is used
```

As long as `super()` is called from the child classes all the way up to `BaseEstimator`, all arguments will be registered for the `Scikit-Learn` API to use.
### Passing arguments to Keras methods

This section refers to passing arguments to `fit`, `predict`, `predict_proba`, and `score` methods of `Keras` models (e.g., `epochs`, `batch_size`),

There are 2 ways to pass these argumements:
1. Pass directly to `KerasClassifier.fit` or `KerasRegressor.fit` (or `score`, etc.).
2. Pass as a keyword argument when initalizing `KerasClassifier` or `KerasRegressor`. This will allow the parameter to be tunable by the `Scikit-Learn` hyperparameter tuning API (`GridSearchCV` or `RandomizedSearchCV`).

## Advanced Usage

### Multi-output problems
`Scikit-Learn` supports a limited number of multi-ouput problems and does not support any multi-input problems. See [`Multiclass and multilabel algorithms`](https://scikit-learn.org/stable/modules/multiclass.html) for more details.

These wrappers suppport all of the multi-output types that `Scikit-Learn` supports out of the box. So for example, you can create a model that has multiple `sigmoid` output layers, resulting in a multiple binary classification problem. This type of problem is denoted `multilabel-indicator` in `Scikit-Learn`. Another example is a model with multiple `softmax` outputs, resulting in what is known as a `multiouput-multiclass` classification. There are many ways to pair up a `Keras` model with a `Scikit-Learn` output type, summarized below:

|                                        | Number of targets | Target cardinality | Valid `type_of_target`   | Keras Output Mode | Keras Output Type |
|----------------------------------------|-------------------|--------------------|--------------------------|-------------------|------------------|
| Multiclass classification              | 1                 | >2                 | ‘multiclass'             | Single softmax    | Numpy array      |
| Multilabel classification              | >1                | 2 (0 or 1)         | 'multilabel-indicator'   | Multiple sigmoid  | List of arrays   |
| Multioutput regression                 | >1                | Continuous         | 'continuous-multioutput' | Single            | Single array     |
| Multioutput- multiclass classification | >1                | >2                 | 'multiclass-multioutput' | Multiple softmax  | List of arrays   |

This table mirrors the [`Scikit-Learn` multi-output documentation](https://scikit-learn.org/stable/modules/multiclass.html).

As noted above, `Keras` returns a list of arrays in many cases. This list is joined back into a single array by `_post_process_y`.

#### Output pre-processing
Conversion from `Scikit-Learn` formatted `y` and `Keras` formatted `y` are done in the wrappers `_pre_process_y` and `_post_process_y` methods. The signatures are:

#### `_pre_process_y`
Signature: `_pre_process_y(y: np.array) -> np.array, dict`
Inputs:
* `y` always a single `numpy.array`
Outputs:
* `y`: a single `numpy.array` for a single `Keras` output or a list of `numpy.array` for a `Keras` model with multiple outputs
*  `extra_args`: a dictionary containing extra parameters determined within `_pre_process_y` such as `classes_`. If used within `fit`, these parameters will overwrite instance parameters of the same name.

#### `_post_process_y`
Signature: `_post_process_y(y: np.array) -> np.array, dict`
Inputs:
* `y` raw output form the `Keras` model's `predict` method, can be an array or list of arrays.
Outputs:
* `y`: a single `numpy.array`. For classificaiton, this should contain class predictions.
*  `extra_args`: for regression, this parameter is unused. For classification, this parameter contains prediction probabilities under the key `class_probabilities`.

To support a custom mapping from `Keras` to `Scikit-Learn`, you can subclass a wrapper and modify `_pre_process_y` and `_post_process_y`. For example, to support a mixed  binary/multiclass classification as a `multioutput-multiclass` problem:

```python3
class FunctionalAPIMultiOutputClassifier(KerasClassifier):
    """Functional API Classifier with 2 outputs of different type.
    """

    def __call__(self, X, n_classes_):
        inp = Input((4,))

        x1 = Dense(100)(inp)

        binary_out = Dense(1, activation="sigmoid")(x1)
        cat_out = Dense(n_classes_[1], activation="softmax")(x1)

        model = Model([inp], [binary_out, cat_out])
        losses = ["binary_crossentropy", "categorical_crossentropy"]
        model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

        return model

    def _post_process_y(self, y):
        """To support targets of different type, we need to post-precess each one
           manually, there is no way to determine the types accurately.

           Takes KerasClassifier._post_process_y as a starting point and
           hardcodes the post-processing.
        """
        classes_ = self.classes_

        class_predictions = [
            classes_[0][np.where(y[0] > 0.5, 1, 0)],
            classes_[1][np.argmax(y[1], axis=1)],
        ]

        class_probabilities = np.squeeze(np.column_stack(y))

        y = np.squeeze(np.column_stack(class_predictions))

        extra_args = {"class_probabilities": class_probabilities}

        return y, extra_args

    def score(self, X, y):
        """Taken from sklearn.multiouput.MultiOutputClassifier
        """
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))
```

The default implementation of `_pre_process_y` for `KerasClassifier` attempts to automatically determine the type of problem using `sklearn.utils.multiclass.type_of_target`. You may need to override this method if it is unable to determine the correct type for your data. The default implementation is provided as a static method so that you can test it without needing to instantiate a `KerasClassifier`.

### Multi-input problems

As mentioned above, `Scikit-Learn` does not support multi-input problems since `X` must be a sinlge `numpy.array`. However, in order to extend this functionality, the wrappers provide a `_pre_process_X` method that allows mapping a single `numpy.arary` to a list of `numpy.array` for multi-input `Keras` models. For example:

```python3
class FunctionalAPIMultiInputClassifier(KerasClassifier):
    """Functional API Classifier with 2 inputs.
    """

    def __call__(self, n_classes_):
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

    @staticmethod  # _pre_process_X does not *need* to be a static method
    def _pre_process_X(X):
        """To support multiple inputs, a custom method must be defined.
        """
        return [X[:, 0], X[:, 1:4]], dict()
```

Note that similar to `_pre_process_y`, `_pre_process_X` returns the modified `X` along with a dictionary of extra parameters. This dictionary is currently unused, but is kept for symmetry with `_pre_process_x` and future flexibility.

### Custom scorers
To override the function used for scoring, set the `_scorer` attribute of the wrapper to point to a scoring function with the signature `scorer(y_true: np.array, y_pred: np.array) -> float`.

### Callbacks
The wrappers fully support `Keras` Callbacks. For general information on Callbacks, see the [`TensorFlow` documenation](https://www.tensorflow.org/guide/keras/custom_callback).

Like any other parameter, callbacks will be passed to the model building function as long as the parameter and attribute names match. Callbacks may be hardcoded in a subclassed model, passed as a parameter to `fit`, passed as a parameter to the wrapper initializer or created dynamically during wrapper initialization.

Here is a simple example of a custom callback in use:

```python3
class SentinalCallback(keras.callbacks.Callback):
    """
    Callback class that sets an internal value once it's been acessed.
    """

    called = 0

    def __init__(self, tolerance):
        # we don't do anything with this, just an example
        self.tolerance = tolerance


    def on_train_begin(self, logs=None):
        """Increments counter."""
        self.called += 1


class ClassifierWithCallback(wrappers.KerasClassifier):
    """Must be defined at top level to be picklable.
    """

    def __init__(self, tolerance, hidden_dim=None):
        self.callbacks = [SentinalCallback(tolerance)]
        self.hidden_dim = hidden_dim
        super().__init__()

    def __call__(self, hidden_dim):
        return build_fn_clf(hidden_dim)
```

In this simple case, the callback is dynamically created during wrapper initialization. This allows great flexibility. In this example, the parameter `tolerance` can be tuned using `Scikit-Learn` hyperparameter tuning tools.

### Model serialization
Several functions within `Scikit-Learn` require estimators to be picklable. In order to enable this, a combination of architechrure, weights and training config serialization are used. At the moment, whole model serialization can only be done to/from file, which is not appropriate for this use case. For more information on saving Keras models see the TensorFlow documentation.

In order to support model serialization, the wrappers `__getstate__` and `__setstate__` methods detect TensorFlow objects as long as they are attributes of the estimator or nested inside simple iterables (lists, tuples), dictionaries or simple classes. More complex nesting may cause failures.

The important thing is that **models subclassed from `tensorflow.keras.Model` must register themselves as serializable**. The easiest way to achieve this is to use the `tensoflow.keras.utils.register_keras_serializable` decorator. For more information, see the TensoFlow documentation [here](https://www.tensorflow.org/api_docs/python/tf/keras/utils/register_keras_serializable).

### Random states

If the wrappers have a `random_state` parameter (set via `**skparams` or by subclassing), they will
use it _almost_ like ScikitLearn estimators use `random_state`.
The only difference is that TensorFlow requires setting the random seed system-wide,
so there may be side effects, for example if training models simulteneously in a multi-threaded
enviroment.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
