Wrappers
========

Using SciKeras Wrappers
-----------------------

The main entry points for SciKeras users are
:py`scikeras.wrappers.KerasClassifier`{.interpreted-text role="class"},
:py`scikeras.wrappers.KerasRegressor`{.interpreted-text role="class"}
and :py`scikeras.wrappers.BaseWrapper`{.interpreted-text role="class"}.
`BaseWrapper` provides general `Keras` wrapping functionality and
`KerasClassifier` and `KerasRegressor` extend this with functionality
specific to classifiers and regressors respectively. This document
focuses on the overall functionality of the wrappers and hence will
refer to :py`scikeras.wrappers.BaseWrapper`{.interpreted-text
role="class"} as a proxy for all of the wrapper classes. Detailed
information on usage of specific classes is available in the
`scikeras-api`{.interpreted-text role="ref"} documentation.

The SciKeras wraps the Keras
:py`~tensorflow.keras.Model`{.interpreted-text role="class"} while
providing an interface that should be familiar for sklearn users.

To get started, define your
:py`~tensorflow.keras.Model`{.interpreted-text role="class"}
architecture like you always do, but within a callable top-level
function (we will call this function `model_build_fn` for the remained
of these docs, but you are free to name it as you wish). Then pass this
function to :py`.BaseWrapper`{.interpreted-text role="class"} in the
`model` parameter. Finally, you can call
:py`~scikeras.wrappers.BaseWrapper.fit`{.interpreted-text role="func"}
and :py`~scikeras.wrappers.BaseWrapper.predict`{.interpreted-text
role="func"}, as with an sklearn estimator. The finished code could look
something like this:

``` {.python}
def model_build_fn():
     model = Model(...)
     ...
     return model

clf = KerasClassifier(model=model_build_fn)
clf.fit(X, y)
y_pred = clf.predict(X_valid)
```

Let\'s see what SciKeras did:

-   wraps `tensorflow.keras.Model` in an sklearn interface
-   handles encoding and decoding of the target `y`
-   compiles the :py`~tensorflow.keras.Model`{.interpreted-text
    role="class"} (unless you do it yourself in `model_build_fn`)
-   makes all `Keras` objects serializable so that they can be used with
    :py`~sklearn.model_selection`{.interpreted-text role="mod"}.

SciKeras abstracts away the incompatibilities and data conversions,
letting you focus on defining your architecture and choosing your
hyperparameters. At the same time, SciKeras is very flexible and can be
extended with ease, getting out of your way as much as possible.

### Initialization

When you instantiate the :py`.KerasClassifier`{.interpreted-text
role="class"} or :py`.KerasRegressor`{.interpreted-text role="class"}
instance, only the given arguments are stored. These arguments are
stored unmodified. For instance, the `model` will remain uninstantiated.
This is to make sure that the arguments you pass are not touched
afterwards, which makes it possible to clone the wrapper instance, for
example in a
:py`~sklearn.model_selection.GridSearchCV`{.interpreted-text
role="class"}.

Only when the
:py`~scikeras.wrappers.KerasClassifier.fit`{.interpreted-text
role="func"} or
:py`~scikeras.wrappers.KerasRegressor.fit`{.interpreted-text
role="func"} method are called, are the different attributes of the
wrapper, such as the `model`, initialized. An initialized attribute\'s
name always ends on an underscore; e.g., the initialized `module` is
called `model_`. (This is the same nomenclature as sklearn uses.)
Therefore, you always know which attributes you set and which ones were
created by the wrappers.

Once initialized by calling `fit`, the wrappers create several
attributes, documented in the `scikeras-api`{.interpreted-text
role="ref"} documentation.

### Compilation of `Model`

You have two options to compile your model: 1. Compile your model within
`model_build_fn` and return this compiled model. In this case, SciKeras
will not re-compile your model and all compilation parameters (such as
`optimizer`) given to
:py`scikeras.wrappers.BaseWrapper.__init__`{.interpreted-text
role="func"} will be ignored. 2. Return an uncompiled model from
`model_build_fn` and let SciKeras handle the compilation. In this case,
SciKeras will apply all of the compilation parameters, including
instantiating losses, metrics and optimizers.

The first route will be more flexible if you wish to determine how to
compile your `Model` within the same function in which you define it.
The latter will offer an easy way to compile and tune compilation
parameters. Examples:

``` {.python}
def model_build_fn(compile_kwargs):
    # you can access the ``optimizer`` param here
    optimizer = compile_kwargs["optimizer"]
    if optimizer is None:
        # and apply any custom logic you wish
        ...
    model = Model(...)
    ...
    model.compile(optimizer=optimizer)
    return model

clf = KerasClassifier(model=model_build_fn)
clf.fit(X, y)
y_pred = clf.predict(X_valid)
```

``` {.python}
from tensorflow.keras.optimizers import Adam

def model_build_fn():
    model = Model(...)
    ...
    # Do not call model.compile
    return model  # That's it, SciKeras will compile your model for you

clf = KerasClassifier(model=model_build_fn, optimizer=Adam)
clf.fit(X, y)
y_pred = clf.predict(X_valid)
```

### Arguments to `model`/`model_build_fn`

You probably wish to pass parameters from
:py`~scikeras.wrappers.BaseWrapper`{.interpreted-text role="class"} to
`model`, or you may want to use attributes from
:py`~scikeras.wrappers.BaseWrapper`{.interpreted-text role="class"} such
as `n_features_in_` while building your model. SciKeras allows you to do
both.

To enable this, SciKeras uses two special arguments to `model` that will
only be passed if they are present in `model`\'s signature (i.e. there
is an argument with the same name in `model`\'s signature):

#### `meta_params`

This is a dictionary containing all of the attributes that
:py`~scikeras.wrappers.BaseWrapper`{.interpreted-text role="class"}
creates when it is initialized These include `n_features_in_`,
`y_dtype_`, etc. For a full list, see the
`scikeras-api`{.interpreted-text role="ref"} documentation.

#### `compile_kwargs`

This is a dictionary of parameters destined for
:py`tensorflow.Keras.Model.compile`{.interpreted-text role="func"}. You
will want to accept this parameter unless you are returning an
un-compiled `Model` instance. Parameters available via this dictionary
are:

-   `optimizer`
-   `loss`
-   `callbacks`
-   Any other parameters with the prefix `optimizer__`, `loss__` or
    `callbacks__`

#### Keyword arguments to :py`scikeras.wrappers.BaseWrapper.__init__`{.interpreted-text role="func"}

Any **other** keyword arguments passed to
:py`scikeras.wrappers.BaseWrapper.__init__`{.interpreted-text
role="func"} when the wrapper is instantiated will also be passed to
`model_build_fn` as keyword arguments. For example, calling
`KerasClassifier(myparam=10)` will result in a
`model_build_fn(my_param=10)` call.

Arguments to :py`scikeras.wrappers.BaseWrapper`{.interpreted-text role="class"}
-------------------------------------------------------------------------------

A complete explanation of all arguments and methods of
:py`.BaseWrapper`{.interpreted-text role="class"} are found in the
`scikeras-api`{.interpreted-text role="ref"} documentation. Here we
focus on the main ones.

### model

This is where you pass your Keras `Model`. Ideally, it should not be
instantiated, although instantiated models are accepted. Instead, the
init arguments for your model should be passed to
:py`.BaseWrapper`{.interpreted-text role="class"} either as bare keyword
arguments (if their name does not conflict with any of the existing
keyword arguments) or with the `model__` prefix, which will override any
arguments of the same name only when passed to `model`. E.g., if your
module takes the arguments `hidden_layer_sizes` and `lr`, the code would
look like this:

``` {.python}
def model_build_fn(hidden_layer_sizes, lr):
    model = Model(...)
    ...
    return model

clf = KerasClassifier(
    model=model_build_fn,
    hidden_layer_sizes=(100,),
    model__lr=0.5,  # also equivalent to just lr=0.5 in this case
)
```

Note that SciKeras automatically interprets the type of classification
task when using `KerasClassifier`, as determined by
:py`~sklearn..utils.multiclass.type_of_target`{.interpreted-text
role="func"}. This means that if you pass a binary target, you need to
define a your `Model`\'s output layer to apply a `sigmoid` nonlinearity
to get good results.

### random_state

This behaves similar to the same parameter in `sklearn` estimators. If
set to an integer or a :py`~numpy.random.RandomState`{.interpreted-text
role="class"} instance, it will be used to seed the random number
generators used to initialize the graph and optimizers. Note that use of
this parameter may have unforeseen consequences since `TensorFlow` only
has a *global* random state.

### optimizer

Like :py`~tensorflow.keras.Model`{.interpreted-text role="class"}, this
can be a string or a class from
:py`~tensorflow.keras.optimizers`{.interpreted-text role="mod"}. Unlike
:py`~tensorflow.keras.Model`{.interpreted-text role="class"}, if you
pass a class it is preferable that you pass an un-instantiated class and
pass it\'s arguments using `param-routing`{.interpreted-text
role="ref"}.

### batch_size

This argument is passed to
:py`~tensorflow.keras.Model.fit`{.interpreted-text role="func"}. See the
the [Keras Model
docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model) for
more details.

### validation_split

This argument is passed to
:py`~tensorflow.keras.Model.fit`{.interpreted-text role="func"}. See the
the [Keras Model
docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model) for
more details.

### callbacks

A single instantiated or uninstantiated callback, a list of instantiated
or uninstantiated callbacks (can be mixed). If using instantiated
callbacks, SciKeras will pass them directly to `Model.compile`. If using
a list or dict of uninstantiated callbacks, SciKeras will instantiate
them using any parameters routed like `callbacks__n__param_name=2` where
this will result in `callbacks[n](param_name=2)` being called from
:py`~scikeras.wrappers.BaseWrapper.compile_model`{.interpreted-text
role="func"}. For a single callback, you would use
`callbacks__param_name=2`. For more information on Keras callbacks, see
the the [Keras Callbacks
docs](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) for
more details. These callbacks are only used if an uncompiled `Model` is
returned from `model_build_fn`.

### metrics

Similar to optimizers, this can be a single instantiated or
uninstantiated Keras metric or a list of instantiated or uninstantiated
metrics. Uninstantiated metrics can be refer to by class or string
shorthand. Similar parameter routing rules to
`callbacks`{.interpreted-text role="ref"} apply. See the the [Keras
Metrics
docs](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) for
more details on using metrics.

### warm_start

This argument determines whether each
:py`~scikeras.wrappers.BaseWrapper.fit`{.interpreted-text role="func"}
call leads to a re-initialization of the
:py`~scikeras.wrappers.BaseWrapper`{.interpreted-text role="class"} or
not. By default, when calling
:py`~scikeras.wrappers.BaseWrapper.fit`{.interpreted-text role="func"},
the parameters of the net are initialized, so your previous training
progress is lost (consistent with the sklearn `fit()` calls). In
contrast, with `warm_start=True`, each
:py`~scikeras.wrappers.BaseWrapper.fit`{.interpreted-text role="func"}
call will continue from the most recent state.

### verbose

`False` disables the progress bar and other logging while `True` enables
it. This argument is passed to multiple methods of
:py`~tensorflow.keras.Model`{.interpreted-text role="class"}. To set
different values for `fit` and `predict` for example, you can use
`fit__verbose=True` and `predict__verbose=False` or `verbose=True` and
`predict__verbose=False` which would have the same effect since the non
routed value from `verbose=True` would be passed to `fit`.

### shuffle

This argument is passed to
:py`~tensorflow.keras.Model.fit`{.interpreted-text role="func"}. See the
the [Keras Model
docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model) for
more details.

### run_eagerly

This argument is passed to
:py`~tensorflow.keras.Model.fit`{.interpreted-text role="func"}. See the
the [Keras Model
docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model) for
more details.

Methods of :py`scikeras.wrappers.BaseWrapper`{.interpreted-text role="class"}
-----------------------------------------------------------------------------

### fit(X, y, sample_weights=None)

This is one of the main methods you will use. It contains everything
required to train the model, be it batching of the data, triggering the
callbacks, or handling the internal validation set.

In general, we assume there to be an `X` and a `y`. And if your task
does not have an actual `y`, you may pass `y=None`.

`X` and `y` are expected to be array-like. SciKeras does not currently
support :py`tensorflow.data.Dataset`{.interpreted-text role="class"}
inputs.

In addition to :py`scikeras.wrapper.BaseWrapper.fit`{.interpreted-text
role="func"}, there is also the
:py`scikeras.wrapper.BaseWrapper.partial_fit`{.interpreted-text
role="func"} method, known from some sklearn estimators.
:py`scikeras.wrapper.BaseWrapper.partial_fit`{.interpreted-text
role="func"} allows you to continue training from your current status,
even if you set `warm_start=False`. A further use case for
:py`scikeras.wrapper.BaseWrapper.partial_fit`{.interpreted-text
role="func"} is when your data does not fit into memory and you thus
need to have several training steps.

### predict(X) and predict_proba(X)

These methods use :py`tensorflow.keras.Model.predict`{.interpreted-text
role="func"} to predict `y` or `y`\'s probabilities based on `X`.
Outputs are cast to numpy arrays of the same dtype and shape as the
input. If :py`tensorflow.keras.Model.predict`{.interpreted-text
role="func"} returns multiple outputs as a list, these are
column-stacked into a single array. This allows the use of simple
multi-output models without any custom logic or intervention. For more
complex cases, you will need to subclass
:py`scikeras.wrappers.BaseWrapper`{.interpreted-text role="class"} and
override the
:py`scikeras.wrappers.BaseWrapper.postprocess_y`{.interpreted-text
role="func"} method.

In case of :py`scikeras.wrappers.KerasClassifier`{.interpreted-text
role="class"}, when
:py`scikeras.wrappers.KerasClassifier.fit`{.interpreted-text
role="func"} is called SciKeras uses the target type (as determined by
:py`~sklearn..utils.multiclass.type_of_target`{.interpreted-text
role="func"}), the loss function used to compile
:py`tensorflow.keras.Model`{.interpreted-text role="class"} and the
number of outputs from :py`tensorflow.keras.Model`{.interpreted-text
role="class"} to automatically determine what encodings and
transformations are necessary.
:py`scikeras.wrappers.KerasClassifier.predict`{.interpreted-text
role="func"} also reverses this encoding to return class labels. On the
other hand,
:py`scikeras.wrappers.KerasClassifier.predict_proba`{.interpreted-text
role="func"} returns the raw class probabilities.

### score(X, y)

This method returns the mean accuracy on the given data and labels for
classifiers and the coefficient of determination R\^2 of the prediction
for regressors. All wrappers rely on the abstract method
:py`scikeras.wrappers.BaseWrapper._scorer`{.interpreted-text
role="func"} with the signature
`_scorer(y_true, y_pred, sample_weights)` to do the scoring. If you want
to swap in an alternative scorer (or implement a scorer in the case of
:py`scikeras.wrappers.BaseWrapper`{.interpreted-text role="class"}) all
you have to do is implement this method.

### Multiple inputs or outputs

In some cases, the input actually consists of multiple inputs. E.g., in
a text classification task, you might have an array that contains the
integers representing the tokens for each sample, and another array
containing the number of tokens of each sample. skorch has you covered
here as well.

Scikit-Learn natively supports multiple outputs, although it technically
requires them to be arrays of equal length (see docs for Scikit-Learn\'s
:py`~sklearn.multioutput.MultiOutputClassifier`{.interpreted-text
role="class"}). Scikit-Learn has no support for multiple inputs. To work
around this issue, SciKeras implements a data conversion abstraction in
the form of `preprocess_{X,y}` and `postprocess_{X, y}`. Within these
methods, you may split a single input `X` into multiple inputs for
:py`tensorflow.keras.Model`{.interpreted-text role="class"} or perform
any other manipulation you need.

This said, note that if you are trying to use outputs of uneven length
or other more complex scenarios, SciKeras may be able to handle them but
the rest of the Scikit-Learn ecosystem likely will not.

Below is an example:

``` {.python}
X = [[1, 2], ["a", "b", "c"]]  # multiple inputs of different lengths
y = np.array([[1, 0, 1], ["apple", "orange", "apple"]]  # a mix of output types

def model_build_fn(meta_params):
    my_n_classes_ = meta_params["my_n_classes_"]
    inp1 = Input((1,))
    inp2 = Input((3,))
    x3 = Concatenate(axis=-1)([x1, x2])
    binary_out = Dense(1, activation="sigmoid")(x3)
    cat_out = Dense(my_n_classes_[1], activation="softmax")(x3)
    model = Model([inp], [binary_out, cat_out])
    return model

class MyWrapper(KerasClassifier):

        def preprocess_y(self, y):
            extra_args = dict()  # this will be used like self.__dict__.update(extra_args)
            my_n_classes_ = [2, np.unique(y[:, 1]).size]
            extra_args["my_n_classes_"] = my_n_classes_
            # split up the targets
            y = [y[:, 0], y[:, 1]]
            return y, extra_args

        def preprocess_X(self, X):
            extra_args = dict()  # this will be used like self.__dict__.update(extra_args)
            # perform some transformation on only one part of the input
            self.input_encoder = OneHotEncoder()
            X[1] = self.input_encoder.fit_transform(X[1])
            return X, extra_args

clf = MyWrapper(
    model=model_build_fn,
    loss=["binary_crossentropy", "categorical_crossentropy"],
    optimizer="adam"
)
clf.fit(X, y)
```

Routed parameters {#param-routing}
-----------------

For more advanced used cases, SciKeras supports Scikit-Learn style
parameter routing to override parameters for individual consumers
(methods or class initializers).

For example, you may want to have multiple callbacks with different
parameters for each.

``` {.python}
from tensorflow.keras.callbacks import BaseLogger, EarlyStopping

clf = KerasClassifier(
    model=model_build_fn,
    loss="binary_crossentropy", "categorical_crossentropy",
    optimizer="sgd",
    metrics="accuracy",
    callbacks=[BaseLogger, EarlyStopping]
    callbacks__0__stateful_metrics="accuracy",
    callbacks__1__patience=2,
)
clf.fit(X, y)
```

The same can be achieved with the special `param_groups` postfix, which
tells SciKeras to expand the list in order:

``` {.python}
from tensorflow.keras.callbacks import BaseLogger, EarlyStopping

clf = KerasClassifier(
    model=model_build_fn,
    loss="binary_crossentropy", "categorical_crossentropy",
    optimizer="sgd",
    metrics="accuracy",
    callbacks=[BaseLogger, EarlyStopping]
    callbacks__param_groups==[
        {"stateful_metrics": "accuracy"},
        {"patience": 2}
    ]
)
clf.fit(X, y)
```

All special prefixes are stored in the `prefixes_` class attribute of
:py`scikeras.wrappers.BaseWrappers`{.interpreted-text role="class"}.
Currently, they are:

-   `model`
-   `fit`
-   `predict`
-   `callbacks`
-   `optimizer`
-   `loss`

All routed parameters will be available for hyperparameter tuning.
