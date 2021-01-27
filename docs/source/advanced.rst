===================================
Advanced Usage of SciKeras Wrappers
===================================

Wrapper Classes
---------------

SciKeras has three wrapper classes avialable to
users: :py:class:`scikeras.wrappers.KerasClassifier`,
:py:class:`scikeras.wrappers.KerasRegressor` and
:py:class:`scikeras.wrappers.BaseWrapper`. ``BaseWrapper`` provides general ``Keras`` wrapping functionality and
``KerasClassifier`` and ``KerasRegressor`` extend this with functionality
specific to classifiers and regressors respectively. Although you will
usually be using either ``KerasClassifier`` and ``KerasRegressor``, this document focuses
on the overall functionality of the wrappers and hence will refer to 
:py:class:`scikeras.wrappers.BaseWrapper` as a proxy for both of the wrapper classes.
Detailed information on usage of specific classes is available in the
:ref:`scikeras-api` documentation.

SciKeras wraps the Keras :py:class:`~tensorflow.keras.Model` to
provide an interface that should be familiar for Scikit-Learn users and is compatible
with most of the Scikit-Learn ecosystem.

To get started, define your :py:class:`~tensorflow.keras.Model` architecture like you always do,
but within a callable top-level function (we will call this function ``model_build_fn`` for
the remained of these docs, but you are free to name it as you wish).
Then pass this function to :py:class:`.BaseWrapper` in the ``model`` parameter.
Finally, you can call :py:func:`~scikeras.wrappers.BaseWrapper.fit`
and :py:func:`~scikeras.wrappers.BaseWrapper.predict`, as with an sklearn
estimator. The finished code could look something like this:

.. code:: python

    def model_build_fn():
         model = Model(...)
         ...
         return model

    clf = KerasClassifier(model=model_build_fn)
    clf.fit(X, y)
    y_pred = clf.predict(X_valid)

Let's see what SciKeras did:

- wraps ``tensorflow.keras.Model`` in an sklearn interface
- handles encoding and decoding of the target ``y``
- compiles the :py:class:`~tensorflow.keras.Model` (unless you do it yourself in ``model_build_fn``)
- makes all ``Keras`` objects serializable so that they can be used with :py:mod:`~sklearn.model_selection`.

SciKeras abstracts away the incompatibilities and data conversions,
letting you focus on defining your architecture and
choosing your hyperparameters.
At the same time, SciKeras is very flexible and can be
extended with ease, getting out of your way as much as possible.

Initialization
^^^^^^^^^^^^^^

When you instantiate the :py:class:`.KerasClassifier` or
:py:class:`.KerasRegressor` instance, only the given arguments are stored.
These arguments are stored unmodified. For instance, the ``model`` will
remain uninstantiated. This is to make sure that the arguments you
pass are not touched afterwards, which makes it possible to clone the
wrapper instance, for example in a :py:class:`~sklearn.model_selection.GridSearchCV`.

Only when the :py:func:`~scikeras.wrappers.BaseWrapper.fit` or
:py:func:`~scikeras.wrappers.BaseWrapper.initialize` methods are called, are the
different attributes of the wrapper, such as ``model_``, initialized.
An initialized attribute's name always ends on an underscore; e.g., the
initialized ``model`` is called ``model_``. (This is the same
nomenclature as sklearn uses.) Therefore, you always know which
attributes you set and which ones were created by the wrappers.

Once initialized by calling ``fit``, the wrappers create several attributes,
documented in the :ref:`scikeras-api` documentation.

Compilation of ``Model``
^^^^^^^^^^^^^^^^^^^^^^^^

You have two options to compile your model:

1. Compile your model within ``model_build_fn`` and return this
compiled model. In this case, SciKeras will not re-compile your model
and all compilation parameters (such as ``optimizer``) given to
:py:func:`scikeras.wrappers.BaseWrapper.__init__` will be ignored.

2. Return an uncompiled model from ``model_build_fn`` and let
SciKeras handle the compilation. In this case, SciKeras will
apply all of the compilation parameters, including instantiating
losses, metrics and optimizers.

The first route will be more flexible if you wish to determine how to compile
your ``Model`` within the same function in which you define it. The latter will
offer an easy way to compile and tune compilation parameters. Examples:

.. code:: python

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

.. code:: python

    from tensorflow.keras.optimizers import Adam

    def model_build_fn():
        model = Model(...)
        ...
        # Do not call model.compile
        return model  # That's it, SciKeras will compile your model for you

    clf = KerasClassifier(model=model_build_fn, optimizer=Adam)
    clf.fit(X, y)
    y_pred = clf.predict(X_valid)


In all cases, returning an un-compiled model is equivalent to
calling ``model.compile(**compile_kwargs)`` within ``model_build_fn``.


Arguments to ``model_build_fn``
-------------------------------

User-defined keyword arguments passed to :py:func:`~scikeras.wrappers.BaseWrapper.__init__`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
All keyword arguments that were given to :py:func:`~scikeras.wrappers.BaseWrapper.__init__`
will be passed to ``model_build_fn`` directly.
For example, calling ``KerasClassifier(myparam=10)`` will result in a
``model_build_fn(my_param=10)`` call.
Note however that ``KerasClassifier(optimizer="sgd")`` will **not** result in
``model_build_fn(optimizer="sgd")``. Instead, you must access ``optimizer`` either
via ``compile_kwargs`` if you want a compiled optimizer
or ``params`` if you want the raw input.

Optional arguments
^^^^^^^^^^^^^^^^^^

You may want to use attributes from
:py:class:`~scikeras.wrappers.BaseWrapper` such as ``n_features_in_`` while building
your model, or you may wish to let SciKeras compile your optimizers and losses
but apply some custom logic on top of that compilation.

To enable this, SciKeras uses three special arguments to ``model`` that will only
be passed if they are present in ``model``'s signature (i.e. there is an argument
with the same name in ``model``'s signature):

``meta``
++++++++
This is a dictionary containing all of the attributes that
:py:class:`~scikeras.wrappers.BaseWrapper` creates when it is initialized
These include ``n_features_in_``, ``y_dtype_``, etc. For a full list,
see the :ref:`scikeras-api` documentation.

``compile_kwargs``
++++++++++++++++++++++++
This is a dictionary of parameters destined for :py:func:`tensorflow.Keras.Model.compile`.
This dictionary can be used like ``model.compile(**compile_kwargs)``.
All optimizers, losses and metrics will be compiled to objects,
even if string shorthands (e.g. ``optimizer="adam"``) were passed.

``params``
++++++++++++++++++++++++
Raw dictionary of parameters passed to :py:func:`~scikeras.wrappers.BaseWrapper.__init__`.
This is basically the same as calling :py:func:`~scikeras.wrappers.BaseWrapper.get_params`.


Data Transformers
^^^^^^^^^^^^^^^^^

Keras supports a much wider range of inputs/outputs than Scikit-Learn does. E.g.,
in a text classification task, you might have an array that contains
the integers representing the tokens for each sample, and another
array containing the number of tokens of each sample.

In order to reconcile Keras' expanded input/output support and Scikit-Learn's more
limited options, SciKeras introduces "data transformers". These are really just
dependency injection points where you can declare custom data transformations,
for example to split an array into a list of arrays, join ``X`` & ``y`` into a ``Dataset``, etc.
In order to keep these transformations in a familiar format, they are implemented as
sklearn-style transformers. You can think of this setup as an sklearn Pipeline:

.. code-block::

                                   ↗ feature_encoder ↘
    SciKeras.fit(features, labels)                    dataset_transformer → keras.Model.fit(data)
                                   ↘ target_encoder  ↗ 


Within SciKeras, this is roughly implemented as follows:

.. code:: python

    class PseudoBaseWrapper:

        def fit(self, X, y, sample_weight):
            self.target_encoder_ = self.target_encoder.fit(X)
            X = self.feature_encoder_.transform(X)
            self.feature_encoder_ = self.feature_encoder.fit(y)
            y = self.target_encoder_.transform(y)
            self.model_ = self._build_keras_model()
            self.dataset_transformer_ = self.dataset_transformer.fit((X, y, sample_weight))
            X, y, sample_weight = self.dataset_transformer_.transform((X, y, sample_weight))
            self.model_.fit(x=X, y=y, sample_weight=sample_weight)  # tf.keras.Model.fit
            return self

        def predict(self, X):
            X = self.feature_encoder_.transform(X)
            X, _, _ = self.dataset_transformer_.fit_transform((X, None, None))
            y_pred = self.model_.predict(X)
            return self.target_encoder_.inverse_transform(y_pred)


``dataset_transformer`` is the last step before passing the data to Keras, and it allows for the greatest
degree of customization because SciKeras does not make any assumptions about the output data
and passes it directly to :py:func:`tensorflow.keras.Model.fit`.
Its signature is:

.. code:: python

    from sklearn.base import BaseEstimator, TransformerMixin

    class DatasetTransformer(BaseEstimator, TransformerMixin):
        def fit(self, data) -> "DatasetTransformer":
            X, y, sample_weight = data  # sample_weight might be None
            ...
            return self

        def transform(self, data):  # return a valid input for keras.Model.fit
            X, y, sample_weight = data  # y and/or sample_weight might be None
            ...
            return (X, y, sample_weight)  # option 1
            return (tensorflow_dataset, None, None)  # option 2


Although you could implement *all* data transformations in a single ``dataset_transformer``,
having several distinct dependency injections points allows for more modularity,
for example to keep the default processing of string-encoded labels but convert
the data to a :py:func:`tensorflow.data.Dataset` before passing to Keras.

For a complete examples implementing custom data processing, see the examples in the
:ref:`tutorials` section.

Multi-input and output models via feature_encoder and target_encoder
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Scikit-Learn natively supports multiple outputs, although it technically
requires them to be arrays of equal length
(see docs for Scikit-Learn's :py:class:`~sklearn.multioutput.MultiOutputClassifier`).
Scikit-Learn has no support for multiple inputs.
To work around this issue, SciKeras implements a data conversion
abstraction in the form of Scikit-Learn style transformers,
one for ``X`` (features) and one for ``y`` (target). These are implemented
via :py:func:`scikeras.wrappers.BaseWrappers.feature_encoder` and
:py:func:`scikeras.wrappers.BaseWrappers.feature_encoder` respectively.

To override the default transformers, simply override
:py:func:`scikeras.wrappers.BaseWrappers.target_encoder` or
:py:func:`scikeras.wrappers.BaseWrappers.feature_encoder` for ``y`` and ``X`` respectively.

By default, SciKeras uses :py:func:`sklearn.utils.multiclass.type_of_target` to categorize the target
type, and implements basic handling of the following cases out of the box:

+--------------------------+--------------+----------------+----------------+---------------+
| type_of_target(y)        | Example y    | No. of Outputs | No. of classes | SciKeras      |
|                          |              |                |                | Supported     |
+==========================+==============+================+================+===============+
| "multiclass"             | [1, 2, 3]    | 1              | >2             | Yes           |
+--------------------------+--------------+----------------+----------------+---------------+
| "binary"                 | [1, 0, 1]    | 1              | 1 or 2         | Yes           |
+--------------------------+--------------+----------------+----------------+---------------+
| "multilabel-indicator"   | [[1, 1],     | 1 or >1        | 2 per target   | Single output |
|                          |              |                |                |               |
|                          | [0, 1],      |                |                | only          |
|                          |              |                |                |               |
|                          | [1, 0]]      |                |                |               |
+--------------------------+--------------+----------------+----------------+---------------+
| "multiclass-multioutput" | [[1, 1],     | >1             | >=2 per target | No            |
|                          |              |                |                |               |
|                          | [3, 2],      |                |                |               |
|                          |              |                |                |               |
|                          | [2, 3]]      |                |                |               |
+--------------------------+--------------+----------------+----------------+---------------+
| "continuous"             | [.1, .3, .9] | 1              | continuous     | Yes           |
+--------------------------+--------------+----------------+----------------+---------------+
| "continuous-multioutput" | [[.1, .1],   | >1             | continuous     | Yes           |
|                          |              |                |                |               |
|                          | [.3, .2],    |                |                |               |
|                          |              |                |                |               |
|                          | [.2, .9]]    |                |                |               |
+--------------------------+--------------+----------------+----------------+---------------+

The supported cases are handled by the default implementation of ``target_encoder``.
The default implementations are available for use as :py:class:`scikeras.utils.transformers.ClassifierLabelEncoder`
and :py:class:`scikeras.utils.transformers.RegressorTargetEncoder` for
:py:class:`scikeras.wrappers.KerasClassifier` and :py:class:`scikeras.wrappers.KerasRegressor` respectively.

As per the table above, if you find that your target is classified as
``"multiclass-multioutput"`` or ``"unknown"``, you will have to implement your own data processing routine.

get_metadata method
+++++++++++++++++++

In addition to converting data, ``feature_encoder`` and ``target_encoder``, allows you to inject data
into your model construction method. This is useful if for example you use ``target_encoder`` to dynamically
determine how many outputs your model should have based on the data and then use this information to
assign the right number of outputs in your Model. To return data from ``feature_encoder`` or ``target_encoder``,
you will need to provide a transformer with a ``get_metadata`` method, which is expected to return a dictionary
which will be injected into your model building function via the ``meta`` parameter.

For example, if you wanted to create a calculated parameter called ``my_param_``:

.. code-block::python

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
    clf.fit(X, y)  # prints 'Got: foobarbaz'
    print(clf.my_param_)  # prints 'foobarbaz'

Note that it is best practice to end your parameter names with a single underscore,
which allows sklearn to know which parameters are stateful and which are stateless.

Routed parameters
-----------------

.. _param-routing:

For more advanced used cases, SciKeras supports
Scikit-Learn style parameter routing to override parameters
for individual consumers (methods or class initializers).

All special prefixes are stored in the ``prefixes_`` class attribute
of :py:class:`scikeras.wrappers.BaseWrappers`. Currently, they are:

- ``model__``: passed to ``model_build_fn`` (or whatever function is passed to the ``model`` param of :class:`scikeras.wrappers.BaseWrapper`).
- ``fit__``: passed to :func:`tensorflow.keras.Model.fit`
- ``predict__``: passed to :func:`tensorflow.keras.Model.predict`. Note that internally SciKeras also uses :func:`tensorflow.keras.Model.predict` within :func:`scikeras.wrappers.BaseWrapper.score` and so this prefix applies to both.
- ``callbacks__``: used to instantiate callbacks.
- ``optimizer__``: used to instantiate optimizers.
- ``loss__``: used to instantiate losses.
- ``metrics__``: used to instantiate metrics.
- ``score__``: passed to the scoring function, i.e. :func:`scikeras.wrappers.BaseWrapper.scorer`.

All routed parameters will be available for hyperparameter tuning.

Below are some example use cases.

Example: multiple losses with routed parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

    clf = KerasClassifier(
        model=model_build_fn,
        loss=[BinaryCrossentropy, CategoricalCrossentropy],
        loss__from_logits=True,  # BinaryCrossentropy(from_logits=True) & CategoricalCrossentropy(from_logits=True)
        loss__label_smoothing=0.1,  # passed to each sub-item, i.e. `loss=[l(label_smoothing=0.1) for l in loss]`
        loss__1__label_smoothing=0.5,  # overrides the above, results in CategoricalCrossentropy(label_smoothing=0.5)
    )


Custom Scorers
--------------

SciKeras uses :func:`sklearn.metrics.accuracy_score` and :func:`sklearn.metrics.accuracy_score`
as the scoring functions for :class:`scikeras.wrappers.KerasClassifier`
and :class:`scikeras.wrappers.KerasRegressor` respectively. To override these scoring functions,
override :func:`scikeras.wrappers.KerasClassifier.scorer`
or :func:`scikeras.wrappers.KerasRegressor.scorer`.


.. _Keras Callbacks docs: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks

.. _Keras Metrics docs: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
