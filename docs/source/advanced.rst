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

Only when the :py:func:`~scikeras.wrappers.KerasClassifier.fit` or
:py:func:`~scikeras.wrappers.KerasRegressor.fit` method are called, are the
different attributes of the wrapper, such as the ``model``, initialized.
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


Multiple inputs or outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some cases, the input actually consists of multiple inputs. E.g.,
in a text classification task, you might have an array that contains
the integers representing the tokens for each sample, and another
array containing the number of tokens of each sample. SciKeras has you
covered here as well.

Scikit-Learn natively supports multiple outputs, although it technically
requires them to be arrays of equal length
(see docs for Scikit-Learn's :py:class:`~sklearn.multioutput.MultiOutputClassifier`).
Scikit-Learn has no support for multiple inputs.
To work around this issue, SciKeras implements a data conversion
abstraction in the form of Scikit-Learn style transformers,
one for ``X`` (features) and one for ``y`` (target).
By implementing a custom transformer, you can split a single input ``X`` into multiple inputs
for :py:class:`tensorflow.keras.Model` or perform any other manipulation you need.
To override the default transformers, simply override
:py:func:`scikeras.wrappers.BaseWrappers.target_transformer` or
:py:func:`scikeras.wrappers.BaseWrappers.function_transformer` for ``y`` and ``X`` respectively.

This said, note that if you are trying to use outputs of uneven length or
other more complex scenarios, SciKeras may be able to handle them but the rest
of the Scikit-Learn ecosystem likely will not.

For a complete example, see the multi-input and multi-output examples in the
:ref:`tutorials` section.

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

.. _Keras Callbacks docs: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks

.. _Keras Metrics docs: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
