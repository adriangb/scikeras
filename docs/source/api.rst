.. _scikeras-api:

============
SciKeras API
============

Arguments to :py:class:`scikeras.wrappers.BaseWrapper`
------------------------------------------------------

A complete explanation of all arguments and methods of
:py:class:`.BaseWrapper` are found in the :ref:`scikeras-api` documentation. Here we
focus on the main ones.

model
^^^^^

This is where you pass your Keras :class:``tensorflow.keras.Model``
building function (``model_build_fn``), or ``Model`` instance.
Unless you are using a pre-instantiated ``Model``, the arguments
for your model should be passed to :py:func:`.BaseWrapper.__init__`
as keyword arguments. These will then be passed to
``model_build_fn`` so that you can use them to build your ``Model``.
For example, if your model takes the argument
``hidden_layer_sizes``, the code would look like this:

.. code:: python

    def model_build_fn(hidden_layer_sizes):
        model = Model(...)
        ...
        return model

    clf = KerasClassifier(
        model=model_build_fn,
        hidden_layer_sizes=(100,),
    )

random_state
^^^^^^^^^^^^

This behaves similar to the same parameter in ``sklearn`` estimators.
If set to an integer or a :py:class:`~numpy.random.RandomState` instance,
it will be used to seed the random number generators used to initialize
the graph and optimizers.

.. note::

    Use of this parameter may have
    unforeseen consequences since ``TensorFlow`` only has a *global* random
    state.

optimizer
^^^^^^^^^

Like :py:class:`~tensorflow.keras.Model`, this can be a string,
optimizer instance or optimizer class. If you pass a class,
you will be able to specify it's arguments via parameter routing (see
:ref:`param-routing`).

batch_size
^^^^^^^^^^

This argument is passed to :py:func:`~tensorflow.keras.Model.fit`. See the the 
`Keras Model docs`_ for more details.

validation_split
^^^^^^^^^^^^^^^^

This argument is passed to :py:func:`~tensorflow.keras.Model.fit`. See the the 
`Keras Model docs`_ for more details.

callbacks
^^^^^^^^^

Single or list of callback instances or classes. Using classes will allow
you to pass their parameters via parameter routing (see
:ref:`param-routing`).
For more information on Keras callbacks, see 
`Keras Callbacks docs`_.

metrics
^^^^^^^

List or dict of metrics. See the the 
`Keras Metrics docs`_ for more details on using metrics.
If you pass classes instead of string names or instances, 
you will be able to pass their parameters via parameter routing (see
:ref:`param-routing`).

warm_start
^^^^^^^^^^

This argument determines whether each
:py:func:`~scikeras.wrappers.BaseWrapper.fit` call leads to a re-initialization of
the :py:class:`~scikeras.wrappers.BaseWrapper` or not. By default, when calling
:py:func:`~scikeras.wrappers.BaseWrapper.fit`, the parameters of the net are
initialized, so your previous training progress is lost (consistent
with the sklearn ``fit()`` calls). In contrast, with
``warm_start=True``, each :py:func:`~scikeras.wrappers.BaseWrapper.fit` call will
continue from the most recent state.

verbose
^^^^^^^

``False`` disables the progress bar and other logging
while ``True`` enables it.
This argument is passed to multiple methods of :py:class:`~tensorflow.keras.Model`.
To set different values for ``fit`` and ``predict`` for example, you can use
``fit__verbose=True`` and ``predict__verbose=False`` or
``verbose=True`` and ``predict__verbose=False`` which would have the same effect
since the non routed value from ``verbose=True`` would be passed to ``fit``.

shuffle
^^^^^^^

This argument is passed to :py:func:`~tensorflow.keras.Model.fit`. See the the 
`Keras Model docs`_ for more details.

run_eagerly
^^^^^^^^^^^

This argument is passed to :py:func:`~tensorflow.keras.Model.fit`. See the the 
`Keras Model docs`_ for more details.


Methods of :py:class:`scikeras.wrappers.BaseWrapper`
----------------------------------------------------

fit(X, y, sample_weights=None)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is one of the main methods you will use. It contains everything
required to train the model, be it batching of the data, triggering
the callbacks, or handling the internal validation set.

The ``X`` and ``y`` arguments to :py:func:`scikeras.wrappers.BaseWrapper.fit`
are expected to by array-like and fit in memory (e.g, NumPy arrays).
SciKeras does not currently support :py:class:`tensorflow.data.Dataset` inputs.


partial_fit(X, y, sample_weights=None)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to :py:func:`scikeras.wrapper.BaseWrapper.fit`, there is also the
:py:func:`scikeras.wrapper.BaseWrapper.partial_fit` method, known from some
sklearn estimators. :py:func:`scikeras.wrapper.BaseWrapper.partial_fit` allows
you to continue training from your current status, even if you set
``warm_start=False``. A further use case for
:py:func:`scikeras.wrapper.BaseWrapper.partial_fit` is when your data does not
fit into memory and you thus need to have several training steps.

For :class:`scikeras.wrappers.KerasClassifier`,
there is an extra ``classes`` parameter available: 
``partial_fit(X, y, sample_weights=None, classes=None)``
The `classes` param is expected to be a list or 1D numpy
array containing all of the classes that will be seen for all `partial_fit`
calls, allowing you to make ``partial_fit`` calls with targets
that only contain a subset of all classes.


predict(X) and predict_proba(X)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These methods use :py:func:`tensorflow.keras.Model.predict` to predict
``y`` or ``y``'s probabilities based on ``X``. Outputs are cast to
numpy arrays of the same dtype and shape as the input. If
:py:func:`tensorflow.keras.Model.predict` returns multiple outputs as a list,
these are column-stacked into a single array.
This allows the use of simple multi-output models
without any custom logic or intervention. For more complex cases,
you will need to subclass :py:class:`scikeras.wrappers.BaseWrapper`
and override the :py:func:`scikeras.wrappers.BaseWrapper.postprocess_y`
method.

In case of :py:class:`scikeras.wrappers.KerasClassifier`, 
when :py:func:`scikeras.wrappers.KerasClassifier.fit` is called
SciKeras uses the target type (as determined by
:py:func:`~sklearn..utils.multiclass.type_of_target`), the loss function
used to compile :py:class:`tensorflow.keras.Model` and the number of
outputs from :py:class:`tensorflow.keras.Model` to automatically determine
what encodings and transformations are necessary.
:py:func:`scikeras.wrappers.KerasClassifier.predict` also reverses
this encoding to return class labels. On the other hand,
:py:func:`scikeras.wrappers.KerasClassifier.predict_proba` returns
the raw class probabilities.

score(X, y)
^^^^^^^^^^^

This method returns the mean accuracy on the given data and labels for
classifiers and the coefficient of determination R^2 of the prediction for
regressors. All wrappers rely on the abstract method
:py:func:`scikeras.wrappers.BaseWrapper._scorer`
with the signature ``_scorer(y_true, y_pred, sample_weights)``
to do the scoring. If you want to swap in an alternative scorer (or implement
a scorer in the case of :py:class:`scikeras.wrappers.BaseWrapper`) all you have
to do is implement this method.

.. autosummary::
   :template: class.rst
   :toctree: generated/
   
   scikeras.wrappers.BaseWrapper
   scikeras.wrappers.KerasClassifier
   scikeras.wrappers.KerasRegressor
