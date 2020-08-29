=========
Wrappers
=========

Using SciKeras Wrappers
-----------------------

:class:`.KerasClassifier` and :class:`.KerasRegressor` are the main touch point
for the user. They wrap the Keras :class:`~tensorflow.keras.Model` while
providing an interface that should be familiar for sklearn users. Both of these
wrapper are based on :class:`.BaseWrapper`, which defines the main functionality
of the SciKeras wrappers.

Define your :class:`~tensorflow.keras.Model` architecture like you always do,
but within a callable top-level function.
Then pass this function to :class:`.BaseWrapper` in the ``model`` parameter.
Finally, you can call :func:`~scikeras.wrappers.BaseWrapper.fit`
and :func:`~scikeras.wrappers.BaseWrapper.predict`, as with an sklearn
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
- compiles the :class:`~tensorflow.keras.Model` (unless you do it yourself in ``model_build_fn``)
- makes all ``Keras`` objects serializable.

SciKeras abstracts away the incompatibilities and data conversions,
letting you focus on defining your architecture and
choosing your hyperparameters.
At the same time, SciKeras is very flexible and can be
extended with ease, getting out of your way as much as possible.

Initialization
^^^^^^^^^^^^^^

When you instantiate the :class:`.KerasClassifier` or
:class:`.KerasRegressor` instance, only the given arguments are stored.
These arguments are stored unmodified. For instance, the ``model`` will
remain uninstantiated. This is to make sure that the arguments you
pass are not touched afterwards, which makes it possible to clone the
wrapper instance, for example in a :class:`~sklearn.model_selection.GridSearchCV`.

Only when the :func:`~scikeras.wrappers.KerasClassifier.fit` or
:func:`~scikeras.wrappers.KerasRegressor.fit` method are called, are the
different attributes of the wrapper, such as the ``model``, initialized.
An initialized attribute's name always ends on an underscore; e.g., the
initialized ``module`` is called ``model_``. (This is the same
nomenclature as sklearn uses.) Therefore, you always know which
attributes you set and which ones were created by the wrappers.

Once initialized by calling ``fit``, the wrappers create several attributes,
documented in the SciKeras API docs.

Compilation of ``Model``
^^^^^^^^^^^^^^^^^^^^^^^^

SciKeras allows you to compile your own model within your ``model`` building
function, or you can choose to return an uncompiled model from ``model``,
in which case SciKeras will compile it for you using the parameters passed
at initialization (``optimizer``, ``loss`` and ``metrics``).

The first route will be more flexible if you wish to determine how to compile
your ``Model`` within the same function in which you define it. The latter will
offer an easy way to compile and tune compilation parameters. Examples:

.. code:: python

    def model_build_fn(compile_params):
        optimizer = compile_params["compile_params"]
        model = Model(...)
        ...
        model.compile(optimizer=optimizer)
        return model

    clf = KerasClassifier(model=model_build_fn)
    clf.fit(X, y)
    y_pred = clf.predict(X_valid)

.. code:: python

    def model_build_fn():
        model = Model(...)
        ...
        # Do not call model.compile
        return model  # That's it, SciKeras will compile your model for you

    clf = KerasClassifier(model=model_build_fn)
    clf.fit(X, y)
    y_pred = clf.predict(X_valid)


Arguments to ``model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You probably wish to pass parameters from :class:`~scikeras.wrappers.BaseWrapper`
to ``model``, or you may want to use attributes from
:class:`~scikeras.wrappers.BaseWrapper` such as ``n_features_in_`` while building
your model. SciKeras allows you to do both.

To enable this, SciKeras uses two special arguments to ``model`` that will only
be passed if they are present in ``model``'s signature (i.e. there is an argument
with the same name in ``model``'s signature):

``meta_params``
+++++++++++++++
This is a dictionary containing all of the attributes that
:class:`~scikeras.wrappers.BaseWrapper` creates when it is initialized
These include ``n_features_in_``, ``y_dtype_``, etc. For a full list,
see the SciKeras API docs.

``compile_params``
++++++++++++++++++
This is a dictionary of parameters destined for :func:`tensorflow.Keras.Model.compile`.
You will want to accept this parameter unless you are returning an un-compiled ``Model``
instance. Parameters available via this dictionary are:

* ``optimizer``
* ``loss``
* ``callbacks``
* Any other parameters with the prefix ``compile__``

Arguments originally passed to :func:`scikeras.wrappers.BaseWrapper.__init__`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Any **other** arguments passed to :func:`scikeras.wrappers.BaseWrapper.__init__` when
the wrapper is instantiated will also be passed to ``model`` as keyword
arguments.

Most important arguments and methods
------------------------------------

A complete explanation of all arguments and methods of
:class:`.BaseWrapper` are found in the SciKeras API documentation. Here we
focus on the main ones.

model
^^^^^

This is where you pass your Keras ``Model``.
Ideally, it should not be instantiated, although instantiated models
are accepted. Instead, the init arguments
for your model should be passed to :class:`.BaseWrapper` either as
bare keyword arguments (if their name does not conflict with any of
the existing keyword arguments) or with the ``model__`` prefix,
which will override any arguments of the same name only when passed to
``model``. E.g., if your module takes the arguments
``hidden_layer_sizes`` and ``lr``, the code would look like this:

.. code:: python

    def model_build_fn(hidden_layer_sizes, lr):
        model = Model(...)
        ...
        return model

    clf = KerasClassifier(
        model=model_build_fn,
        hidden_layer_sizes=(100,),
        model__lr=0.5,  # also equivalent to just lr=0.5 in this case
    )

Note that SciKeras automatically interprets the type of classification
task when using ``KerasClassifier``, as determined by
:func:`~sklearn..utils.multiclass.type_of_target`. This means that if you
pass a binary target, you need to define a your ``Model``'s output layer
to apply a ``sigmoid`` nonlinearity to get good results.

random_state
^^^^^^^^^^^^

This behaves similar to the same parameter in ``sklearn`` estimators.
If set to an integer or a :class:`~numpy.random.RandomState` instance,
it will be used to seed the random number generators used to initialize
the graph and optimizers. Note that use of this parameter may have
unforeseen consequences since ``TensorFlow`` only has a *global* random
state.

optimizer
^^^^^^^^^

If this is an instantiated class from :module:`~tensorflow.keras.optimizers`
it will be passed directly to ``Model`` during compilation or to ``model``
via the ``compile_params`` argument.
After initializing the :class:`.NeuralNet`, the initialized optimizer
will stored in the ``optimizer_`` attribute.  During initialization
you can define param groups, for example to set different learning
rates for certain parameters. The parameters are selected by name with
support for wildcards (globbing):

.. code:: python

    optimizer__param_groups=[
        ('embedding.*', {'lr': 0.0}),
        ('linear0.bias', {'lr': 1}),
    ]

lr
^^^

The learning rate. This argument exists for convenience, since it
could also be set by ``optimizer__lr`` instead. However, it is used so
often that we provided this shortcut. If you set both ``lr`` and
``optimizer__lr``, the latter have precedence.

max_epochs
^^^^^^^^^^

The maximum number of epochs to train with each
:func:`~skorch.net.NeuralNet.fit` call. When you call
:func:`~skorch.net.NeuralNet.fit`, the net will train for this many
epochs, except if you interrupt training before the end (e.g. by using
an early stopping callback or interrupt manually with *ctrl+c*).

If you want to change the number of epochs to train, you can either
set a different value for ``max_epochs``, or you call
:func:`~skorch.net.NeuralNet.fit_loop` instead of
:func:`~skorch.net.NeuralNet.fit` and pass the desired number of
epochs explicitly:

.. code:: python

    net.fit_loop(X, y, epochs=20)


batch_size
^^^^^^^^^^

This argument controls the batch size for ``iterator_train`` and
``iterator_valid`` at the same time. ``batch_size=128`` is thus a
convenient shortcut for explicitly typing
``iterator_train__batch_size=128`` and
``iterator_valid__batch_size=128``. If you set all three arguments,
the latter two will have precedence.

train_split
^^^^^^^^^^^

This determines the :class:`.NeuralNet`\'s internal train/validation
split. By default, 20% of the incoming data is reserved for
validation. If you set this value to ``None``, all the data is used
for training.

For more details, please look at :ref:`dataset <dataset>`.

callbacks
^^^^^^^^^

For more details on the callback classes, please look at
:ref:`callbacks <skorch.callbacks>`.

By default, :class:`.NeuralNet` and its subclasses start with a couple
of useful callbacks. Those are defined in the
:func:`~skorch.net.NeuralNet.get_default_callbacks` method and
include, for instance, callbacks for measuring and printing model
performance.

In addition to the default callbacks, you may provide your own
callbacks. There are a couple of ways to pass callbacks to the
:class:`.NeuralNet` instance. The easiest way is to pass a list of all
your callbacks to the ``callbacks`` argument:

.. code:: python

    net = NeuralNet(
        module=MyModule,
        callbacks=[
            MyCallback1(...),
            MyCallback2(...),
        ],
    )

Inside the :class:`.NeuralNet` instance, each callback will receive a
separate name. Since we provide no name in the example above, the
class name will taken, which will lead to a name collision in case of
two or more callbacks of the same class. This is why it is better to
initialize the callbacks with a list of tuples of *name* and *callback
instance*, like this:

.. code:: python

    net = NeuralNet(
        module=MyModule,
        callbacks=[
            ('cb1', MyCallback1(...)),
            ('cb2', MyCallback2(...)),
        ],
    )

This approach of passing a list of *name*, *instance* tuples should be
familiar to users of sklearn\ :class:`~sklearn.pipeline.Pipeline`\s
and :class:`~sklearn.pipeline.FeatureUnion`\s.

An additonal advantage of this way of passing callbacks is that it
allows to pass arguments to the callbacks by name (using the
double-underscore notation):

.. code:: python

    net.set_params(callbacks__cb1__foo=123, callbacks__cb2__bar=456)

Use this, for instance, when trying out different callback parameters
in a grid search.

*Note*: The user-defined callbacks are always called in the same order
as they appeared in the list. If there are dependencies between the
callbacks, the user has to make sure that the order respects them.
Also note that the user-defined callbacks will be called *after* the
default callbacks so that they can make use of the things provided by
the default callbacks. The only exception is the default callback
:class:`~skorch.callbacks.PrintLog`, which is always called last.

warm_start
^^^^^^^^^^

This argument determines whether each
:func:`~skorch.net.NeuralNet.fit` call leads to a re-initialization of
the :class:`.NeuralNet` or not. By default, when calling
:func:`~skorch.net.NeuralNet.fit`, the parameters of the net are
initialized, so your previous training progress is lost (consistent
with the sklearn ``fit()`` calls). In contrast, with
``warm_start=True``, each :func:`~skorch.net.NeuralNet.fit` call will
continue from the most recent state.

device
^^^^^^

As the name suggests, this determines which computation device should
be used. If set to ``cuda``, the incoming data will be transferred to
CUDA before being passed to the PyTorch :class:`~torch.nn.Module`. The
device parameter adheres to the general syntax of the PyTorch device
parameter.

initialize()
^^^^^^^^^^^^

As mentioned earlier, upon instantiating the :class:`.NeuralNet`
instance, the net's components are not yet initialized. That means,
e.g., that the weights and biases of the layers are not yet set. This
only happens after the :func:`~skorch.net.NeuralNet.initialize` call.
However, when you call :func:`~skorch.net.NeuralNet.fit` and the net
is not yet initialized, :func:`~skorch.net.NeuralNet.initialize` is
called automatically. You thus rarely need to call it manually.

The :func:`~skorch.net.NeuralNet.initialize` method itself calls a
couple of other initialization methods that are specific to each
component. E.g., :func:`~skorch.net.NeuralNet.initialize_module` is
responsible for initializing the PyTorch module. Therefore, if you
have special needs for initializing the module, it is enough to
override :func:`~skorch.net.NeuralNet.initialize_module`, you don't
need to override the whole :func:`~skorch.net.NeuralNet.initialize`
method.

fit(X, y)
^^^^^^^^^

This is one of the main methods you will use. It contains everything
required to train the model, be it batching of the data, triggering
the callbacks, or handling the internal validation set.

In general, we assume there to be an ``X`` and a ``y``. If you have
more input data than just one array, it is possible for ``X`` to be a
list or dictionary of data (see :ref:`dataset <dataset>`). And if your
task does not have an actual ``y``, you may pass ``y=None``.

If you fit with a PyTorch :class:`~torch.utils.data.Dataset` and don't
explicitly pass ``y``, several components down the line might not work
anymore, since sklearn sometimes requires an explicit ``y`` (e.g. for
scoring). In general, PyTorch :class:`~torch.utils.data.Dataset`\s
should work, though.

In addition to :func:`~skorch.net.NeuralNet.fit`, there is also the
:func:`~skorch.net.NeuralNet.partial_fit` method, known from some
sklearn estimators. :func:`~skorch.net.NeuralNet.partial_fit` allows
you to continue training from your current status, even if you set
``warm_start=False``. A further use case for
:func:`~skorch.net.NeuralNet.partial_fit` is when your data does not
fit into memory and you thus need to have several training steps.

*Tip* :
skorch gracefully catches the ``KeyboardInterrupt``
exception. Therefore, during a training run, you can send a
``KeyboardInterrupt`` signal without the Python process exiting
(typically, ``KeyboardInterrupt`` can be triggered by *ctrl+c* or, in
a Jupyter notebook, by clicking *Kernel* -> *Interrupt*). This way, when
your model has reached a good score before ``max_epochs`` have been
reached, you can dynamically stop training.

predict(X) and predict_proba(X)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These methods perform an inference step on the input data and return
:class:`numpy.ndarray`\s. By default,
:func:`~skorch.net.NeuralNet.predict_proba` will return whatever it is
that the ``module``\'s :func:`~torch.nn.Module.forward` method
returns, cast to a :class:`numpy.ndarray`. If
:func:`~torch.nn.Module.forward` returns multiple outputs as a tuple,
only the first output is used, the rest is discarded.

If the :func:`~torch.nn.Module.forward`\-output can not be cast to a
:class:`numpy.ndarray`, or if you need access to all outputs in the
multiple-outputs case, consider using either of
:func:`~skorch.net.NeuralNet.forward` or
:func:`~skorch.net.NeuralNet.forward_iter` methods to generate outputs
from the ``module``. Alternatively, you may directly call
``net.module_(X)``.

In case of :class:`.NeuralNetClassifier`, the
:func:`~skorch.classifier.NeuralNetClassifier.predict` method tries to
return the class labels by applying the argmax over the last axis of
the result of
:func:`~skorch.classifier.NeuralNetClassifier.predict_proba`.
Obviously, this only makes sense if
:func:`~skorch.classifier.NeuralNetClassifier.predict_proba` returns
class probabilities. If this is not true, you should just use
:func:`~skorch.classifier.NeuralNetClassifier.predict_proba`.

score(X, y)
^^^^^^^^^^^

This method returns the mean accuracy on the given data and labels for
classifiers and the coefficient of determination R^2 of the prediction for
regressors. :class:`.NeuralNet` still has no score method. If you need it,
you have to implement it yourself.

model persistence
^^^^^^^^^^^^^^^^^

In general there are different ways of saving and loading models, each
with their own advantages and disadvantages. More details and usage
examples can be found here: :ref:`save_load`.

If you would like to use pickle (the default way when using
scikit-learn models), this is possible with skorch nets. This saves
the whole net including hyperparameters etc. The advantage is that you
can restore everything to exactly the state it was before. The
disadvantage is it's easier for code changes to break your old saves.

Additionally, it is possible to save and load specific attributes of
the net, such as the ``module``, ``optimizer``, or ``history``, by
calling :func:`~skorch.net.NeuralNet.save_params` and
:func:`~skorch.net.NeuralNet.load_params`. This is useful if you're
only interested in saving a particular part of your model, and is more
robust to code changes.

Finally, it is also possible to use callbacks to save and load models,
e.g. :class:`.Checkpoint`. Those should be used if you need to have
your model saved or loaded at specific times, e.g. at the start or end
of the training process.

Input data
----------

Regular data
^^^^^^^^^^^^

skorch supports numerous input types for data. Regular input types
that should just work are numpy arrays, torch tensors, scipy sparse
CSR matrices, and pandas DataFrames (see also
:class:`~skorch.helper.DataFrameTransformer`).

Typically, your task should involve an ``X`` and a ``y``. If you're
dealing with a task that doesn't require a target (say, training an
autoencoder), you can just pass ``y=None``. Make sure your loss
function deals with this appropriately.

Datasets
^^^^^^^^

:class:`~torch.utils.data.Dataset`\s are also supported, with the
requirement that they should return exactly two items (``X`` and
``y``). For more information on that, take a look at the
:ref:`Dataset` documentation.

Many PyTorch libraries, like torchvision, implement their own
``Dataset``\s. These usually work seamlessly with skorch, as long as
their ``__getitem__`` methods return two outputs. In case they don't,
consider overriding the ``__getitem__`` class and re-arranging the
ouputs so that ``__getitem__`` returns exactly two elements. If the
original implementation returns more than two elements, take a look at
the next section to get an idea how to deal with that.

Multiple input arguments
^^^^^^^^^^^^^^^^^^^^^^^^

In some cases, the input actually consists of multiple inputs. E.g.,
in a text classification task, you might have an array that contains
the integers representing the tokens for each sample, and another
array containing the number of tokens of each sample. skorch has you
covered here as well.

You could supply a list or tuple with all your inputs (``net.fit([tokens,
num_tokens], y)``), but we actually recommend another approach. The best
way is to pass the different arguments as a dictionary. Then the keys
of that dictionary have to correspond to the argument names of your
module's ``forward`` method. Below is an example:

.. code:: python

    X_dict = {'tokens': tokens, 'num_tokens': num_tokens}

    class MyModule(nn.Module):
        def forward(self, tokens, num_tokens):  # <- same names as in your dict
            ...

    net = NeuralNet(MyModule, ...)
    net.fit(X_dict, y)

As you can see, the ``forward`` method takes arguments with exactly
the same name as the keys in the dictionary. This is how the different
inputs are matched. To make this work with
:class:`~sklearn.model_selection.GridSearchCV`, please use
:class:`~skorch.helper.SliceDict`.

Using a dict should cover most use cases that involve multiple
inputs. However, it will fail if your inputs have different
sizes. E.g., if your array of tokens has 1000 elements but your array
of number of tokens has 2000 elements, this would fail. The main
reason for this is batching: How can we know which elements of the two
arrays belong in the same batch?

If your input consists of multiple inputs with different sizes, your
best bet is to implement your own dataset class. That class should
know how it deals with the different inputs, i.e. which elements
belong to the same sample. Again, please refer to the :ref:`Dataset`
section for more details.

Special arguments
-----------------

In addition to the arguments explicitly listed for
:class:`.NeuralNet`, there are some arguments with special prefixes,
as shown below:

.. code:: python

    class MyModule(torch.nn.Module):
        def __init__(self, num_units, dropout):
            ...

    net = NeuralNet(
        module=MyModule,
        module__num_units=100,
        module__dropout=0.5,
        criterion=torch.nn.NLLLoss,
        criterion__weight=weight,
        optimizer=torch.optim.SGD,
        optimizer__momentum=0.9,
    )

Those arguments are used to initialize your ``module``, ``criterion``,
etc. They are not fixed because we cannot know them in advance; in
fact, you can define any parameter for your ``module`` or other
components.

All special prefixes are stored in the ``prefixes_`` class attribute
of :class:`.NeuralNet`. Currently, they are:

- ``module``
- ``iterator_train``
- ``iterator_valid``
- ``optimizer``
- ``criterion``
- ``callbacks``
- ``dataset``

Subclassing NeuralNet
---------------------

Apart from the :class:`.NeuralNet` base class, we provide
:class:`.NeuralNetClassifier`, :class:`.NeuralNetBinaryClassifier`,
and :class:`.NeuralNetRegressor` for typical classification, binary
classification, and regressions tasks. They should work as drop-in
replacements for sklearn classifiers and regressors.

The :class:`.NeuralNet` class is a little less opinionated about the
incoming data, e.g. it does not determine a loss function by default.
Therefore, if you want to write your own subclass for a special use
case, you would typically subclass from :class:`.NeuralNet`.

skorch aims at making subclassing as easy as possible, so that it
doesn't stand in your way. For instance, all components (``module``,
``optimizer``, etc.) have their own initialization method
(:meth:`.initialize_module`, :meth:`.initialize_optimizer`,
etc.). That way, if you want to modify the initialization of a
component, you can easily do so.

Additonally, :class:`.NeuralNet` has a couple of ``get_*`` methods for
when a component is retrieved repeatedly. E.g.,
:func:`~skorch.net.NeuralNet.get_loss` is called when the loss is
determined. Below we show an example of overriding
:func:`~skorch.net.NeuralNet.get_loss` to add L1 regularization to our
total loss:

.. code:: python

    class RegularizedNet(NeuralNet):
        def __init__(self, *args, lambda1=0.01, **kwargs):
            super().__init__(*args, **kwargs)
            self.lambda1 = lambda1

        def get_loss(self, y_pred, y_true, X=None, training=False):
            loss = super().get_loss(y_pred, y_true, X=X, training=training)
            loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
            return loss

.. note:: This example also regularizes the biases, which you typically
    don't need to do.

It is possible to add your own criterion, module, or optimizer to your
customized neural net class. You should follow a few rules when you do
so:

1. Set this attribute inside the corresponding method. E.g., when
   setting an optimizer, use :meth:`.initialize_optimizer` for that.
2. Inside the initialization method, use :meth:`.get_params_for` (or,
   if dealing with an optimizer, :meth:`.get_params_for_optimizer`) to
   retrieve the arguments for the constructor.
3. The attribute name should contain the substring ``"module"`` if
   it's a module, ``"criterion"`` if a criterion, and ``"optimizer"``
   if an optimizer. This way, skorch knows if a change in
   parameters (say, because :meth:`.set_params` was called) should
   trigger re-initialization.

When you follow these rules, you will make sure that your added
components are amenable to :meth:`.set_params` and hence to things
like grid search.

Here is an example of how this could look like in practice:

.. code:: python

    class MyNet(NeuralNet):
        def initialize_criterion(self, *args, **kwargs):
            super().initialize_criterion(*args, **kwargs)

            # add an additional criterion
            params = self.get_params_for('other_criterion')
            self.other_criterion_ = nn.BCELoss(**params)
            return self

        def initialize_module(self, *args, **kwargs):
            super().initialize_module(*args, **kwargs)

            # add an additional module called 'mymodule'
            params = self.get_params_for('mymodule')
            self.mymodule_ = MyModule(**params)
            return self

        def initialize_optimizer(self, *args, **kwargs):
            super().initialize_optimizer(*args, **kwargs)

            # add an additional optimizer called 'optimizer2' that is
            # responsible for 'mymodule'
            named_params = self.mymodule_.named_parameters()
            pgroups, params = self.get_params_for_optimizer('optimizer2', named_params)
            self.optimizer2_ = torch.optim.SGD(*pgroups, **params)
            return self

        ...  # additional changes


    net = MyNet(
        ...,
        other_criterion__reduction='sum',
        mymodule__num_units=123,
        optimizer2__lr=0.1,
    )
    net.fit(X, y)

    # set_params works
    net.set_params(optimizer2__lr=0.05)
    net.partial_fit(X, y)

    # grid search et al. works
    search = GridSearchCV(net, {'mymodule__num_units': [10, 50, 100]}, ...)
    search.fit(X, y)

In this example, a new criterion, a new module, and a new optimizer
were added. Of course, additional changes should be made to the net so
that those new components are actually being used for something, but
this example should illustrate how to start. Since the rules outlined
above are being followed, we can use grid search on our customly
defined components.