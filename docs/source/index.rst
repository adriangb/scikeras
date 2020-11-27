Welcome to SciKeras's documentation!
====================================

The goal of scikeras is to make it possible to use Keras/TensorFlow with sklearn. 
This is achieved by providing a wrapper around Keras that has an Scikit-Learn interface.
SciKeras is the successor to ``tf.keras.wrappers.scikit_learn``.

SciKeras tries to make things easy for you while staying out of your way.
If you are familiar with Scikit-Learn and Keras, you don’t have to learn any new concepts, and the syntax should be well known.

Overall, SciKeras aims at being as flexible as Keras while having a clean interface as Scikit-Learn.

.. toctree::
   :maxdepth: 2

   install
   quickstart
   tutorials
   advanced
   api

Differences with ``tf.keras.wrappers.scikit_learn``
---------------------------------------------------

* Full compatibility with the Scikit-Learn API, including grid searches, ensembles, transformers, etc.
* Support for Keras Functional and Subclassed Models.
* Support for pre-trained models.
* Support for dynamically set Keras parameters depending on inputs (e.g. input shape).
* Support for hyperparameter tuning of optimizers and losses.
* Support for multi-input and multi-ouput Keras models.
* Functional `random_state` for reproducible training.
* Many more that you will discover as you use SciKeras!


Transitioning from ``tf.keras.wrappers.scikit_learn``
-----------------------------------------------------

SciKeras is largely backwards compatible with the existing wrappers. For most cases, you can just change your import statement from:

.. code:: python

   from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  # from
   from scikeras.wrappers import KerasClassifier, KerasRegressor  # to


SciKeras does however have some backward incompatible changes:

One-hot encoding of targets for categorical crossentropy losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SciKeras will not longer implicitly inspect your Model's loss function to determine if
it needs to one-hot encode your target to match the loss function. Instead, you must explicitly
pass your loss function to the constructor:

.. code:: python

   clf = KerasClassifier(loss="categorical_crossentropy")

Removal of ``**kwargs`` from fit and predict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a future release of SciKeras, ``**kwargs`` will be removed from fit and predict. To future
proof your code, you should instead declare these parameters in your constructor:

.. code:: python

   clf = KerasClassifier(batch_size=32)

Or to declare separate values for ``fit`` and ``predict``:

.. code:: python

   clf = KerasClassifier(fit__batch_size=32, predict__batch_size=32)

Renaming of ``build_fn`` to ``model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SciKeras renamed the constructor argument ``build_fn`` to ``model``. In a future release,
passing ``build_fn`` as a _keyword_ argument will raise a ``TypeError``. Passing it as a positional
argument remains unchanged. You can make the following change to future proof your code:

.. code:: python

   clf = KerasClassifier(build_fn=...)  # from
   clf = KerasClassifier(model=...)  # to

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
