Welcome to SciKeras's documentation!
====================================

The goal of scikeras is to make it possible to use Keras/TensorFlow with sklearn. 
This is achieved by providing a wrapper around Keras that has an Scikit-Learn interface.
SciKeras is the successor to ``tf.keras.wrappers.scikit_learn``.

Some advantages over the ``tf.keras.wrappers.scikit_learn`` wrappers are:

* Full compatibility with the Scikit-Learn API, including grid searches, ensembles, transformers, etc.
* Support for pre-trained models.
* Support for dynamically set Keras parameters depending on inputs (e.g. input shape).
* Support for hyperparameter tuning of optimizers and losses.
* Support for multi-input and multi-ouput Keras models.
* Functional `random_state` for reproducible training.
* Many more that you will discover as you use SciKeras!

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
