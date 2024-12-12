============
Installation
============

.. contents::
   :local:


Users Installation
~~~~~~~~~~~~~~~~~~

To install with pip, run:

.. code:: bash

    pip install scikeras

This will install SciKeras and Keras.
Keras does not automatically install a backend.
For example, to install TensorFlow you can do:

.. code:: bash

    pip install tensorflow

You can also install SciKeras without any dependencies, for example to install a nightly version of Scikit-Learn:

.. code:: bash

    pip install --no-deps scikeras

As of SciKeras v0.5.0, the minimum required versions are as follows:

- Keras: v3.2.0
- Scikit-Learn: v1.4.1post1

Developer Installation
~~~~~~~~~~~~~~~~~~~~~~

If you would like to use the must recent additions to SciKeras or
help development, you should install SciKeras from source.

We use Poetry_ to manage dependencies.

.. code:: bash

    git clone https://github.com/adriangb/scikeras.git
    cd scikeras
    poetry install
    poetry shell

    pytest -v


.. _Poetry: https://python-poetry.org/
