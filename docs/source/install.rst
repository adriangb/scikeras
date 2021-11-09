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

We recommend to use a virtual environment for this.

You will need to manually install TensorFlow; due to TensorFlow's packaging it is not a direct dependency of SciKeras.
You can do this by running:

.. code:: bash

    pip install tensorflow

This allows you to install an alternative TensorFlow binary, for example `tensorflow-cpu`_.

You can also install SciKeras without any dependencies, for example to install a nightly version of Scikit-Learn:

.. code:: bash

    pip install --no-deps scikeras

As of SciKeras v0.4.0, the minimum required versions are as follows:

- TensorFlow: v2.7.0
- Scikit-Learn: v1.0.0

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
.. _tensorflow-cpu: https://pypi.org/project/tensorflow-cpu/
