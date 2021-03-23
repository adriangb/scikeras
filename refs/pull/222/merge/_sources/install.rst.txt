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

To install without any dependencies, eg. to use a custom
TensorFlow build or `tensorflow-cpu`_, you can instruct pip to ignore dependencies:

.. code:: bash

    pip install --no-deps scikeras

Then manually install compatible versions of TensorFlow and Scikit-Learn.
Currently, our minium versions are:

- TensorFlow: v2.4.0
- Scikit-Learn: v0.22.0

Developer Installation
~~~~~~~~~~~~~~~~~~~~~~

If you would like to use the must recent additions to scikeras or
help development, you should install scikeras from source.

We use Poetry_ to manage dependencies.

.. code:: bash

    git clone https://github.com/adriangb/scikeras.git
    cd scikeras
    poetry install
    poetry shell

    pytest -n auto  # parallelized tests via pytest-xdist


.. _Poetry: https://python-poetry.org/
.. _tensorflow-cpu: https://pypi.org/project/tensorflow-cpu/
