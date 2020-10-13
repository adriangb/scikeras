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


.. _Poetry: http://pytorch.org/
