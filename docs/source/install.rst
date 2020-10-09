============
Installation
============

.. contents::
   :local:


pip installation
~~~~~~~~~~~~~~~~

To install with pip, run:

.. code:: bash

    pip install -U scikeras

We recommend to use a virtual environment for this.

From source
~~~~~~~~~~~

If you would like to use the must recent additions to scikeras or
help development, you should install scikeras from source.

We use `Poetry<http://pytorch.org/>`__ to manage dependencies.

.. code:: bash

    git clone https://github.com/adriangb/scikeras.git
    cd scikeras
    poetry install
    poetry shell

    pytest -n auto  # parallelized tests via pytest-xdist
