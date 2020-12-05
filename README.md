# Scikit-Learn Wrapper for Keras

[![Build Status](https://github.com/adriangb/scikeras/workflows/Tests/badge.svg)](https://github.com/adriangb/scikeras/actions?query=workflow%3ATests+branch%3Amaster)
[![Coverage Status](https://codecov.io/gh/adriangb/scikeras/branch/master/graph/badge.svg)](https://codecov.io/gh/adriangb/scikeras)
[![Docs Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://scikeras.readthedocs.io/en/latest/?badge=latest)

Scikit-Learn compatible wrappers for Keras Models.

## Why SciKeras

SciKeras is derived from and API compatible with `tf.keras.wrappers.scikit_learn`. The original TensorFlow (TF) wrappers are not actively mantained,
and [may be deprecated](https://github.com/tensorflow/tensorflow/pull/37201#pullrequestreview-391650001) at some point.
In addition, they have many incompatibilities with both the Keras ecosystem and the Scikit-Learn ecosystem.
SciKeras attempts to resolve these issues by providing maintained, documented wrappers that are fully compatible with the
entire Scikit-Learn and Keras ecosystems. Some advantages over the TF wrappers are:

* Full compatibility with the Scikit-Learn API, including grid searches, ensembles, transformers, etc.
* Support for Functional and Subclassed Keras Models.
* Support for pre-trained models.
* Support for dynamically set Keras parameters depending on inputs (e.g. input shape).
* Support for hyperparameter tuning of optimizers and losses.
* Support for multi-input and multi-ouput Keras models.
* Functional `random_state` for reproducible training.
* Many more that you will discover as you use SciKeras!

## Installation

This package is available on PyPi:

```bash
pip install scikeras
```

The only dependencies are `scikit-learn>=0.22` and `TensorFlow>=2.2.0`.

## Migrating from `tf.keras.wrappers.scikit_learn`

Please see the [migration](https://scikeras.readthedocs.io/en/latest/migration.html) section of our documentation.

## Documentation

Documentation is available on [ReadTheDocs](https://scikeras.readthedocs.io/en/latest/).

## Contributing

See [CONTRIBUTING.md](https://github.com/adriangb/scikeras/blob/master/CONTRIBUTING.md)
