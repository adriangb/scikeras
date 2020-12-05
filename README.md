# Scikit-Learn Wrapper for Keras

[![Build Status](https://github.com/adriangb/scikeras/workflows/Tests/badge.svg)](https://github.com/adriangb/scikeras/actions?query=workflow%3ATests+branch%3Amaster)
[![Coverage Status](https://codecov.io/gh/adriangb/scikeras/branch/master/graph/badge.svg)](https://codecov.io/gh/adriangb/scikeras)
[![Docs Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://scikeras.readthedocs.io/en/latest/?badge=latest)

Scikit-Learn compatible wrappers for Keras Models.

## Why SciKeras

SciKeras is derived from and API compatible with `tf.keras.wrappers.scikit_learn`. The original TensorFlow (TF) wrappers are not actively mantained,
and [may be deprecated](https://github.com/tensorflow/tensorflow/pull/37201#pullrequestreview-391650001) at some point.

An overview of the advantages and differences as compared to the TF wrappers can be found in our
[migration](https://scikeras.readthedocs.io/en/latest/migration.html) guide.

## Installation

This package is available on PyPi:

```bash
pip install scikeras
```

The only dependencies are `scikit-learn>=0.22` and `TensorFlow>=2.2.0`.

### Migrating from `tf.keras.wrappers.scikit_learn`

Please see the [migration](https://scikeras.readthedocs.io/en/latest/migration.html) section of our documentation.

## Documentation

Documentation is available on [ReadTheDocs](https://scikeras.readthedocs.io/en/latest/).

## Contributing

See [CONTRIBUTING.md](https://github.com/adriangb/scikeras/blob/master/CONTRIBUTING.md)
