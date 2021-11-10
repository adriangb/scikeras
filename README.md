# Scikit-Learn Wrapper for Keras

[![Build Status](https://github.com/adriangb/scikeras/workflows/Tests/badge.svg)](https://github.com/adriangb/scikeras/actions?query=workflow%3ATests+branch%3Amaster)
[![Coverage Status](https://codecov.io/gh/adriangb/scikeras/branch/master/graph/badge.svg)](https://codecov.io/gh/adriangb/scikeras)
[![Docs](https://readthedocs.org/projects/docs/badge/?version=latest)](https://www.adriangb.com/scikeras/)

Scikit-Learn compatible wrappers for Keras Models.

## Why SciKeras

SciKeras is derived from and API compatible with `tf.keras.wrappers.scikit_learn`. The original TensorFlow (TF) wrappers are not actively maintained,
and [will be removed](https://github.com/tensorflow/tensorflow/pull/36137#issuecomment-726271760) in a future release.

An overview of the advantages and differences as compared to the TF wrappers can be found in our
[migration](https://www.adriangb.com/scikeras/stable/migration.html) guide.

## Installation

This package is available on PyPi:

```bash
pip install scikeras
```

The current version of SciKeras depends on `scikit-learn>=1.0.0` and `TensorFlow>=2.7.0`.

You will need to manually install TensorFlow; due to TensorFlow's packaging it is not a direct dependency of SciKeras.
You can do this by running:

```bash
pip install tensorflow
```

### Migrating from `tf.keras.wrappers.scikit_learn`

Please see the [migration](https://www.adriangb.com/scikeras/stable/migration.html) section of our documentation.

## Documentation

Documentation is available at [https://www.adriangb.com/scikeras/](https://www.adriangb.com/scikeras/).

## Contributing

See [CONTRIBUTING.md](https://github.com/adriangb/scikeras/blob/master/CONTRIBUTING.md)
