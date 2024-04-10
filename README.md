# Scikit-Learn Wrapper for Keras

[![Build Status](https://github.com/adriangb/scikeras/workflows/Tests/badge.svg)](https://github.com/adriangb/scikeras/actions?query=workflow%3ATests+branch%3Amaster)
[![Coverage Status](https://codecov.io/gh/adriangb/scikeras/branch/master/graph/badge.svg)](https://codecov.io/gh/adriangb/scikeras)
[![Docs](https://github.com/adriangb/scikeras/workflows/Build%20Docs/badge.svg)](https://www.adriangb.com/scikeras/)

Scikit-Learn compatible wrappers for Keras Models.

## Why SciKeras

SciKeras is derived from and API compatible with the now deprecated / removed `tf.keras.wrappers.scikit_learn`.

An overview of the differences as compared to the TF wrappers can be found in our
[migration](https://www.adriangb.com/scikeras/stable/migration.html) guide.

## Installation

This package is available on PyPi:

```bash
# Tensorflow
pip install scikeras[tensorflow]
```

Note that `pip install scikeras[tensorflow]` is basically equivalent to `pip install scikeras tensorflow`
and is offered just for convenience. You can also install just SciKeras with
`pip install scikeras`, but you will need a version of tensorflow installed at
runtime or SciKeras will throw an error when you try to import it.

The current version of SciKeras depends on `scikit-learn>=1.4.1post1` and `Keras>=3.2.0`.

### Migrating from `keras.wrappers.scikit_learn`

Please see the [migration](https://www.adriangb.com/scikeras/stable/migration.html) section of our documentation.

## Documentation

Documentation is available at [https://www.adriangb.com/scikeras/](https://www.adriangb.com/scikeras/).

## Contributing

See [CONTRIBUTING.md](https://github.com/adriangb/scikeras/blob/master/CONTRIBUTING.md)
