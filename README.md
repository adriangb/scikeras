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
# Normal tensorflow
pip install scikeras[tensorflow]

# or tensorflow-cpu
pip install scikeras[tensorflow-cpu]
```

SciKeras packages TensorFlow as an optional dependency because there are
several flavors of TensorFlow available (`tensorflow`, `tensorflow-cpu`, etc.).
Depending on _one_ of them in particular disallows the usage of the other, which is why
they need to be optional.

`pip install scikeras[tensorflow]` is basically equivalent to `pip install scikeras tensorflow`
and is offered just for convenience. You can also install just SciKeras with
`pip install scikera`s, but you will need a version of tensorflow installed at
runtime or SciKeras will throw an error when you try to import it.

The current version of SciKeras depends on `scikit-learn>=1.0.0` and `TensorFlow>=2.7.0`.

### Migrating from `tf.keras.wrappers.scikit_learn`

Please see the [migration](https://www.adriangb.com/scikeras/stable/migration.html) section of our documentation.

## Documentation

Documentation is available at [https://www.adriangb.com/scikeras/](https://www.adriangb.com/scikeras/).

## Contributing

See [CONTRIBUTING.md](https://github.com/adriangb/scikeras/blob/master/CONTRIBUTING.md)
