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
* Support for Keras Functional and Subclassed Models.
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

## Transitioning from `tf.keras.wrappers.scikit_learn`

SciKeras is largely backwards compatible with the existing wrappers. For most cases, you can just change your import statement from:

```python
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  # from
from scikeras.wrappers import KerasClassifier, KerasRegressor  # to
```

SciKeras does however have some backward incompatible changes:

### Automatic one-hot encoding of targets for categorical crossentropy losses

SciKeras will not longer implicitly inspect your Model's loss function to determine if
it needs to one-hot encode your target to match the loss function. Instead, you must explicitly
pass your loss function to the constructor:

```python
clf = KerasClassifier(loss="categorical_crossentropy")
```

### Removal of `**kwargs` from fit and predict

In a future release of SciKeras, `**kwargs` will be removed from fit and predict. To future
proof your code, you should instead declare these parameters in your constructor:

```python
clf = KerasClassifier(batch_size=32)
```

Or to declare separate values for `fit` and `predict`:

```python
clf = KerasClassifier(fit__batch_size=32, predict__batch_size=32)
```

### Renaming of `build_fn` to `model`

SciKeras renamed the constructor argument `build_fn` to `model`. In a future release,
passing `build_fn` as a _keyword_ argument will raise a `TypeError`. Passing it as a positional
argument remains unchanged. You can make the following change to future proof your code:

```python
clf = KerasClassifier(build_fn=...)  # from
clf = KerasClassifier(model=...)  # to
```

### Default arguments in `build_fn`/`model`

SciKeras will no longer introspect your callable `model` for _user defined_ parameters. You
must now "declare" them as keyword arguments to the constructor if you want them to be
tunable parameters (i.e. settable via `set_params`):

```python
# From this:
def get_model(my_param=123):
    ...
clf = KerasClassifier(get_model)
# To this:
def get_model(my_param=123):  # Note: you can remove the default here to avoid duplication
    ...
clf = KerasClassifier(get_model, my_param=123)
```

That said, if you do not need them to work with `set_params` (which is only really
necessary if you are doing hyperparameter tuning), you do not need to make any changes.

## Documentation

Documentation is available on [ReadTheDocs](https://scikeras.readthedocs.io/en/latest/).

## Contributing

See [CONTRIBUTING.md](https://github.com/adriangb/scikeras/blob/master/CONTRIBUTING.md)
