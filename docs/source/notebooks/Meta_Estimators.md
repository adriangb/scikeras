---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #raw -->
<a href="https://colab.research.google.com/github/adriangb/scikeras/blob/docs-deploy/refs/heads/master/notebooks/Meta_Estimators.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>
<!-- #endraw -->

# Meta Estimators in SciKeras

In this notebook, we implement sklearn ensemble and tree meta-estimators backed by a Keras MLP model.

## Table of contents

* [1. Setup](#1.-Setup)
* [2. Defining the Keras Model](#2.-Defining-the-Keras-Model)
  * [2.1 Building a boosting ensemble](#2.1-Building-a-boosting-ensemble)
* [3. Testing with a toy dataset](#3.-Testing-with-a-toy-dataset)
* [4. Bagging ensemble](#4.-Bagging-ensemble)

## 1. Setup

```python
try:
    import scikeras
except ImportError:
    !python -m pip install scikeras[tensorflow]
```

Silence TensorFlow logging to keep output succinct.

```python
import warnings
from tensorflow import get_logger
get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="Setting the random state for TF")
```

```python
import numpy as np
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow import keras
```

## 2. Defining the Keras Model

We borrow our MLPClassifier implementation from the [MLPClassifier notebook](https://colab.research.google.com/github/adriangb/scikeras/blob/master/notebooks/MLPClassifier_and_MLPRegressor.ipynb).

```python
from typing import Dict, Iterable, Any


def get_clf_model(hidden_layer_sizes: Iterable[int], meta: Dict[str, Any], compile_kwargs: Dict[str, Any]):
    model = keras.Sequential()
    inp = keras.layers.Input(shape=(meta["n_features_in_"]))
    model.add(inp)
    for hidden_layer_size in hidden_layer_sizes:
        layer = keras.layers.Dense(hidden_layer_size, activation="relu")
        model.add(layer)
    if meta["target_type_"] == "binary":
        n_output_units = 1
        output_activation = "sigmoid"
        loss = "binary_crossentropy"
    elif meta["target_type_"] == "multiclass":
        n_output_units = meta["n_classes_"]
        output_activation = "softmax"
        loss = "sparse_categorical_crossentropy"
    else:
        raise NotImplementedError(f"Unsupported task type: {meta['target_type_']}")
    out = keras.layers.Dense(n_output_units, activation=output_activation)
    model.add(out)
    model.compile(loss=loss, optimizer=compile_kwargs["optimizer"])
    return model
```

Next we wrap this Keras model with SciKeras

```python
clf = KerasClassifier(
    model=get_clf_model,
    hidden_layer_sizes=(100, ),
    optimizer="adam",
    optimizer__learning_rate=0.001,
    verbose=0,
    random_state=0,
)
```

### 2.1 Building a boosting ensemble

Because SciKeras estimators are fully compliant with the Scikit-Learn API, we can make use of Scikit-Learn's built in utilities. In particular example, we will use `AdaBoostClassifier` from `sklearn.ensemble.AdaBoostClassifier`, but the process is the same for most Scikit-Learn meta-estimators.


```python
from sklearn.ensemble import AdaBoostClassifier


adaboost = AdaBoostClassifier(base_estimator=clf, random_state=0)
```

## 3. Testing with a toy dataset

Before continouing, we will run a small test to make sure we get somewhat reasonable results.


```python
from sklearn.datasets import make_moons


X, y = make_moons()

single_score = clf.fit(X, y).score(X, y)

adaboost_score = adaboost.fit(X, y).score(X, y)

print(f"Single score: {single_score:.2f}")
print(f"AdaBoost score: {adaboost_score:.2f}")
```

We see that the score for the AdaBoost classifier is slightly higher than that of an individual MLPRegressor instance. We can explore the individual classifiers, and see that each one is composed of a Keras Model with it's own individual weights.


```python
print(adaboost.estimators_[0].model_.get_weights()[0][0, :5])  # first sub-estimator
print(adaboost.estimators_[1].model_.get_weights()[0][0, :5])  # second sub-estimator
```

## 4. Bagging ensemble

For comparison, we run the same test with an ensemble built using `sklearn.ensemble.BaggingClassifier`.

```python
from sklearn.ensemble import BaggingClassifier


bagging = BaggingClassifier(base_estimator=clf, random_state=0, n_jobs=-1)

bagging_score = bagging.fit(X, y).score(X, y)

print(f"Bagging score: {bagging_score:.2f}")
```
