# Sparse Inputs


SciKeras supports sparse inputs (`X`/features).
You don't have to do anything special for this to work, you can just pass a sparse matrix to `fit()`.

In this notebook, we'll demonstrate how this works and compare memory consumption of sparse inputs to dense inputs.


## Setup

```python
!pip install memory_profiler
%load_ext memory_profiler
```

```python
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import get_logger
get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="Setting the random state for TF")
```

```python
try:
    import scikeras
except ImportError:
    !python -m pip install scikeras
```

```python
import scipy
import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from tensorflow import keras
```

# Data

The dataset we'll be using is designed to demostrate a worst-case/best-case scenario for dense and sparse input features respectively.
It consists of a single categorical feature with equal number of categories as rows.
This means the one-hot encoded representation will require as many columns as it does rows, making it very ineffienct to store as a dense matrix but very efficient to store as a sparse matrix.

```python
N_SAMPLES = 20_000  # hand tuned to be ~4GB peak

X = np.arange(0, N_SAMPLES).reshape(-1, 1)
y = np.random.uniform(0, 1, size=(X.shape[0],))
```

## Model

The model here is nothing special, just a basic multilayer perceptron with one hidden layer.

```python
def get_clf(meta) -> keras.Model:
    n_features_in_ = meta["n_features_in_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    # a single hidden layer
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(1))
    return model
```

## Pipelines

Here is where it gets interesting.
We make two Scikit-Learn pipelines that use `OneHotEncoder`: one that uses `sparse=False` to force a dense matrix as the output and another that uses `sparse=True` (the default).

```python
dense_pipeline = Pipeline(
    [
        ("encoder", OneHotEncoder(sparse=False)),
        ("model", KerasRegressor(get_clf, loss="mse", epochs=5, verbose=False))
    ]
)

sparse_pipeline = Pipeline(
    [
        ("encoder", OneHotEncoder(sparse=True)),
        ("model", KerasRegressor(get_clf, loss="mse", epochs=5, verbose=False))
    ]
)
```

## Benchmark

Our benchmark will be to just train each one of these pipelines and measure peak memory consumption.

```python
%memit dense_pipeline.fit(X, y)
```

```python
%memit sparse_pipeline.fit(X, y)
```

You should see at least 100x more memory consumption **increment** in the dense pipeline.


## Bonus: dtypes

You might be able to save even more memory by changing the output dtype of `OneHotEncoder`.

```python
sparse_pipline_uint8 = Pipeline(
    [
        ("encoder", OneHotEncoder(sparse=True, dtype=np.uint8)),
        ("model", KerasRegressor(get_clf, loss="mse", epochs=5, verbose=False))
    ]
)
```

```python
%memit sparse_pipline_uint8.fit(X, y)
```
